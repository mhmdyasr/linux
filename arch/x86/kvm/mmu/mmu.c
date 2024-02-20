// SPDX-License-Identifier: GPL-2.0-only
/*
 * Kernel-based Virtual Machine driver for Linux
 *
 * This module enables machines with Intel VT-x extensions to run virtual
 * machines without emulation or binary translation.
 *
 * MMU support
 *
 * Copyright (C) 2006 Qumranet, Inc.
 * Copyright 2010 Red Hat, Inc. and/or its affiliates.
 *
 * Authors:
 *   Yaniv Kamay  <yaniv@qumranet.com>
 *   Avi Kivity   <avi@qumranet.com>
 */
#define pr_fmt(fmt) KBUILD_MODNAME ": " fmt

#include "irq.h"
#include "ioapic.h"
#include "mmu.h"
#include "mmu_internal.h"
#include "tdp_mmu.h"
#include "x86.h"
#include "kvm_cache_regs.h"
#include "smm.h"
#include "kvm_emulate.h"
#include "page_track.h"
#include "cpuid.h"
#include "spte.h"

#include <linux/kvm_host.h>
#include <linux/types.h>
#include <linux/string.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/moduleparam.h>
#include <linux/export.h>
#include <linux/swap.h>
#include <linux/hugetlb.h>
#include <linux/compiler.h>
#include <linux/srcu.h>
#include <linux/slab.h>
#include <linux/sched/signal.h>
#include <linux/uaccess.h>
#include <linux/hash.h>
#include <linux/kern_levels.h>
#include <linux/kstrtox.h>
#include <linux/kthread.h>

#include <asm/page.h>
#include <asm/memtype.h>
#include <asm/cmpxchg.h>
#include <asm/io.h>
#include <asm/set_memory.h>
#include <asm/vmx.h>

#include "trace.h"

extern bool itlb_multihit_kvm_mitigation;

static bool nx_hugepage_mitigation_hard_disabled;

int __read_mostly nx_huge_pages = -1;
static uint __read_mostly nx_huge_pages_recovery_period_ms;
#ifdef CONFIG_PREEMPT_RT
/* Recovery can cause latency spikes, disable it for PREEMPT_RT.  */
static uint __read_mostly nx_huge_pages_recovery_ratio = 0;
#else
static uint __read_mostly nx_huge_pages_recovery_ratio = 60;
#endif

static int get_nx_huge_pages(char *buffer, const struct kernel_param *kp);
static int set_nx_huge_pages(const char *val, const struct kernel_param *kp);
static int set_nx_huge_pages_recovery_param(const char *val, const struct kernel_param *kp);

static const struct kernel_param_ops nx_huge_pages_ops = {
	.set = set_nx_huge_pages,
	.get = get_nx_huge_pages,
};

static const struct kernel_param_ops nx_huge_pages_recovery_param_ops = {
	.set = set_nx_huge_pages_recovery_param,
	.get = param_get_uint,
};

module_param_cb(nx_huge_pages, &nx_huge_pages_ops, &nx_huge_pages, 0644);
__MODULE_PARM_TYPE(nx_huge_pages, "bool");
module_param_cb(nx_huge_pages_recovery_ratio, &nx_huge_pages_recovery_param_ops,
		&nx_huge_pages_recovery_ratio, 0644);
__MODULE_PARM_TYPE(nx_huge_pages_recovery_ratio, "uint");
module_param_cb(nx_huge_pages_recovery_period_ms, &nx_huge_pages_recovery_param_ops,
		&nx_huge_pages_recovery_period_ms, 0644);
__MODULE_PARM_TYPE(nx_huge_pages_recovery_period_ms, "uint");

static bool __read_mostly force_flush_and_sync_on_reuse;
module_param_named(flush_on_reuse, force_flush_and_sync_on_reuse, bool, 0644);

/*
 * When setting this variable to true it enables Two-Dimensional-Paging
 * where the hardware walks 2 page tables:
 * 1. the guest-virtual to guest-physical
 * 2. while doing 1. it walks guest-physical to host-physical
 * If the hardware supports that we don't need to do shadow paging.
 */
bool tdp_enabled = false;

static bool __ro_after_init tdp_mmu_allowed;

#ifdef CONFIG_X86_64
bool __read_mostly tdp_mmu_enabled = true;
module_param_named(tdp_mmu, tdp_mmu_enabled, bool, 0444);
#endif

static int max_huge_page_level __read_mostly;
static int tdp_root_level __read_mostly;
static int max_tdp_level __read_mostly;

#define PTE_PREFETCH_NUM		8

#include <trace/events/kvm.h>

/* make pte_list_desc fit well in cache lines */
#define PTE_LIST_EXT 14

/*
 * struct pte_list_desc is the core data structure used to implement a custom
 * list for tracking a set of related SPTEs, e.g. all the SPTEs that map a
 * given GFN when used in the context of rmaps.  Using a custom list allows KVM
 * to optimize for the common case where many GFNs will have at most a handful
 * of SPTEs pointing at them, i.e. allows packing multiple SPTEs into a small
 * memory footprint, which in turn improves runtime performance by exploiting
 * cache locality.
 *
 * A list is comprised of one or more pte_list_desc objects (descriptors).
 * Each individual descriptor stores up to PTE_LIST_EXT SPTEs.  If a descriptor
 * is full and a new SPTEs needs to be added, a new descriptor is allocated and
 * becomes the head of the list.  This means that by definitions, all tail
 * descriptors are full.
 *
 * Note, the meta data fields are deliberately placed at the start of the
 * structure to optimize the cacheline layout; accessing the descriptor will
 * touch only a single cacheline so long as @spte_count<=6 (or if only the
 * descriptors metadata is accessed).
 */
struct pte_list_desc {
	struct pte_list_desc *more;
	/* The number of PTEs stored in _this_ descriptor. */
	u32 spte_count;
	/* The number of PTEs stored in all tails of this descriptor. */
	u32 tail_count;
	u64 *sptes[PTE_LIST_EXT];
};

struct kvm_shadow_walk_iterator {
	u64 addr;
	hpa_t shadow_addr;
	u64 *sptep;
	int level;
	unsigned index;
};

#define for_each_shadow_entry_using_root(_vcpu, _root, _addr, _walker)     \
	for (shadow_walk_init_using_root(&(_walker), (_vcpu),              \
					 (_root), (_addr));                \
	     shadow_walk_okay(&(_walker));			           \
	     shadow_walk_next(&(_walker)))

#define for_each_shadow_entry(_vcpu, _addr, _walker)            \
	for (shadow_walk_init(&(_walker), _vcpu, _addr);	\
	     shadow_walk_okay(&(_walker));			\
	     shadow_walk_next(&(_walker)))

#define for_each_shadow_entry_lockless(_vcpu, _addr, _walker, spte)	\
	for (shadow_walk_init(&(_walker), _vcpu, _addr);		\
	     shadow_walk_okay(&(_walker)) &&				\
		({ spte = mmu_spte_get_lockless(_walker.sptep); 1; });	\
	     __shadow_walk_next(&(_walker), spte))

static struct kmem_cache *pte_list_desc_cache;
struct kmem_cache *mmu_page_header_cache;
static struct percpu_counter kvm_total_used_mmu_pages;

static void mmu_spte_set(u64 *sptep, u64 spte);

struct kvm_mmu_role_regs {
	const unsigned long cr0;
	const unsigned long cr4;
	const u64 efer;
};

#define CREATE_TRACE_POINTS
#include "mmutrace.h"

/*
 * Yes, lot's of underscores.  They're a hint that you probably shouldn't be
 * reading from the role_regs.  Once the root_role is constructed, it becomes
 * the single source of truth for the MMU's state.
 */
#define BUILD_MMU_ROLE_REGS_ACCESSOR(reg, name, flag)			\
static inline bool __maybe_unused					\
____is_##reg##_##name(const struct kvm_mmu_role_regs *regs)		\
{									\
	return !!(regs->reg & flag);					\
}
BUILD_MMU_ROLE_REGS_ACCESSOR(cr0, pg, X86_CR0_PG);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr0, wp, X86_CR0_WP);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr4, pse, X86_CR4_PSE);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr4, pae, X86_CR4_PAE);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr4, smep, X86_CR4_SMEP);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr4, smap, X86_CR4_SMAP);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr4, pke, X86_CR4_PKE);
BUILD_MMU_ROLE_REGS_ACCESSOR(cr4, la57, X86_CR4_LA57);
BUILD_MMU_ROLE_REGS_ACCESSOR(efer, nx, EFER_NX);
BUILD_MMU_ROLE_REGS_ACCESSOR(efer, lma, EFER_LMA);

/*
 * The MMU itself (with a valid role) is the single source of truth for the
 * MMU.  Do not use the regs used to build the MMU/role, nor the vCPU.  The
 * regs don't account for dependencies, e.g. clearing CR4 bits if CR0.PG=1,
 * and the vCPU may be incorrect/irrelevant.
 */
#define BUILD_MMU_ROLE_ACCESSOR(base_or_ext, reg, name)		\
static inline bool __maybe_unused is_##reg##_##name(struct kvm_mmu *mmu)	\
{								\
	return !!(mmu->cpu_role. base_or_ext . reg##_##name);	\
}
BUILD_MMU_ROLE_ACCESSOR(base, cr0, wp);
BUILD_MMU_ROLE_ACCESSOR(ext,  cr4, pse);
BUILD_MMU_ROLE_ACCESSOR(ext,  cr4, smep);
BUILD_MMU_ROLE_ACCESSOR(ext,  cr4, smap);
BUILD_MMU_ROLE_ACCESSOR(ext,  cr4, pke);
BUILD_MMU_ROLE_ACCESSOR(ext,  cr4, la57);
BUILD_MMU_ROLE_ACCESSOR(base, efer, nx);
BUILD_MMU_ROLE_ACCESSOR(ext,  efer, lma);

static inline bool is_cr0_pg(struct kvm_mmu *mmu)
{
        return mmu->cpu_role.base.level > 0;
}

static inline bool is_cr4_pae(struct kvm_mmu *mmu)
{
        return !mmu->cpu_role.base.has_4_byte_gpte;
}

static struct kvm_mmu_role_regs vcpu_to_role_regs(struct kvm_vcpu *vcpu)
{
	struct kvm_mmu_role_regs regs = {
		.cr0 = kvm_read_cr0_bits(vcpu, KVM_MMU_CR0_ROLE_BITS),
		.cr4 = kvm_read_cr4_bits(vcpu, KVM_MMU_CR4_ROLE_BITS),
		.efer = vcpu->arch.efer,
	};

	return regs;
}

static unsigned long get_guest_cr3(struct kvm_vcpu *vcpu)
{
	return kvm_read_cr3(vcpu);
}

static inline unsigned long kvm_mmu_get_guest_pgd(struct kvm_vcpu *vcpu,
						  struct kvm_mmu *mmu)
{
	if (IS_ENABLED(CONFIG_RETPOLINE) && mmu->get_guest_pgd == get_guest_cr3)
		return kvm_read_cr3(vcpu);

	return mmu->get_guest_pgd(vcpu);
}

static inline bool kvm_available_flush_remote_tlbs_range(void)
{
#if IS_ENABLED(CONFIG_HYPERV)
	return kvm_x86_ops.flush_remote_tlbs_range;
#else
	return false;
#endif
}

static gfn_t kvm_mmu_page_get_gfn(struct kvm_mmu_page *sp, int index);

/* Flush the range of guest memory mapped by the given SPTE. */
static void kvm_flush_remote_tlbs_sptep(struct kvm *kvm, u64 *sptep)
{
	struct kvm_mmu_page *sp = sptep_to_sp(sptep);
	gfn_t gfn = kvm_mmu_page_get_gfn(sp, spte_index(sptep));

	kvm_flush_remote_tlbs_gfn(kvm, gfn, sp->role.level);
}

static void mark_mmio_spte(struct kvm_vcpu *vcpu, u64 *sptep, u64 gfn,
			   unsigned int access)
{
	u64 spte = make_mmio_spte(vcpu, gfn, access);

	trace_mark_mmio_spte(sptep, gfn, spte);
	mmu_spte_set(sptep, spte);
}

static gfn_t get_mmio_spte_gfn(u64 spte)
{
	u64 gpa = spte & shadow_nonpresent_or_rsvd_lower_gfn_mask;

	gpa |= (spte >> SHADOW_NONPRESENT_OR_RSVD_MASK_LEN)
	       & shadow_nonpresent_or_rsvd_mask;

	return gpa >> PAGE_SHIFT;
}

static unsigned get_mmio_spte_access(u64 spte)
{
	return spte & shadow_mmio_access_mask;
}

static bool check_mmio_spte(struct kvm_vcpu *vcpu, u64 spte)
{
	u64 kvm_gen, spte_gen, gen;

	gen = kvm_vcpu_memslots(vcpu)->generation;
	if (unlikely(gen & KVM_MEMSLOT_GEN_UPDATE_IN_PROGRESS))
		return false;

	kvm_gen = gen & MMIO_SPTE_GEN_MASK;
	spte_gen = get_mmio_spte_generation(spte);

	trace_check_mmio_spte(spte, kvm_gen, spte_gen);
	return likely(kvm_gen == spte_gen);
}

static int is_cpuid_PSE36(void)
{
	return 1;
}

#ifdef CONFIG_X86_64
static void __set_spte(u64 *sptep, u64 spte)
{
	WRITE_ONCE(*sptep, spte);
}

static void __update_clear_spte_fast(u64 *sptep, u64 spte)
{
	WRITE_ONCE(*sptep, spte);
}

static u64 __update_clear_spte_slow(u64 *sptep, u64 spte)
{
	return xchg(sptep, spte);
}

static u64 __get_spte_lockless(u64 *sptep)
{
	return READ_ONCE(*sptep);
}
#else
union split_spte {
	struct {
		u32 spte_low;
		u32 spte_high;
	};
	u64 spte;
};

static void count_spte_clear(u64 *sptep, u64 spte)
{
	struct kvm_mmu_page *sp =  sptep_to_sp(sptep);

	if (is_shadow_present_pte(spte))
		return;

	/* Ensure the spte is completely set before we increase the count */
	smp_wmb();
	sp->clear_spte_count++;
}

static void __set_spte(u64 *sptep, u64 spte)
{
	union split_spte *ssptep, sspte;

	ssptep = (union split_spte *)sptep;
	sspte = (union split_spte)spte;

	ssptep->spte_high = sspte.spte_high;

	/*
	 * If we map the spte from nonpresent to present, We should store
	 * the high bits firstly, then set present bit, so cpu can not
	 * fetch this spte while we are setting the spte.
	 */
	smp_wmb();

	WRITE_ONCE(ssptep->spte_low, sspte.spte_low);
}

static void __update_clear_spte_fast(u64 *sptep, u64 spte)
{
	union split_spte *ssptep, sspte;

	ssptep = (union split_spte *)sptep;
	sspte = (union split_spte)spte;

	WRITE_ONCE(ssptep->spte_low, sspte.spte_low);

	/*
	 * If we map the spte from present to nonpresent, we should clear
	 * present bit firstly to avoid vcpu fetch the old high bits.
	 */
	smp_wmb();

	ssptep->spte_high = sspte.spte_high;
	count_spte_clear(sptep, spte);
}

static u64 __update_clear_spte_slow(u64 *sptep, u64 spte)
{
	union split_spte *ssptep, sspte, orig;

	ssptep = (union split_spte *)sptep;
	sspte = (union split_spte)spte;

	/* xchg acts as a barrier before the setting of the high bits */
	orig.spte_low = xchg(&ssptep->spte_low, sspte.spte_low);
	orig.spte_high = ssptep->spte_high;
	ssptep->spte_high = sspte.spte_high;
	count_spte_clear(sptep, spte);

	return orig.spte;
}

/*
 * The idea using the light way get the spte on x86_32 guest is from
 * gup_get_pte (mm/gup.c).
 *
 * An spte tlb flush may be pending, because kvm_set_pte_rmap
 * coalesces them and we are running out of the MMU lock.  Therefore
 * we need to protect against in-progress updates of the spte.
 *
 * Reading the spte while an update is in progress may get the old value
 * for the high part of the spte.  The race is fine for a present->non-present
 * change (because the high part of the spte is ignored for non-present spte),
 * but for a present->present change we must reread the spte.
 *
 * All such changes are done in two steps (present->non-present and
 * non-present->present), hence it is enough to count the number of
 * present->non-present updates: if it changed while reading the spte,
 * we might have hit the race.  This is done using clear_spte_count.
 */
static u64 __get_spte_lockless(u64 *sptep)
{
	struct kvm_mmu_page *sp =  sptep_to_sp(sptep);
	union split_spte spte, *orig = (union split_spte *)sptep;
	int count;

retry:
	count = sp->clear_spte_count;
	smp_rmb();

	spte.spte_low = orig->spte_low;
	smp_rmb();

	spte.spte_high = orig->spte_high;
	smp_rmb();

	if (unlikely(spte.spte_low != orig->spte_low ||
	      count != sp->clear_spte_count))
		goto retry;

	return spte.spte;
}
#endif

/* Rules for using mmu_spte_set:
 * Set the sptep from nonpresent to present.
 * Note: the sptep being assigned *must* be either not present
 * or in a state where the hardware will not attempt to update
 * the spte.
 */
static void mmu_spte_set(u64 *sptep, u64 new_spte)
{
	WARN_ON_ONCE(is_shadow_present_pte(*sptep));
	__set_spte(sptep, new_spte);
}

/*
 * Update the SPTE (excluding the PFN), but do not track changes in its
 * accessed/dirty status.
 */
static u64 mmu_spte_update_no_track(u64 *sptep, u64 new_spte)
{
	u64 old_spte = *sptep;

	WARN_ON_ONCE(!is_shadow_present_pte(new_spte));
	check_spte_writable_invariants(new_spte);

	if (!is_shadow_present_pte(old_spte)) {
		mmu_spte_set(sptep, new_spte);
		return old_spte;
	}

	if (!spte_has_volatile_bits(old_spte))
		__update_clear_spte_fast(sptep, new_spte);
	else
		old_spte = __update_clear_spte_slow(sptep, new_spte);

	WARN_ON_ONCE(spte_to_pfn(old_spte) != spte_to_pfn(new_spte));

	return old_spte;
}

/* Rules for using mmu_spte_update:
 * Update the state bits, it means the mapped pfn is not changed.
 *
 * Whenever an MMU-writable SPTE is overwritten with a read-only SPTE, remote
 * TLBs must be flushed. Otherwise rmap_write_protect will find a read-only
 * spte, even though the writable spte might be cached on a CPU's TLB.
 *
 * Returns true if the TLB needs to be flushed
 */
static bool mmu_spte_update(u64 *sptep, u64 new_spte)
{
	bool flush = false;
	u64 old_spte = mmu_spte_update_no_track(sptep, new_spte);

	if (!is_shadow_present_pte(old_spte))
		return false;

	/*
	 * For the spte updated out of mmu-lock is safe, since
	 * we always atomically update it, see the comments in
	 * spte_has_volatile_bits().
	 */
	if (is_mmu_writable_spte(old_spte) &&
	      !is_writable_pte(new_spte))
		flush = true;

	/*
	 * Flush TLB when accessed/dirty states are changed in the page tables,
	 * to guarantee consistency between TLB and page tables.
	 */

	if (is_accessed_spte(old_spte) && !is_accessed_spte(new_spte)) {
		flush = true;
		kvm_set_pfn_accessed(spte_to_pfn(old_spte));
	}

	if (is_dirty_spte(old_spte) && !is_dirty_spte(new_spte)) {
		flush = true;
		kvm_set_pfn_dirty(spte_to_pfn(old_spte));
	}

	return flush;
}

/*
 * Rules for using mmu_spte_clear_track_bits:
 * It sets the sptep from present to nonpresent, and track the
 * state bits, it is used to clear the last level sptep.
 * Returns the old PTE.
 */
static u64 mmu_spte_clear_track_bits(struct kvm *kvm, u64 *sptep)
{
	kvm_pfn_t pfn;
	u64 old_spte = *sptep;
	int level = sptep_to_sp(sptep)->role.level;
	struct page *page;

	if (!is_shadow_present_pte(old_spte) ||
	    !spte_has_volatile_bits(old_spte))
		__update_clear_spte_fast(sptep, 0ull);
	else
		old_spte = __update_clear_spte_slow(sptep, 0ull);

	if (!is_shadow_present_pte(old_spte))
		return old_spte;

	kvm_update_page_stats(kvm, level, -1);

	pfn = spte_to_pfn(old_spte);

	/*
	 * KVM doesn't hold a reference to any pages mapped into the guest, and
	 * instead uses the mmu_notifier to ensure that KVM unmaps any pages
	 * before they are reclaimed.  Sanity check that, if the pfn is backed
	 * by a refcounted page, the refcount is elevated.
	 */
	page = kvm_pfn_to_refcounted_page(pfn);
	WARN_ON_ONCE(page && !page_count(page));

	if (is_accessed_spte(old_spte))
		kvm_set_pfn_accessed(pfn);

	if (is_dirty_spte(old_spte))
		kvm_set_pfn_dirty(pfn);

	return old_spte;
}

/*
 * Rules for using mmu_spte_clear_no_track:
 * Directly clear spte without caring the state bits of sptep,
 * it is used to set the upper level spte.
 */
static void mmu_spte_clear_no_track(u64 *sptep)
{
	__update_clear_spte_fast(sptep, 0ull);
}

static u64 mmu_spte_get_lockless(u64 *sptep)
{
	return __get_spte_lockless(sptep);
}

/* Returns the Accessed status of the PTE and resets it at the same time. */
static bool mmu_spte_age(u64 *sptep)
{
	u64 spte = mmu_spte_get_lockless(sptep);

	if (!is_accessed_spte(spte))
		return false;

	if (spte_ad_enabled(spte)) {
		clear_bit((ffs(shadow_accessed_mask) - 1),
			  (unsigned long *)sptep);
	} else {
		/*
		 * Capture the dirty status of the page, so that it doesn't get
		 * lost when the SPTE is marked for access tracking.
		 */
		if (is_writable_pte(spte))
			kvm_set_pfn_dirty(spte_to_pfn(spte));

		spte = mark_spte_for_access_track(spte);
		mmu_spte_update_no_track(sptep, spte);
	}

	return true;
}

static inline bool is_tdp_mmu_active(struct kvm_vcpu *vcpu)
{
	return tdp_mmu_enabled && vcpu->arch.mmu->root_role.direct;
}

static void walk_shadow_page_lockless_begin(struct kvm_vcpu *vcpu)
{
	if (is_tdp_mmu_active(vcpu)) {
		kvm_tdp_mmu_walk_lockless_begin();
	} else {
		/*
		 * Prevent page table teardown by making any free-er wait during
		 * kvm_flush_remote_tlbs() IPI to all active vcpus.
		 */
		local_irq_disable();

		/*
		 * Make sure a following spte read is not reordered ahead of the write
		 * to vcpu->mode.
		 */
		smp_store_mb(vcpu->mode, READING_SHADOW_PAGE_TABLES);
	}
}

static void walk_shadow_page_lockless_end(struct kvm_vcpu *vcpu)
{
	if (is_tdp_mmu_active(vcpu)) {
		kvm_tdp_mmu_walk_lockless_end();
	} else {
		/*
		 * Make sure the write to vcpu->mode is not reordered in front of
		 * reads to sptes.  If it does, kvm_mmu_commit_zap_page() can see us
		 * OUTSIDE_GUEST_MODE and proceed to free the shadow page table.
		 */
		smp_store_release(&vcpu->mode, OUTSIDE_GUEST_MODE);
		local_irq_enable();
	}
}

static int mmu_topup_memory_caches(struct kvm_vcpu *vcpu, bool maybe_indirect)
{
	int r;

	/* 1 rmap, 1 parent PTE per level, and the prefetched rmaps. */
	r = kvm_mmu_topup_memory_cache(&vcpu->arch.mmu_pte_list_desc_cache,
				       1 + PT64_ROOT_MAX_LEVEL + PTE_PREFETCH_NUM);
	if (r)
		return r;
	r = kvm_mmu_topup_memory_cache(&vcpu->arch.mmu_shadow_page_cache,
				       PT64_ROOT_MAX_LEVEL);
	if (r)
		return r;
	if (maybe_indirect) {
		r = kvm_mmu_topup_memory_cache(&vcpu->arch.mmu_shadowed_info_cache,
					       PT64_ROOT_MAX_LEVEL);
		if (r)
			return r;
	}
	return kvm_mmu_topup_memory_cache(&vcpu->arch.mmu_page_header_cache,
					  PT64_ROOT_MAX_LEVEL);
}

static void mmu_free_memory_caches(struct kvm_vcpu *vcpu)
{
	kvm_mmu_free_memory_cache(&vcpu->arch.mmu_pte_list_desc_cache);
	kvm_mmu_free_memory_cache(&vcpu->arch.mmu_shadow_page_cache);
	kvm_mmu_free_memory_cache(&vcpu->arch.mmu_shadowed_info_cache);
	kvm_mmu_free_memory_cache(&vcpu->arch.mmu_page_header_cache);
}

static void mmu_free_pte_list_desc(struct pte_list_desc *pte_list_desc)
{
	kmem_cache_free(pte_list_desc_cache, pte_list_desc);
}

static bool sp_has_gptes(struct kvm_mmu_page *sp);

static gfn_t kvm_mmu_page_get_gfn(struct kvm_mmu_page *sp, int index)
{
	if (sp->role.passthrough)
		return sp->gfn;

	if (!sp->role.direct)
		return sp->shadowed_translation[index] >> PAGE_SHIFT;

	return sp->gfn + (index << ((sp->role.level - 1) * SPTE_LEVEL_BITS));
}

/*
 * For leaf SPTEs, fetch the *guest* access permissions being shadowed. Note
 * that the SPTE itself may have a more constrained access permissions that
 * what the guest enforces. For example, a guest may create an executable
 * huge PTE but KVM may disallow execution to mitigate iTLB multihit.
 */
static u32 kvm_mmu_page_get_access(struct kvm_mmu_page *sp, int index)
{
	if (sp_has_gptes(sp))
		return sp->shadowed_translation[index] & ACC_ALL;

	/*
	 * For direct MMUs (e.g. TDP or non-paging guests) or passthrough SPs,
	 * KVM is not shadowing any guest page tables, so the "guest access
	 * permissions" are just ACC_ALL.
	 *
	 * For direct SPs in indirect MMUs (shadow paging), i.e. when KVM
	 * is shadowing a guest huge page with small pages, the guest access
	 * permissions being shadowed are the access permissions of the huge
	 * page.
	 *
	 * In both cases, sp->role.access contains the correct access bits.
	 */
	return sp->role.access;
}

static void kvm_mmu_page_set_translation(struct kvm_mmu_page *sp, int index,
					 gfn_t gfn, unsigned int access)
{
	if (sp_has_gptes(sp)) {
		sp->shadowed_translation[index] = (gfn << PAGE_SHIFT) | access;
		return;
	}

	WARN_ONCE(access != kvm_mmu_page_get_access(sp, index),
	          "access mismatch under %s page %llx (expected %u, got %u)\n",
	          sp->role.passthrough ? "passthrough" : "direct",
	          sp->gfn, kvm_mmu_page_get_access(sp, index), access);

	WARN_ONCE(gfn != kvm_mmu_page_get_gfn(sp, index),
	          "gfn mismatch under %s page %llx (expected %llx, got %llx)\n",
	          sp->role.passthrough ? "passthrough" : "direct",
	          sp->gfn, kvm_mmu_page_get_gfn(sp, index), gfn);
}

static void kvm_mmu_page_set_access(struct kvm_mmu_page *sp, int index,
				    unsigned int access)
{
	gfn_t gfn = kvm_mmu_page_get_gfn(sp, index);

	kvm_mmu_page_set_translation(sp, index, gfn, access);
}

/*
 * Return the pointer to the large page information for a given gfn,
 * handling slots that are not large page aligned.
 */
static struct kvm_lpage_info *lpage_info_slot(gfn_t gfn,
		const struct kvm_memory_slot *slot, int level)
{
	unsigned long idx;

	idx = gfn_to_index(gfn, slot->base_gfn, level);
	return &slot->arch.lpage_info[level - 2][idx];
}

/*
 * The most significant bit in disallow_lpage tracks whether or not memory
 * attributes are mixed, i.e. not identical for all gfns at the current level.
 * The lower order bits are used to refcount other cases where a hugepage is
 * disallowed, e.g. if KVM has shadow a page table at the gfn.
 */
#define KVM_LPAGE_MIXED_FLAG	BIT(31)

static void update_gfn_disallow_lpage_count(const struct kvm_memory_slot *slot,
					    gfn_t gfn, int count)
{
	struct kvm_lpage_info *linfo;
	int old, i;

	for (i = PG_LEVEL_2M; i <= KVM_MAX_HUGEPAGE_LEVEL; ++i) {
		linfo = lpage_info_slot(gfn, slot, i);

		old = linfo->disallow_lpage;
		linfo->disallow_lpage += count;
		WARN_ON_ONCE((old ^ linfo->disallow_lpage) & KVM_LPAGE_MIXED_FLAG);
	}
}

void kvm_mmu_gfn_disallow_lpage(const struct kvm_memory_slot *slot, gfn_t gfn)
{
	update_gfn_disallow_lpage_count(slot, gfn, 1);
}

void kvm_mmu_gfn_allow_lpage(const struct kvm_memory_slot *slot, gfn_t gfn)
{
	update_gfn_disallow_lpage_count(slot, gfn, -1);
}

static void account_shadowed(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	struct kvm_memslots *slots;
	struct kvm_memory_slot *slot;
	gfn_t gfn;

	kvm->arch.indirect_shadow_pages++;
	gfn = sp->gfn;
	slots = kvm_memslots_for_spte_role(kvm, sp->role);
	slot = __gfn_to_memslot(slots, gfn);

	/* the non-leaf shadow pages are keeping readonly. */
	if (sp->role.level > PG_LEVEL_4K)
		return __kvm_write_track_add_gfn(kvm, slot, gfn);

	kvm_mmu_gfn_disallow_lpage(slot, gfn);

	if (kvm_mmu_slot_gfn_write_protect(kvm, slot, gfn, PG_LEVEL_4K))
		kvm_flush_remote_tlbs_gfn(kvm, gfn, PG_LEVEL_4K);
}

void track_possible_nx_huge_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	/*
	 * If it's possible to replace the shadow page with an NX huge page,
	 * i.e. if the shadow page is the only thing currently preventing KVM
	 * from using a huge page, add the shadow page to the list of "to be
	 * zapped for NX recovery" pages.  Note, the shadow page can already be
	 * on the list if KVM is reusing an existing shadow page, i.e. if KVM
	 * links a shadow page at multiple points.
	 */
	if (!list_empty(&sp->possible_nx_huge_page_link))
		return;

	++kvm->stat.nx_lpage_splits;
	list_add_tail(&sp->possible_nx_huge_page_link,
		      &kvm->arch.possible_nx_huge_pages);
}

static void account_nx_huge_page(struct kvm *kvm, struct kvm_mmu_page *sp,
				 bool nx_huge_page_possible)
{
	sp->nx_huge_page_disallowed = true;

	if (nx_huge_page_possible)
		track_possible_nx_huge_page(kvm, sp);
}

static void unaccount_shadowed(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	struct kvm_memslots *slots;
	struct kvm_memory_slot *slot;
	gfn_t gfn;

	kvm->arch.indirect_shadow_pages--;
	gfn = sp->gfn;
	slots = kvm_memslots_for_spte_role(kvm, sp->role);
	slot = __gfn_to_memslot(slots, gfn);
	if (sp->role.level > PG_LEVEL_4K)
		return __kvm_write_track_remove_gfn(kvm, slot, gfn);

	kvm_mmu_gfn_allow_lpage(slot, gfn);
}

void untrack_possible_nx_huge_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	if (list_empty(&sp->possible_nx_huge_page_link))
		return;

	--kvm->stat.nx_lpage_splits;
	list_del_init(&sp->possible_nx_huge_page_link);
}

static void unaccount_nx_huge_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	sp->nx_huge_page_disallowed = false;

	untrack_possible_nx_huge_page(kvm, sp);
}

static struct kvm_memory_slot *gfn_to_memslot_dirty_bitmap(struct kvm_vcpu *vcpu,
							   gfn_t gfn,
							   bool no_dirty_log)
{
	struct kvm_memory_slot *slot;

	slot = kvm_vcpu_gfn_to_memslot(vcpu, gfn);
	if (!slot || slot->flags & KVM_MEMSLOT_INVALID)
		return NULL;
	if (no_dirty_log && kvm_slot_dirty_track_enabled(slot))
		return NULL;

	return slot;
}

/*
 * About rmap_head encoding:
 *
 * If the bit zero of rmap_head->val is clear, then it points to the only spte
 * in this rmap chain. Otherwise, (rmap_head->val & ~1) points to a struct
 * pte_list_desc containing more mappings.
 */

/*
 * Returns the number of pointers in the rmap chain, not counting the new one.
 */
static int pte_list_add(struct kvm_mmu_memory_cache *cache, u64 *spte,
			struct kvm_rmap_head *rmap_head)
{
	struct pte_list_desc *desc;
	int count = 0;

	if (!rmap_head->val) {
		rmap_head->val = (unsigned long)spte;
	} else if (!(rmap_head->val & 1)) {
		desc = kvm_mmu_memory_cache_alloc(cache);
		desc->sptes[0] = (u64 *)rmap_head->val;
		desc->sptes[1] = spte;
		desc->spte_count = 2;
		desc->tail_count = 0;
		rmap_head->val = (unsigned long)desc | 1;
		++count;
	} else {
		desc = (struct pte_list_desc *)(rmap_head->val & ~1ul);
		count = desc->tail_count + desc->spte_count;

		/*
		 * If the previous head is full, allocate a new head descriptor
		 * as tail descriptors are always kept full.
		 */
		if (desc->spte_count == PTE_LIST_EXT) {
			desc = kvm_mmu_memory_cache_alloc(cache);
			desc->more = (struct pte_list_desc *)(rmap_head->val & ~1ul);
			desc->spte_count = 0;
			desc->tail_count = count;
			rmap_head->val = (unsigned long)desc | 1;
		}
		desc->sptes[desc->spte_count++] = spte;
	}
	return count;
}

static void pte_list_desc_remove_entry(struct kvm *kvm,
				       struct kvm_rmap_head *rmap_head,
				       struct pte_list_desc *desc, int i)
{
	struct pte_list_desc *head_desc = (struct pte_list_desc *)(rmap_head->val & ~1ul);
	int j = head_desc->spte_count - 1;

	/*
	 * The head descriptor should never be empty.  A new head is added only
	 * when adding an entry and the previous head is full, and heads are
	 * removed (this flow) when they become empty.
	 */
	KVM_BUG_ON_DATA_CORRUPTION(j < 0, kvm);

	/*
	 * Replace the to-be-freed SPTE with the last valid entry from the head
	 * descriptor to ensure that tail descriptors are full at all times.
	 * Note, this also means that tail_count is stable for each descriptor.
	 */
	desc->sptes[i] = head_desc->sptes[j];
	head_desc->sptes[j] = NULL;
	head_desc->spte_count--;
	if (head_desc->spte_count)
		return;

	/*
	 * The head descriptor is empty.  If there are no tail descriptors,
	 * nullify the rmap head to mark the list as empty, else point the rmap
	 * head at the next descriptor, i.e. the new head.
	 */
	if (!head_desc->more)
		rmap_head->val = 0;
	else
		rmap_head->val = (unsigned long)head_desc->more | 1;
	mmu_free_pte_list_desc(head_desc);
}

static void pte_list_remove(struct kvm *kvm, u64 *spte,
			    struct kvm_rmap_head *rmap_head)
{
	struct pte_list_desc *desc;
	int i;

	if (KVM_BUG_ON_DATA_CORRUPTION(!rmap_head->val, kvm))
		return;

	if (!(rmap_head->val & 1)) {
		if (KVM_BUG_ON_DATA_CORRUPTION((u64 *)rmap_head->val != spte, kvm))
			return;

		rmap_head->val = 0;
	} else {
		desc = (struct pte_list_desc *)(rmap_head->val & ~1ul);
		while (desc) {
			for (i = 0; i < desc->spte_count; ++i) {
				if (desc->sptes[i] == spte) {
					pte_list_desc_remove_entry(kvm, rmap_head,
								   desc, i);
					return;
				}
			}
			desc = desc->more;
		}

		KVM_BUG_ON_DATA_CORRUPTION(true, kvm);
	}
}

static void kvm_zap_one_rmap_spte(struct kvm *kvm,
				  struct kvm_rmap_head *rmap_head, u64 *sptep)
{
	mmu_spte_clear_track_bits(kvm, sptep);
	pte_list_remove(kvm, sptep, rmap_head);
}

/* Return true if at least one SPTE was zapped, false otherwise */
static bool kvm_zap_all_rmap_sptes(struct kvm *kvm,
				   struct kvm_rmap_head *rmap_head)
{
	struct pte_list_desc *desc, *next;
	int i;

	if (!rmap_head->val)
		return false;

	if (!(rmap_head->val & 1)) {
		mmu_spte_clear_track_bits(kvm, (u64 *)rmap_head->val);
		goto out;
	}

	desc = (struct pte_list_desc *)(rmap_head->val & ~1ul);

	for (; desc; desc = next) {
		for (i = 0; i < desc->spte_count; i++)
			mmu_spte_clear_track_bits(kvm, desc->sptes[i]);
		next = desc->more;
		mmu_free_pte_list_desc(desc);
	}
out:
	/* rmap_head is meaningless now, remember to reset it */
	rmap_head->val = 0;
	return true;
}

unsigned int pte_list_count(struct kvm_rmap_head *rmap_head)
{
	struct pte_list_desc *desc;

	if (!rmap_head->val)
		return 0;
	else if (!(rmap_head->val & 1))
		return 1;

	desc = (struct pte_list_desc *)(rmap_head->val & ~1ul);
	return desc->tail_count + desc->spte_count;
}

static struct kvm_rmap_head *gfn_to_rmap(gfn_t gfn, int level,
					 const struct kvm_memory_slot *slot)
{
	unsigned long idx;

	idx = gfn_to_index(gfn, slot->base_gfn, level);
	return &slot->arch.rmap[level - PG_LEVEL_4K][idx];
}

static void rmap_remove(struct kvm *kvm, u64 *spte)
{
	struct kvm_memslots *slots;
	struct kvm_memory_slot *slot;
	struct kvm_mmu_page *sp;
	gfn_t gfn;
	struct kvm_rmap_head *rmap_head;

	sp = sptep_to_sp(spte);
	gfn = kvm_mmu_page_get_gfn(sp, spte_index(spte));

	/*
	 * Unlike rmap_add, rmap_remove does not run in the context of a vCPU
	 * so we have to determine which memslots to use based on context
	 * information in sp->role.
	 */
	slots = kvm_memslots_for_spte_role(kvm, sp->role);

	slot = __gfn_to_memslot(slots, gfn);
	rmap_head = gfn_to_rmap(gfn, sp->role.level, slot);

	pte_list_remove(kvm, spte, rmap_head);
}

/*
 * Used by the following functions to iterate through the sptes linked by a
 * rmap.  All fields are private and not assumed to be used outside.
 */
struct rmap_iterator {
	/* private fields */
	struct pte_list_desc *desc;	/* holds the sptep if not NULL */
	int pos;			/* index of the sptep */
};

/*
 * Iteration must be started by this function.  This should also be used after
 * removing/dropping sptes from the rmap link because in such cases the
 * information in the iterator may not be valid.
 *
 * Returns sptep if found, NULL otherwise.
 */
static u64 *rmap_get_first(struct kvm_rmap_head *rmap_head,
			   struct rmap_iterator *iter)
{
	u64 *sptep;

	if (!rmap_head->val)
		return NULL;

	if (!(rmap_head->val & 1)) {
		iter->desc = NULL;
		sptep = (u64 *)rmap_head->val;
		goto out;
	}

	iter->desc = (struct pte_list_desc *)(rmap_head->val & ~1ul);
	iter->pos = 0;
	sptep = iter->desc->sptes[iter->pos];
out:
	BUG_ON(!is_shadow_present_pte(*sptep));
	return sptep;
}

/*
 * Must be used with a valid iterator: e.g. after rmap_get_first().
 *
 * Returns sptep if found, NULL otherwise.
 */
static u64 *rmap_get_next(struct rmap_iterator *iter)
{
	u64 *sptep;

	if (iter->desc) {
		if (iter->pos < PTE_LIST_EXT - 1) {
			++iter->pos;
			sptep = iter->desc->sptes[iter->pos];
			if (sptep)
				goto out;
		}

		iter->desc = iter->desc->more;

		if (iter->desc) {
			iter->pos = 0;
			/* desc->sptes[0] cannot be NULL */
			sptep = iter->desc->sptes[iter->pos];
			goto out;
		}
	}

	return NULL;
out:
	BUG_ON(!is_shadow_present_pte(*sptep));
	return sptep;
}

#define for_each_rmap_spte(_rmap_head_, _iter_, _spte_)			\
	for (_spte_ = rmap_get_first(_rmap_head_, _iter_);		\
	     _spte_; _spte_ = rmap_get_next(_iter_))

static void drop_spte(struct kvm *kvm, u64 *sptep)
{
	u64 old_spte = mmu_spte_clear_track_bits(kvm, sptep);

	if (is_shadow_present_pte(old_spte))
		rmap_remove(kvm, sptep);
}

static void drop_large_spte(struct kvm *kvm, u64 *sptep, bool flush)
{
	struct kvm_mmu_page *sp;

	sp = sptep_to_sp(sptep);
	WARN_ON_ONCE(sp->role.level == PG_LEVEL_4K);

	drop_spte(kvm, sptep);

	if (flush)
		kvm_flush_remote_tlbs_sptep(kvm, sptep);
}

/*
 * Write-protect on the specified @sptep, @pt_protect indicates whether
 * spte write-protection is caused by protecting shadow page table.
 *
 * Note: write protection is difference between dirty logging and spte
 * protection:
 * - for dirty logging, the spte can be set to writable at anytime if
 *   its dirty bitmap is properly set.
 * - for spte protection, the spte can be writable only after unsync-ing
 *   shadow page.
 *
 * Return true if tlb need be flushed.
 */
static bool spte_write_protect(u64 *sptep, bool pt_protect)
{
	u64 spte = *sptep;

	if (!is_writable_pte(spte) &&
	    !(pt_protect && is_mmu_writable_spte(spte)))
		return false;

	if (pt_protect)
		spte &= ~shadow_mmu_writable_mask;
	spte = spte & ~PT_WRITABLE_MASK;

	return mmu_spte_update(sptep, spte);
}

static bool rmap_write_protect(struct kvm_rmap_head *rmap_head,
			       bool pt_protect)
{
	u64 *sptep;
	struct rmap_iterator iter;
	bool flush = false;

	for_each_rmap_spte(rmap_head, &iter, sptep)
		flush |= spte_write_protect(sptep, pt_protect);

	return flush;
}

static bool spte_clear_dirty(u64 *sptep)
{
	u64 spte = *sptep;

	KVM_MMU_WARN_ON(!spte_ad_enabled(spte));
	spte &= ~shadow_dirty_mask;
	return mmu_spte_update(sptep, spte);
}

static bool spte_wrprot_for_clear_dirty(u64 *sptep)
{
	bool was_writable = test_and_clear_bit(PT_WRITABLE_SHIFT,
					       (unsigned long *)sptep);
	if (was_writable && !spte_ad_enabled(*sptep))
		kvm_set_pfn_dirty(spte_to_pfn(*sptep));

	return was_writable;
}

/*
 * Gets the GFN ready for another round of dirty logging by clearing the
 *	- D bit on ad-enabled SPTEs, and
 *	- W bit on ad-disabled SPTEs.
 * Returns true iff any D or W bits were cleared.
 */
static bool __rmap_clear_dirty(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			       const struct kvm_memory_slot *slot)
{
	u64 *sptep;
	struct rmap_iterator iter;
	bool flush = false;

	for_each_rmap_spte(rmap_head, &iter, sptep)
		if (spte_ad_need_write_protect(*sptep))
			flush |= spte_wrprot_for_clear_dirty(sptep);
		else
			flush |= spte_clear_dirty(sptep);

	return flush;
}

/**
 * kvm_mmu_write_protect_pt_masked - write protect selected PT level pages
 * @kvm: kvm instance
 * @slot: slot to protect
 * @gfn_offset: start of the BITS_PER_LONG pages we care about
 * @mask: indicates which pages we should protect
 *
 * Used when we do not need to care about huge page mappings.
 */
static void kvm_mmu_write_protect_pt_masked(struct kvm *kvm,
				     struct kvm_memory_slot *slot,
				     gfn_t gfn_offset, unsigned long mask)
{
	struct kvm_rmap_head *rmap_head;

	if (tdp_mmu_enabled)
		kvm_tdp_mmu_clear_dirty_pt_masked(kvm, slot,
				slot->base_gfn + gfn_offset, mask, true);

	if (!kvm_memslots_have_rmaps(kvm))
		return;

	while (mask) {
		rmap_head = gfn_to_rmap(slot->base_gfn + gfn_offset + __ffs(mask),
					PG_LEVEL_4K, slot);
		rmap_write_protect(rmap_head, false);

		/* clear the first set bit */
		mask &= mask - 1;
	}
}

/**
 * kvm_mmu_clear_dirty_pt_masked - clear MMU D-bit for PT level pages, or write
 * protect the page if the D-bit isn't supported.
 * @kvm: kvm instance
 * @slot: slot to clear D-bit
 * @gfn_offset: start of the BITS_PER_LONG pages we care about
 * @mask: indicates which pages we should clear D-bit
 *
 * Used for PML to re-log the dirty GPAs after userspace querying dirty_bitmap.
 */
static void kvm_mmu_clear_dirty_pt_masked(struct kvm *kvm,
					 struct kvm_memory_slot *slot,
					 gfn_t gfn_offset, unsigned long mask)
{
	struct kvm_rmap_head *rmap_head;

	if (tdp_mmu_enabled)
		kvm_tdp_mmu_clear_dirty_pt_masked(kvm, slot,
				slot->base_gfn + gfn_offset, mask, false);

	if (!kvm_memslots_have_rmaps(kvm))
		return;

	while (mask) {
		rmap_head = gfn_to_rmap(slot->base_gfn + gfn_offset + __ffs(mask),
					PG_LEVEL_4K, slot);
		__rmap_clear_dirty(kvm, rmap_head, slot);

		/* clear the first set bit */
		mask &= mask - 1;
	}
}

/**
 * kvm_arch_mmu_enable_log_dirty_pt_masked - enable dirty logging for selected
 * PT level pages.
 *
 * It calls kvm_mmu_write_protect_pt_masked to write protect selected pages to
 * enable dirty logging for them.
 *
 * We need to care about huge page mappings: e.g. during dirty logging we may
 * have such mappings.
 */
void kvm_arch_mmu_enable_log_dirty_pt_masked(struct kvm *kvm,
				struct kvm_memory_slot *slot,
				gfn_t gfn_offset, unsigned long mask)
{
	/*
	 * Huge pages are NOT write protected when we start dirty logging in
	 * initially-all-set mode; must write protect them here so that they
	 * are split to 4K on the first write.
	 *
	 * The gfn_offset is guaranteed to be aligned to 64, but the base_gfn
	 * of memslot has no such restriction, so the range can cross two large
	 * pages.
	 */
	if (kvm_dirty_log_manual_protect_and_init_set(kvm)) {
		gfn_t start = slot->base_gfn + gfn_offset + __ffs(mask);
		gfn_t end = slot->base_gfn + gfn_offset + __fls(mask);

		if (READ_ONCE(eager_page_split))
			kvm_mmu_try_split_huge_pages(kvm, slot, start, end + 1, PG_LEVEL_4K);

		kvm_mmu_slot_gfn_write_protect(kvm, slot, start, PG_LEVEL_2M);

		/* Cross two large pages? */
		if (ALIGN(start << PAGE_SHIFT, PMD_SIZE) !=
		    ALIGN(end << PAGE_SHIFT, PMD_SIZE))
			kvm_mmu_slot_gfn_write_protect(kvm, slot, end,
						       PG_LEVEL_2M);
	}

	/* Now handle 4K PTEs.  */
	if (kvm_x86_ops.cpu_dirty_log_size)
		kvm_mmu_clear_dirty_pt_masked(kvm, slot, gfn_offset, mask);
	else
		kvm_mmu_write_protect_pt_masked(kvm, slot, gfn_offset, mask);
}

int kvm_cpu_dirty_log_size(void)
{
	return kvm_x86_ops.cpu_dirty_log_size;
}

bool kvm_mmu_slot_gfn_write_protect(struct kvm *kvm,
				    struct kvm_memory_slot *slot, u64 gfn,
				    int min_level)
{
	struct kvm_rmap_head *rmap_head;
	int i;
	bool write_protected = false;

	if (kvm_memslots_have_rmaps(kvm)) {
		for (i = min_level; i <= KVM_MAX_HUGEPAGE_LEVEL; ++i) {
			rmap_head = gfn_to_rmap(gfn, i, slot);
			write_protected |= rmap_write_protect(rmap_head, true);
		}
	}

	if (tdp_mmu_enabled)
		write_protected |=
			kvm_tdp_mmu_write_protect_gfn(kvm, slot, gfn, min_level);

	return write_protected;
}

static bool kvm_vcpu_write_protect_gfn(struct kvm_vcpu *vcpu, u64 gfn)
{
	struct kvm_memory_slot *slot;

	slot = kvm_vcpu_gfn_to_memslot(vcpu, gfn);
	return kvm_mmu_slot_gfn_write_protect(vcpu->kvm, slot, gfn, PG_LEVEL_4K);
}

static bool __kvm_zap_rmap(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			   const struct kvm_memory_slot *slot)
{
	return kvm_zap_all_rmap_sptes(kvm, rmap_head);
}

static bool kvm_zap_rmap(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			 struct kvm_memory_slot *slot, gfn_t gfn, int level,
			 pte_t unused)
{
	return __kvm_zap_rmap(kvm, rmap_head, slot);
}

static bool kvm_set_pte_rmap(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			     struct kvm_memory_slot *slot, gfn_t gfn, int level,
			     pte_t pte)
{
	u64 *sptep;
	struct rmap_iterator iter;
	bool need_flush = false;
	u64 new_spte;
	kvm_pfn_t new_pfn;

	WARN_ON_ONCE(pte_huge(pte));
	new_pfn = pte_pfn(pte);

restart:
	for_each_rmap_spte(rmap_head, &iter, sptep) {
		need_flush = true;

		if (pte_write(pte)) {
			kvm_zap_one_rmap_spte(kvm, rmap_head, sptep);
			goto restart;
		} else {
			new_spte = kvm_mmu_changed_pte_notifier_make_spte(
					*sptep, new_pfn);

			mmu_spte_clear_track_bits(kvm, sptep);
			mmu_spte_set(sptep, new_spte);
		}
	}

	if (need_flush && kvm_available_flush_remote_tlbs_range()) {
		kvm_flush_remote_tlbs_gfn(kvm, gfn, level);
		return false;
	}

	return need_flush;
}

struct slot_rmap_walk_iterator {
	/* input fields. */
	const struct kvm_memory_slot *slot;
	gfn_t start_gfn;
	gfn_t end_gfn;
	int start_level;
	int end_level;

	/* output fields. */
	gfn_t gfn;
	struct kvm_rmap_head *rmap;
	int level;

	/* private field. */
	struct kvm_rmap_head *end_rmap;
};

static void rmap_walk_init_level(struct slot_rmap_walk_iterator *iterator,
				 int level)
{
	iterator->level = level;
	iterator->gfn = iterator->start_gfn;
	iterator->rmap = gfn_to_rmap(iterator->gfn, level, iterator->slot);
	iterator->end_rmap = gfn_to_rmap(iterator->end_gfn, level, iterator->slot);
}

static void slot_rmap_walk_init(struct slot_rmap_walk_iterator *iterator,
				const struct kvm_memory_slot *slot,
				int start_level, int end_level,
				gfn_t start_gfn, gfn_t end_gfn)
{
	iterator->slot = slot;
	iterator->start_level = start_level;
	iterator->end_level = end_level;
	iterator->start_gfn = start_gfn;
	iterator->end_gfn = end_gfn;

	rmap_walk_init_level(iterator, iterator->start_level);
}

static bool slot_rmap_walk_okay(struct slot_rmap_walk_iterator *iterator)
{
	return !!iterator->rmap;
}

static void slot_rmap_walk_next(struct slot_rmap_walk_iterator *iterator)
{
	while (++iterator->rmap <= iterator->end_rmap) {
		iterator->gfn += (1UL << KVM_HPAGE_GFN_SHIFT(iterator->level));

		if (iterator->rmap->val)
			return;
	}

	if (++iterator->level > iterator->end_level) {
		iterator->rmap = NULL;
		return;
	}

	rmap_walk_init_level(iterator, iterator->level);
}

#define for_each_slot_rmap_range(_slot_, _start_level_, _end_level_,	\
	   _start_gfn, _end_gfn, _iter_)				\
	for (slot_rmap_walk_init(_iter_, _slot_, _start_level_,		\
				 _end_level_, _start_gfn, _end_gfn);	\
	     slot_rmap_walk_okay(_iter_);				\
	     slot_rmap_walk_next(_iter_))

typedef bool (*rmap_handler_t)(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			       struct kvm_memory_slot *slot, gfn_t gfn,
			       int level, pte_t pte);

static __always_inline bool kvm_handle_gfn_range(struct kvm *kvm,
						 struct kvm_gfn_range *range,
						 rmap_handler_t handler)
{
	struct slot_rmap_walk_iterator iterator;
	bool ret = false;

	for_each_slot_rmap_range(range->slot, PG_LEVEL_4K, KVM_MAX_HUGEPAGE_LEVEL,
				 range->start, range->end - 1, &iterator)
		ret |= handler(kvm, iterator.rmap, range->slot, iterator.gfn,
			       iterator.level, range->arg.pte);

	return ret;
}

bool kvm_unmap_gfn_range(struct kvm *kvm, struct kvm_gfn_range *range)
{
	bool flush = false;

	if (kvm_memslots_have_rmaps(kvm))
		flush = kvm_handle_gfn_range(kvm, range, kvm_zap_rmap);

	if (tdp_mmu_enabled)
		flush = kvm_tdp_mmu_unmap_gfn_range(kvm, range, flush);

	if (kvm_x86_ops.set_apic_access_page_addr &&
	    range->slot->id == APIC_ACCESS_PAGE_PRIVATE_MEMSLOT)
		kvm_make_all_cpus_request(kvm, KVM_REQ_APIC_PAGE_RELOAD);

	return flush;
}

bool kvm_set_spte_gfn(struct kvm *kvm, struct kvm_gfn_range *range)
{
	bool flush = false;

	if (kvm_memslots_have_rmaps(kvm))
		flush = kvm_handle_gfn_range(kvm, range, kvm_set_pte_rmap);

	if (tdp_mmu_enabled)
		flush |= kvm_tdp_mmu_set_spte_gfn(kvm, range);

	return flush;
}

static bool kvm_age_rmap(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			 struct kvm_memory_slot *slot, gfn_t gfn, int level,
			 pte_t unused)
{
	u64 *sptep;
	struct rmap_iterator iter;
	int young = 0;

	for_each_rmap_spte(rmap_head, &iter, sptep)
		young |= mmu_spte_age(sptep);

	return young;
}

static bool kvm_test_age_rmap(struct kvm *kvm, struct kvm_rmap_head *rmap_head,
			      struct kvm_memory_slot *slot, gfn_t gfn,
			      int level, pte_t unused)
{
	u64 *sptep;
	struct rmap_iterator iter;

	for_each_rmap_spte(rmap_head, &iter, sptep)
		if (is_accessed_spte(*sptep))
			return true;
	return false;
}

#define RMAP_RECYCLE_THRESHOLD 1000

static void __rmap_add(struct kvm *kvm,
		       struct kvm_mmu_memory_cache *cache,
		       const struct kvm_memory_slot *slot,
		       u64 *spte, gfn_t gfn, unsigned int access)
{
	struct kvm_mmu_page *sp;
	struct kvm_rmap_head *rmap_head;
	int rmap_count;

	sp = sptep_to_sp(spte);
	kvm_mmu_page_set_translation(sp, spte_index(spte), gfn, access);
	kvm_update_page_stats(kvm, sp->role.level, 1);

	rmap_head = gfn_to_rmap(gfn, sp->role.level, slot);
	rmap_count = pte_list_add(cache, spte, rmap_head);

	if (rmap_count > kvm->stat.max_mmu_rmap_size)
		kvm->stat.max_mmu_rmap_size = rmap_count;
	if (rmap_count > RMAP_RECYCLE_THRESHOLD) {
		kvm_zap_all_rmap_sptes(kvm, rmap_head);
		kvm_flush_remote_tlbs_gfn(kvm, gfn, sp->role.level);
	}
}

static void rmap_add(struct kvm_vcpu *vcpu, const struct kvm_memory_slot *slot,
		     u64 *spte, gfn_t gfn, unsigned int access)
{
	struct kvm_mmu_memory_cache *cache = &vcpu->arch.mmu_pte_list_desc_cache;

	__rmap_add(vcpu->kvm, cache, slot, spte, gfn, access);
}

bool kvm_age_gfn(struct kvm *kvm, struct kvm_gfn_range *range)
{
	bool young = false;

	if (kvm_memslots_have_rmaps(kvm))
		young = kvm_handle_gfn_range(kvm, range, kvm_age_rmap);

	if (tdp_mmu_enabled)
		young |= kvm_tdp_mmu_age_gfn_range(kvm, range);

	return young;
}

bool kvm_test_age_gfn(struct kvm *kvm, struct kvm_gfn_range *range)
{
	bool young = false;

	if (kvm_memslots_have_rmaps(kvm))
		young = kvm_handle_gfn_range(kvm, range, kvm_test_age_rmap);

	if (tdp_mmu_enabled)
		young |= kvm_tdp_mmu_test_age_gfn(kvm, range);

	return young;
}

static void kvm_mmu_check_sptes_at_free(struct kvm_mmu_page *sp)
{
#ifdef CONFIG_KVM_PROVE_MMU
	int i;

	for (i = 0; i < SPTE_ENT_PER_PAGE; i++) {
		if (KVM_MMU_WARN_ON(is_shadow_present_pte(sp->spt[i])))
			pr_err_ratelimited("SPTE %llx (@ %p) for gfn %llx shadow-present at free",
					   sp->spt[i], &sp->spt[i],
					   kvm_mmu_page_get_gfn(sp, i));
	}
#endif
}

/*
 * This value is the sum of all of the kvm instances's
 * kvm->arch.n_used_mmu_pages values.  We need a global,
 * aggregate version in order to make the slab shrinker
 * faster
 */
static inline void kvm_mod_used_mmu_pages(struct kvm *kvm, long nr)
{
	kvm->arch.n_used_mmu_pages += nr;
	percpu_counter_add(&kvm_total_used_mmu_pages, nr);
}

static void kvm_account_mmu_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	kvm_mod_used_mmu_pages(kvm, +1);
	kvm_account_pgtable_pages((void *)sp->spt, +1);
}

static void kvm_unaccount_mmu_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	kvm_mod_used_mmu_pages(kvm, -1);
	kvm_account_pgtable_pages((void *)sp->spt, -1);
}

static void kvm_mmu_free_shadow_page(struct kvm_mmu_page *sp)
{
	kvm_mmu_check_sptes_at_free(sp);

	hlist_del(&sp->hash_link);
	list_del(&sp->link);
	free_page((unsigned long)sp->spt);
	if (!sp->role.direct)
		free_page((unsigned long)sp->shadowed_translation);
	kmem_cache_free(mmu_page_header_cache, sp);
}

static unsigned kvm_page_table_hashfn(gfn_t gfn)
{
	return hash_64(gfn, KVM_MMU_HASH_SHIFT);
}

static void mmu_page_add_parent_pte(struct kvm_mmu_memory_cache *cache,
				    struct kvm_mmu_page *sp, u64 *parent_pte)
{
	if (!parent_pte)
		return;

	pte_list_add(cache, parent_pte, &sp->parent_ptes);
}

static void mmu_page_remove_parent_pte(struct kvm *kvm, struct kvm_mmu_page *sp,
				       u64 *parent_pte)
{
	pte_list_remove(kvm, parent_pte, &sp->parent_ptes);
}

static void drop_parent_pte(struct kvm *kvm, struct kvm_mmu_page *sp,
			    u64 *parent_pte)
{
	mmu_page_remove_parent_pte(kvm, sp, parent_pte);
	mmu_spte_clear_no_track(parent_pte);
}

static void mark_unsync(u64 *spte);
static void kvm_mmu_mark_parents_unsync(struct kvm_mmu_page *sp)
{
	u64 *sptep;
	struct rmap_iterator iter;

	for_each_rmap_spte(&sp->parent_ptes, &iter, sptep) {
		mark_unsync(sptep);
	}
}

static void mark_unsync(u64 *spte)
{
	struct kvm_mmu_page *sp;

	sp = sptep_to_sp(spte);
	if (__test_and_set_bit(spte_index(spte), sp->unsync_child_bitmap))
		return;
	if (sp->unsync_children++)
		return;
	kvm_mmu_mark_parents_unsync(sp);
}

#define KVM_PAGE_ARRAY_NR 16

struct kvm_mmu_pages {
	struct mmu_page_and_offset {
		struct kvm_mmu_page *sp;
		unsigned int idx;
	} page[KVM_PAGE_ARRAY_NR];
	unsigned int nr;
};

static int mmu_pages_add(struct kvm_mmu_pages *pvec, struct kvm_mmu_page *sp,
			 int idx)
{
	int i;

	if (sp->unsync)
		for (i=0; i < pvec->nr; i++)
			if (pvec->page[i].sp == sp)
				return 0;

	pvec->page[pvec->nr].sp = sp;
	pvec->page[pvec->nr].idx = idx;
	pvec->nr++;
	return (pvec->nr == KVM_PAGE_ARRAY_NR);
}

static inline void clear_unsync_child_bit(struct kvm_mmu_page *sp, int idx)
{
	--sp->unsync_children;
	WARN_ON_ONCE((int)sp->unsync_children < 0);
	__clear_bit(idx, sp->unsync_child_bitmap);
}

static int __mmu_unsync_walk(struct kvm_mmu_page *sp,
			   struct kvm_mmu_pages *pvec)
{
	int i, ret, nr_unsync_leaf = 0;

	for_each_set_bit(i, sp->unsync_child_bitmap, 512) {
		struct kvm_mmu_page *child;
		u64 ent = sp->spt[i];

		if (!is_shadow_present_pte(ent) || is_large_pte(ent)) {
			clear_unsync_child_bit(sp, i);
			continue;
		}

		child = spte_to_child_sp(ent);

		if (child->unsync_children) {
			if (mmu_pages_add(pvec, child, i))
				return -ENOSPC;

			ret = __mmu_unsync_walk(child, pvec);
			if (!ret) {
				clear_unsync_child_bit(sp, i);
				continue;
			} else if (ret > 0) {
				nr_unsync_leaf += ret;
			} else
				return ret;
		} else if (child->unsync) {
			nr_unsync_leaf++;
			if (mmu_pages_add(pvec, child, i))
				return -ENOSPC;
		} else
			clear_unsync_child_bit(sp, i);
	}

	return nr_unsync_leaf;
}

#define INVALID_INDEX (-1)

static int mmu_unsync_walk(struct kvm_mmu_page *sp,
			   struct kvm_mmu_pages *pvec)
{
	pvec->nr = 0;
	if (!sp->unsync_children)
		return 0;

	mmu_pages_add(pvec, sp, INVALID_INDEX);
	return __mmu_unsync_walk(sp, pvec);
}

static void kvm_unlink_unsync_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	WARN_ON_ONCE(!sp->unsync);
	trace_kvm_mmu_sync_page(sp);
	sp->unsync = 0;
	--kvm->stat.mmu_unsync;
}

static bool kvm_mmu_prepare_zap_page(struct kvm *kvm, struct kvm_mmu_page *sp,
				     struct list_head *invalid_list);
static void kvm_mmu_commit_zap_page(struct kvm *kvm,
				    struct list_head *invalid_list);

static bool sp_has_gptes(struct kvm_mmu_page *sp)
{
	if (sp->role.direct)
		return false;

	if (sp->role.passthrough)
		return false;

	return true;
}

#define for_each_valid_sp(_kvm, _sp, _list)				\
	hlist_for_each_entry(_sp, _list, hash_link)			\
		if (is_obsolete_sp((_kvm), (_sp))) {			\
		} else

#define for_each_gfn_valid_sp_with_gptes(_kvm, _sp, _gfn)		\
	for_each_valid_sp(_kvm, _sp,					\
	  &(_kvm)->arch.mmu_page_hash[kvm_page_table_hashfn(_gfn)])	\
		if ((_sp)->gfn != (_gfn) || !sp_has_gptes(_sp)) {} else

static bool kvm_sync_page_check(struct kvm_vcpu *vcpu, struct kvm_mmu_page *sp)
{
	union kvm_mmu_page_role root_role = vcpu->arch.mmu->root_role;

	/*
	 * Ignore various flags when verifying that it's safe to sync a shadow
	 * page using the current MMU context.
	 *
	 *  - level: not part of the overall MMU role and will never match as the MMU's
	 *           level tracks the root level
	 *  - access: updated based on the new guest PTE
	 *  - quadrant: not part of the overall MMU role (similar to level)
	 */
	const union kvm_mmu_page_role sync_role_ign = {
		.level = 0xf,
		.access = 0x7,
		.quadrant = 0x3,
		.passthrough = 0x1,
	};

	/*
	 * Direct pages can never be unsync, and KVM should never attempt to
	 * sync a shadow page for a different MMU context, e.g. if the role
	 * differs then the memslot lookup (SMM vs. non-SMM) will be bogus, the
	 * reserved bits checks will be wrong, etc...
	 */
	if (WARN_ON_ONCE(sp->role.direct || !vcpu->arch.mmu->sync_spte ||
			 (sp->role.word ^ root_role.word) & ~sync_role_ign.word))
		return false;

	return true;
}

static int kvm_sync_spte(struct kvm_vcpu *vcpu, struct kvm_mmu_page *sp, int i)
{
	if (!sp->spt[i])
		return 0;

	return vcpu->arch.mmu->sync_spte(vcpu, sp, i);
}

static int __kvm_sync_page(struct kvm_vcpu *vcpu, struct kvm_mmu_page *sp)
{
	int flush = 0;
	int i;

	if (!kvm_sync_page_check(vcpu, sp))
		return -1;

	for (i = 0; i < SPTE_ENT_PER_PAGE; i++) {
		int ret = kvm_sync_spte(vcpu, sp, i);

		if (ret < -1)
			return -1;
		flush |= ret;
	}

	/*
	 * Note, any flush is purely for KVM's correctness, e.g. when dropping
	 * an existing SPTE or clearing W/A/D bits to ensure an mmu_notifier
	 * unmap or dirty logging event doesn't fail to flush.  The guest is
	 * responsible for flushing the TLB to ensure any changes in protection
	 * bits are recognized, i.e. until the guest flushes or page faults on
	 * a relevant address, KVM is architecturally allowed to let vCPUs use
	 * cached translations with the old protection bits.
	 */
	return flush;
}

static int kvm_sync_page(struct kvm_vcpu *vcpu, struct kvm_mmu_page *sp,
			 struct list_head *invalid_list)
{
	int ret = __kvm_sync_page(vcpu, sp);

	if (ret < 0)
		kvm_mmu_prepare_zap_page(vcpu->kvm, sp, invalid_list);
	return ret;
}

static bool kvm_mmu_remote_flush_or_zap(struct kvm *kvm,
					struct list_head *invalid_list,
					bool remote_flush)
{
	if (!remote_flush && list_empty(invalid_list))
		return false;

	if (!list_empty(invalid_list))
		kvm_mmu_commit_zap_page(kvm, invalid_list);
	else
		kvm_flush_remote_tlbs(kvm);
	return true;
}

static bool is_obsolete_sp(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	if (sp->role.invalid)
		return true;

	/* TDP MMU pages do not use the MMU generation. */
	return !is_tdp_mmu_page(sp) &&
	       unlikely(sp->mmu_valid_gen != kvm->arch.mmu_valid_gen);
}

struct mmu_page_path {
	struct kvm_mmu_page *parent[PT64_ROOT_MAX_LEVEL];
	unsigned int idx[PT64_ROOT_MAX_LEVEL];
};

#define for_each_sp(pvec, sp, parents, i)			\
		for (i = mmu_pages_first(&pvec, &parents);	\
			i < pvec.nr && ({ sp = pvec.page[i].sp; 1;});	\
			i = mmu_pages_next(&pvec, &parents, i))

static int mmu_pages_next(struct kvm_mmu_pages *pvec,
			  struct mmu_page_path *parents,
			  int i)
{
	int n;

	for (n = i+1; n < pvec->nr; n++) {
		struct kvm_mmu_page *sp = pvec->page[n].sp;
		unsigned idx = pvec->page[n].idx;
		int level = sp->role.level;

		parents->idx[level-1] = idx;
		if (level == PG_LEVEL_4K)
			break;

		parents->parent[level-2] = sp;
	}

	return n;
}

static int mmu_pages_first(struct kvm_mmu_pages *pvec,
			   struct mmu_page_path *parents)
{
	struct kvm_mmu_page *sp;
	int level;

	if (pvec->nr == 0)
		return 0;

	WARN_ON_ONCE(pvec->page[0].idx != INVALID_INDEX);

	sp = pvec->page[0].sp;
	level = sp->role.level;
	WARN_ON_ONCE(level == PG_LEVEL_4K);

	parents->parent[level-2] = sp;

	/* Also set up a sentinel.  Further entries in pvec are all
	 * children of sp, so this element is never overwritten.
	 */
	parents->parent[level-1] = NULL;
	return mmu_pages_next(pvec, parents, 0);
}

static void mmu_pages_clear_parents(struct mmu_page_path *parents)
{
	struct kvm_mmu_page *sp;
	unsigned int level = 0;

	do {
		unsigned int idx = parents->idx[level];
		sp = parents->parent[level];
		if (!sp)
			return;

		WARN_ON_ONCE(idx == INVALID_INDEX);
		clear_unsync_child_bit(sp, idx);
		level++;
	} while (!sp->unsync_children);
}

static int mmu_sync_children(struct kvm_vcpu *vcpu,
			     struct kvm_mmu_page *parent, bool can_yield)
{
	int i;
	struct kvm_mmu_page *sp;
	struct mmu_page_path parents;
	struct kvm_mmu_pages pages;
	LIST_HEAD(invalid_list);
	bool flush = false;

	while (mmu_unsync_walk(parent, &pages)) {
		bool protected = false;

		for_each_sp(pages, sp, parents, i)
			protected |= kvm_vcpu_write_protect_gfn(vcpu, sp->gfn);

		if (protected) {
			kvm_mmu_remote_flush_or_zap(vcpu->kvm, &invalid_list, true);
			flush = false;
		}

		for_each_sp(pages, sp, parents, i) {
			kvm_unlink_unsync_page(vcpu->kvm, sp);
			flush |= kvm_sync_page(vcpu, sp, &invalid_list) > 0;
			mmu_pages_clear_parents(&parents);
		}
		if (need_resched() || rwlock_needbreak(&vcpu->kvm->mmu_lock)) {
			kvm_mmu_remote_flush_or_zap(vcpu->kvm, &invalid_list, flush);
			if (!can_yield) {
				kvm_make_request(KVM_REQ_MMU_SYNC, vcpu);
				return -EINTR;
			}

			cond_resched_rwlock_write(&vcpu->kvm->mmu_lock);
			flush = false;
		}
	}

	kvm_mmu_remote_flush_or_zap(vcpu->kvm, &invalid_list, flush);
	return 0;
}

static void __clear_sp_write_flooding_count(struct kvm_mmu_page *sp)
{
	atomic_set(&sp->write_flooding_count,  0);
}

static void clear_sp_write_flooding_count(u64 *spte)
{
	__clear_sp_write_flooding_count(sptep_to_sp(spte));
}

/*
 * The vCPU is required when finding indirect shadow pages; the shadow
 * page may already exist and syncing it needs the vCPU pointer in
 * order to read guest page tables.  Direct shadow pages are never
 * unsync, thus @vcpu can be NULL if @role.direct is true.
 */
static struct kvm_mmu_page *kvm_mmu_find_shadow_page(struct kvm *kvm,
						     struct kvm_vcpu *vcpu,
						     gfn_t gfn,
						     struct hlist_head *sp_list,
						     union kvm_mmu_page_role role)
{
	struct kvm_mmu_page *sp;
	int ret;
	int collisions = 0;
	LIST_HEAD(invalid_list);

	for_each_valid_sp(kvm, sp, sp_list) {
		if (sp->gfn != gfn) {
			collisions++;
			continue;
		}

		if (sp->role.word != role.word) {
			/*
			 * If the guest is creating an upper-level page, zap
			 * unsync pages for the same gfn.  While it's possible
			 * the guest is using recursive page tables, in all
			 * likelihood the guest has stopped using the unsync
			 * page and is installing a completely unrelated page.
			 * Unsync pages must not be left as is, because the new
			 * upper-level page will be write-protected.
			 */
			if (role.level > PG_LEVEL_4K && sp->unsync)
				kvm_mmu_prepare_zap_page(kvm, sp,
							 &invalid_list);
			continue;
		}

		/* unsync and write-flooding only apply to indirect SPs. */
		if (sp->role.direct)
			goto out;

		if (sp->unsync) {
			if (KVM_BUG_ON(!vcpu, kvm))
				break;

			/*
			 * The page is good, but is stale.  kvm_sync_page does
			 * get the latest guest state, but (unlike mmu_unsync_children)
			 * it doesn't write-protect the page or mark it synchronized!
			 * This way the validity of the mapping is ensured, but the
			 * overhead of write protection is not incurred until the
			 * guest invalidates the TLB mapping.  This allows multiple
			 * SPs for a single gfn to be unsync.
			 *
			 * If the sync fails, the page is zapped.  If so, break
			 * in order to rebuild it.
			 */
			ret = kvm_sync_page(vcpu, sp, &invalid_list);
			if (ret < 0)
				break;

			WARN_ON_ONCE(!list_empty(&invalid_list));
			if (ret > 0)
				kvm_flush_remote_tlbs(kvm);
		}

		__clear_sp_write_flooding_count(sp);

		goto out;
	}

	sp = NULL;
	++kvm->stat.mmu_cache_miss;

out:
	kvm_mmu_commit_zap_page(kvm, &invalid_list);

	if (collisions > kvm->stat.max_mmu_page_hash_collisions)
		kvm->stat.max_mmu_page_hash_collisions = collisions;
	return sp;
}

/* Caches used when allocating a new shadow page. */
struct shadow_page_caches {
	struct kvm_mmu_memory_cache *page_header_cache;
	struct kvm_mmu_memory_cache *shadow_page_cache;
	struct kvm_mmu_memory_cache *shadowed_info_cache;
};

static struct kvm_mmu_page *kvm_mmu_alloc_shadow_page(struct kvm *kvm,
						      struct shadow_page_caches *caches,
						      gfn_t gfn,
						      struct hlist_head *sp_list,
						      union kvm_mmu_page_role role)
{
	struct kvm_mmu_page *sp;

	sp = kvm_mmu_memory_cache_alloc(caches->page_header_cache);
	sp->spt = kvm_mmu_memory_cache_alloc(caches->shadow_page_cache);
	if (!role.direct)
		sp->shadowed_translation = kvm_mmu_memory_cache_alloc(caches->shadowed_info_cache);

	set_page_private(virt_to_page(sp->spt), (unsigned long)sp);

	INIT_LIST_HEAD(&sp->possible_nx_huge_page_link);

	/*
	 * active_mmu_pages must be a FIFO list, as kvm_zap_obsolete_pages()
	 * depends on valid pages being added to the head of the list.  See
	 * comments in kvm_zap_obsolete_pages().
	 */
	sp->mmu_valid_gen = kvm->arch.mmu_valid_gen;
	list_add(&sp->link, &kvm->arch.active_mmu_pages);
	kvm_account_mmu_page(kvm, sp);

	sp->gfn = gfn;
	sp->role = role;
	hlist_add_head(&sp->hash_link, sp_list);
	if (sp_has_gptes(sp))
		account_shadowed(kvm, sp);

	return sp;
}

/* Note, @vcpu may be NULL if @role.direct is true; see kvm_mmu_find_shadow_page. */
static struct kvm_mmu_page *__kvm_mmu_get_shadow_page(struct kvm *kvm,
						      struct kvm_vcpu *vcpu,
						      struct shadow_page_caches *caches,
						      gfn_t gfn,
						      union kvm_mmu_page_role role)
{
	struct hlist_head *sp_list;
	struct kvm_mmu_page *sp;
	bool created = false;

	sp_list = &kvm->arch.mmu_page_hash[kvm_page_table_hashfn(gfn)];

	sp = kvm_mmu_find_shadow_page(kvm, vcpu, gfn, sp_list, role);
	if (!sp) {
		created = true;
		sp = kvm_mmu_alloc_shadow_page(kvm, caches, gfn, sp_list, role);
	}

	trace_kvm_mmu_get_page(sp, created);
	return sp;
}

static struct kvm_mmu_page *kvm_mmu_get_shadow_page(struct kvm_vcpu *vcpu,
						    gfn_t gfn,
						    union kvm_mmu_page_role role)
{
	struct shadow_page_caches caches = {
		.page_header_cache = &vcpu->arch.mmu_page_header_cache,
		.shadow_page_cache = &vcpu->arch.mmu_shadow_page_cache,
		.shadowed_info_cache = &vcpu->arch.mmu_shadowed_info_cache,
	};

	return __kvm_mmu_get_shadow_page(vcpu->kvm, vcpu, &caches, gfn, role);
}

static union kvm_mmu_page_role kvm_mmu_child_role(u64 *sptep, bool direct,
						  unsigned int access)
{
	struct kvm_mmu_page *parent_sp = sptep_to_sp(sptep);
	union kvm_mmu_page_role role;

	role = parent_sp->role;
	role.level--;
	role.access = access;
	role.direct = direct;
	role.passthrough = 0;

	/*
	 * If the guest has 4-byte PTEs then that means it's using 32-bit,
	 * 2-level, non-PAE paging. KVM shadows such guests with PAE paging
	 * (i.e. 8-byte PTEs). The difference in PTE size means that KVM must
	 * shadow each guest page table with multiple shadow page tables, which
	 * requires extra bookkeeping in the role.
	 *
	 * Specifically, to shadow the guest's page directory (which covers a
	 * 4GiB address space), KVM uses 4 PAE page directories, each mapping
	 * 1GiB of the address space. @role.quadrant encodes which quarter of
	 * the address space each maps.
	 *
	 * To shadow the guest's page tables (which each map a 4MiB region), KVM
	 * uses 2 PAE page tables, each mapping a 2MiB region. For these,
	 * @role.quadrant encodes which half of the region they map.
	 *
	 * Concretely, a 4-byte PDE consumes bits 31:22, while an 8-byte PDE
	 * consumes bits 29:21.  To consume bits 31:30, KVM's uses 4 shadow
	 * PDPTEs; those 4 PAE page directories are pre-allocated and their
	 * quadrant is assigned in mmu_alloc_root().   A 4-byte PTE consumes
	 * bits 21:12, while an 8-byte PTE consumes bits 20:12.  To consume
	 * bit 21 in the PTE (the child here), KVM propagates that bit to the
	 * quadrant, i.e. sets quadrant to '0' or '1'.  The parent 8-byte PDE
	 * covers bit 21 (see above), thus the quadrant is calculated from the
	 * _least_ significant bit of the PDE index.
	 */
	if (role.has_4_byte_gpte) {
		WARN_ON_ONCE(role.level != PG_LEVEL_4K);
		role.quadrant = spte_index(sptep) & 1;
	}

	return role;
}

static struct kvm_mmu_page *kvm_mmu_get_child_sp(struct kvm_vcpu *vcpu,
						 u64 *sptep, gfn_t gfn,
						 bool direct, unsigned int access)
{
	union kvm_mmu_page_role role;

	if (is_shadow_present_pte(*sptep) && !is_large_pte(*sptep))
		return ERR_PTR(-EEXIST);

	role = kvm_mmu_child_role(sptep, direct, access);
	return kvm_mmu_get_shadow_page(vcpu, gfn, role);
}

static void shadow_walk_init_using_root(struct kvm_shadow_walk_iterator *iterator,
					struct kvm_vcpu *vcpu, hpa_t root,
					u64 addr)
{
	iterator->addr = addr;
	iterator->shadow_addr = root;
	iterator->level = vcpu->arch.mmu->root_role.level;

	if (iterator->level >= PT64_ROOT_4LEVEL &&
	    vcpu->arch.mmu->cpu_role.base.level < PT64_ROOT_4LEVEL &&
	    !vcpu->arch.mmu->root_role.direct)
		iterator->level = PT32E_ROOT_LEVEL;

	if (iterator->level == PT32E_ROOT_LEVEL) {
		/*
		 * prev_root is currently only used for 64-bit hosts. So only
		 * the active root_hpa is valid here.
		 */
		BUG_ON(root != vcpu->arch.mmu->root.hpa);

		iterator->shadow_addr
			= vcpu->arch.mmu->pae_root[(addr >> 30) & 3];
		iterator->shadow_addr &= SPTE_BASE_ADDR_MASK;
		--iterator->level;
		if (!iterator->shadow_addr)
			iterator->level = 0;
	}
}

static void shadow_walk_init(struct kvm_shadow_walk_iterator *iterator,
			     struct kvm_vcpu *vcpu, u64 addr)
{
	shadow_walk_init_using_root(iterator, vcpu, vcpu->arch.mmu->root.hpa,
				    addr);
}

static bool shadow_walk_okay(struct kvm_shadow_walk_iterator *iterator)
{
	if (iterator->level < PG_LEVEL_4K)
		return false;

	iterator->index = SPTE_INDEX(iterator->addr, iterator->level);
	iterator->sptep	= ((u64 *)__va(iterator->shadow_addr)) + iterator->index;
	return true;
}

static void __shadow_walk_next(struct kvm_shadow_walk_iterator *iterator,
			       u64 spte)
{
	if (!is_shadow_present_pte(spte) || is_last_spte(spte, iterator->level)) {
		iterator->level = 0;
		return;
	}

	iterator->shadow_addr = spte & SPTE_BASE_ADDR_MASK;
	--iterator->level;
}

static void shadow_walk_next(struct kvm_shadow_walk_iterator *iterator)
{
	__shadow_walk_next(iterator, *iterator->sptep);
}

static void __link_shadow_page(struct kvm *kvm,
			       struct kvm_mmu_memory_cache *cache, u64 *sptep,
			       struct kvm_mmu_page *sp, bool flush)
{
	u64 spte;

	BUILD_BUG_ON(VMX_EPT_WRITABLE_MASK != PT_WRITABLE_MASK);

	/*
	 * If an SPTE is present already, it must be a leaf and therefore
	 * a large one.  Drop it, and flush the TLB if needed, before
	 * installing sp.
	 */
	if (is_shadow_present_pte(*sptep))
		drop_large_spte(kvm, sptep, flush);

	spte = make_nonleaf_spte(sp->spt, sp_ad_disabled(sp));

	mmu_spte_set(sptep, spte);

	mmu_page_add_parent_pte(cache, sp, sptep);

	/*
	 * The non-direct sub-pagetable must be updated before linking.  For
	 * L1 sp, the pagetable is updated via kvm_sync_page() in
	 * kvm_mmu_find_shadow_page() without write-protecting the gfn,
	 * so sp->unsync can be true or false.  For higher level non-direct
	 * sp, the pagetable is updated/synced via mmu_sync_children() in
	 * FNAME(fetch)(), so sp->unsync_children can only be false.
	 * WARN_ON_ONCE() if anything happens unexpectedly.
	 */
	if (WARN_ON_ONCE(sp->unsync_children) || sp->unsync)
		mark_unsync(sptep);
}

static void link_shadow_page(struct kvm_vcpu *vcpu, u64 *sptep,
			     struct kvm_mmu_page *sp)
{
	__link_shadow_page(vcpu->kvm, &vcpu->arch.mmu_pte_list_desc_cache, sptep, sp, true);
}

static void validate_direct_spte(struct kvm_vcpu *vcpu, u64 *sptep,
				   unsigned direct_access)
{
	if (is_shadow_present_pte(*sptep) && !is_large_pte(*sptep)) {
		struct kvm_mmu_page *child;

		/*
		 * For the direct sp, if the guest pte's dirty bit
		 * changed form clean to dirty, it will corrupt the
		 * sp's access: allow writable in the read-only sp,
		 * so we should update the spte at this point to get
		 * a new sp with the correct access.
		 */
		child = spte_to_child_sp(*sptep);
		if (child->role.access == direct_access)
			return;

		drop_parent_pte(vcpu->kvm, child, sptep);
		kvm_flush_remote_tlbs_sptep(vcpu->kvm, sptep);
	}
}

/* Returns the number of zapped non-leaf child shadow pages. */
static int mmu_page_zap_pte(struct kvm *kvm, struct kvm_mmu_page *sp,
			    u64 *spte, struct list_head *invalid_list)
{
	u64 pte;
	struct kvm_mmu_page *child;

	pte = *spte;
	if (is_shadow_present_pte(pte)) {
		if (is_last_spte(pte, sp->role.level)) {
			drop_spte(kvm, spte);
		} else {
			child = spte_to_child_sp(pte);
			drop_parent_pte(kvm, child, spte);

			/*
			 * Recursively zap nested TDP SPs, parentless SPs are
			 * unlikely to be used again in the near future.  This
			 * avoids retaining a large number of stale nested SPs.
			 */
			if (tdp_enabled && invalid_list &&
			    child->role.guest_mode && !child->parent_ptes.val)
				return kvm_mmu_prepare_zap_page(kvm, child,
								invalid_list);
		}
	} else if (is_mmio_spte(pte)) {
		mmu_spte_clear_no_track(spte);
	}
	return 0;
}

static int kvm_mmu_page_unlink_children(struct kvm *kvm,
					struct kvm_mmu_page *sp,
					struct list_head *invalid_list)
{
	int zapped = 0;
	unsigned i;

	for (i = 0; i < SPTE_ENT_PER_PAGE; ++i)
		zapped += mmu_page_zap_pte(kvm, sp, sp->spt + i, invalid_list);

	return zapped;
}

static void kvm_mmu_unlink_parents(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	u64 *sptep;
	struct rmap_iterator iter;

	while ((sptep = rmap_get_first(&sp->parent_ptes, &iter)))
		drop_parent_pte(kvm, sp, sptep);
}

static int mmu_zap_unsync_children(struct kvm *kvm,
				   struct kvm_mmu_page *parent,
				   struct list_head *invalid_list)
{
	int i, zapped = 0;
	struct mmu_page_path parents;
	struct kvm_mmu_pages pages;

	if (parent->role.level == PG_LEVEL_4K)
		return 0;

	while (mmu_unsync_walk(parent, &pages)) {
		struct kvm_mmu_page *sp;

		for_each_sp(pages, sp, parents, i) {
			kvm_mmu_prepare_zap_page(kvm, sp, invalid_list);
			mmu_pages_clear_parents(&parents);
			zapped++;
		}
	}

	return zapped;
}

static bool __kvm_mmu_prepare_zap_page(struct kvm *kvm,
				       struct kvm_mmu_page *sp,
				       struct list_head *invalid_list,
				       int *nr_zapped)
{
	bool list_unstable, zapped_root = false;

	lockdep_assert_held_write(&kvm->mmu_lock);
	trace_kvm_mmu_prepare_zap_page(sp);
	++kvm->stat.mmu_shadow_zapped;
	*nr_zapped = mmu_zap_unsync_children(kvm, sp, invalid_list);
	*nr_zapped += kvm_mmu_page_unlink_children(kvm, sp, invalid_list);
	kvm_mmu_unlink_parents(kvm, sp);

	/* Zapping children means active_mmu_pages has become unstable. */
	list_unstable = *nr_zapped;

	if (!sp->role.invalid && sp_has_gptes(sp))
		unaccount_shadowed(kvm, sp);

	if (sp->unsync)
		kvm_unlink_unsync_page(kvm, sp);
	if (!sp->root_count) {
		/* Count self */
		(*nr_zapped)++;

		/*
		 * Already invalid pages (previously active roots) are not on
		 * the active page list.  See list_del() in the "else" case of
		 * !sp->root_count.
		 */
		if (sp->role.invalid)
			list_add(&sp->link, invalid_list);
		else
			list_move(&sp->link, invalid_list);
		kvm_unaccount_mmu_page(kvm, sp);
	} else {
		/*
		 * Remove the active root from the active page list, the root
		 * will be explicitly freed when the root_count hits zero.
		 */
		list_del(&sp->link);

		/*
		 * Obsolete pages cannot be used on any vCPUs, see the comment
		 * in kvm_mmu_zap_all_fast().  Note, is_obsolete_sp() also
		 * treats invalid shadow pages as being obsolete.
		 */
		zapped_root = !is_obsolete_sp(kvm, sp);
	}

	if (sp->nx_huge_page_disallowed)
		unaccount_nx_huge_page(kvm, sp);

	sp->role.invalid = 1;

	/*
	 * Make the request to free obsolete roots after marking the root
	 * invalid, otherwise other vCPUs may not see it as invalid.
	 */
	if (zapped_root)
		kvm_make_all_cpus_request(kvm, KVM_REQ_MMU_FREE_OBSOLETE_ROOTS);
	return list_unstable;
}

static bool kvm_mmu_prepare_zap_page(struct kvm *kvm, struct kvm_mmu_page *sp,
				     struct list_head *invalid_list)
{
	int nr_zapped;

	__kvm_mmu_prepare_zap_page(kvm, sp, invalid_list, &nr_zapped);
	return nr_zapped;
}

static void kvm_mmu_commit_zap_page(struct kvm *kvm,
				    struct list_head *invalid_list)
{
	struct kvm_mmu_page *sp, *nsp;

	if (list_empty(invalid_list))
		return;

	/*
	 * We need to make sure everyone sees our modifications to
	 * the page tables and see changes to vcpu->mode here. The barrier
	 * in the kvm_flush_remote_tlbs() achieves this. This pairs
	 * with vcpu_enter_guest and walk_shadow_page_lockless_begin/end.
	 *
	 * In addition, kvm_flush_remote_tlbs waits for all vcpus to exit
	 * guest mode and/or lockless shadow page table walks.
	 */
	kvm_flush_remote_tlbs(kvm);

	list_for_each_entry_safe(sp, nsp, invalid_list, link) {
		WARN_ON_ONCE(!sp->role.invalid || sp->root_count);
		kvm_mmu_free_shadow_page(sp);
	}
}

static unsigned long kvm_mmu_zap_oldest_mmu_pages(struct kvm *kvm,
						  unsigned long nr_to_zap)
{
	unsigned long total_zapped = 0;
	struct kvm_mmu_page *sp, *tmp;
	LIST_HEAD(invalid_list);
	bool unstable;
	int nr_zapped;

	if (list_empty(&kvm->arch.active_mmu_pages))
		return 0;

restart:
	list_for_each_entry_safe_reverse(sp, tmp, &kvm->arch.active_mmu_pages, link) {
		/*
		 * Don't zap active root pages, the page itself can't be freed
		 * and zapping it will just force vCPUs to realloc and reload.
		 */
		if (sp->root_count)
			continue;

		unstable = __kvm_mmu_prepare_zap_page(kvm, sp, &invalid_list,
						      &nr_zapped);
		total_zapped += nr_zapped;
		if (total_zapped >= nr_to_zap)
			break;

		if (unstable)
			goto restart;
	}

	kvm_mmu_commit_zap_page(kvm, &invalid_list);

	kvm->stat.mmu_recycled += total_zapped;
	return total_zapped;
}

static inline unsigned long kvm_mmu_available_pages(struct kvm *kvm)
{
	if (kvm->arch.n_max_mmu_pages > kvm->arch.n_used_mmu_pages)
		return kvm->arch.n_max_mmu_pages -
			kvm->arch.n_used_mmu_pages;

	return 0;
}

static int make_mmu_pages_available(struct kvm_vcpu *vcpu)
{
	unsigned long avail = kvm_mmu_available_pages(vcpu->kvm);

	if (likely(avail >= KVM_MIN_FREE_MMU_PAGES))
		return 0;

	kvm_mmu_zap_oldest_mmu_pages(vcpu->kvm, KVM_REFILL_PAGES - avail);

	/*
	 * Note, this check is intentionally soft, it only guarantees that one
	 * page is available, while the caller may end up allocating as many as
	 * four pages, e.g. for PAE roots or for 5-level paging.  Temporarily
	 * exceeding the (arbitrary by default) limit will not harm the host,
	 * being too aggressive may unnecessarily kill the guest, and getting an
	 * exact count is far more trouble than it's worth, especially in the
	 * page fault paths.
	 */
	if (!kvm_mmu_available_pages(vcpu->kvm))
		return -ENOSPC;
	return 0;
}

/*
 * Changing the number of mmu pages allocated to the vm
 * Note: if goal_nr_mmu_pages is too small, you will get dead lock
 */
void kvm_mmu_change_mmu_pages(struct kvm *kvm, unsigned long goal_nr_mmu_pages)
{
	write_lock(&kvm->mmu_lock);

	if (kvm->arch.n_used_mmu_pages > goal_nr_mmu_pages) {
		kvm_mmu_zap_oldest_mmu_pages(kvm, kvm->arch.n_used_mmu_pages -
						  goal_nr_mmu_pages);

		goal_nr_mmu_pages = kvm->arch.n_used_mmu_pages;
	}

	kvm->arch.n_max_mmu_pages = goal_nr_mmu_pages;

	write_unlock(&kvm->mmu_lock);
}

int kvm_mmu_unprotect_page(struct kvm *kvm, gfn_t gfn)
{
	struct kvm_mmu_page *sp;
	LIST_HEAD(invalid_list);
	int r;

	r = 0;
	write_lock(&kvm->mmu_lock);
	for_each_gfn_valid_sp_with_gptes(kvm, sp, gfn) {
		r = 1;
		kvm_mmu_prepare_zap_page(kvm, sp, &invalid_list);
	}
	kvm_mmu_commit_zap_page(kvm, &invalid_list);
	write_unlock(&kvm->mmu_lock);

	return r;
}

static int kvm_mmu_unprotect_page_virt(struct kvm_vcpu *vcpu, gva_t gva)
{
	gpa_t gpa;
	int r;

	if (vcpu->arch.mmu->root_role.direct)
		return 0;

	gpa = kvm_mmu_gva_to_gpa_read(vcpu, gva, NULL);

	r = kvm_mmu_unprotect_page(vcpu->kvm, gpa >> PAGE_SHIFT);

	return r;
}

static void kvm_unsync_page(struct kvm *kvm, struct kvm_mmu_page *sp)
{
	trace_kvm_mmu_unsync_page(sp);
	++kvm->stat.mmu_unsync;
	sp->unsync = 1;

	kvm_mmu_mark_parents_unsync(sp);
}

/*
 * Attempt to unsync any shadow pages that can be reached by the specified gfn,
 * KVM is creating a writable mapping for said gfn.  Returns 0 if all pages
 * were marked unsync (or if there is no shadow page), -EPERM if the SPTE must
 * be write-protected.
 */
int mmu_try_to_unsync_pages(struct kvm *kvm, const struct kvm_memory_slot *slot,
			    gfn_t gfn, bool can_unsync, bool prefetch)
{
	struct kvm_mmu_page *sp;
	bool locked = false;

	/*
	 * Force write-protection if the page is being tracked.  Note, the page
	 * track machinery is used to write-protect upper-level shadow pages,
	 * i.e. this guards the role.level == 4K assertion below!
	 */
	if (kvm_gfn_is_write_tracked(kvm, slot, gfn))
		return -EPERM;

	/*
	 * The page is not write-tracked, mark existing shadow pages unsync
	 * unless KVM is synchronizing an unsync SP (can_unsync = false).  In
	 * that case, KVM must complete emulation of the guest TLB flush before
	 * allowing shadow pages to become unsync (writable by the guest).
	 */
	for_each_gfn_valid_sp_with_gptes(kvm, sp, gfn) {
		if (!can_unsync)
			return -EPERM;

		if (sp->unsync)
			continue;

		if (prefetch)
			return -EEXIST;

		/*
		 * TDP MMU page faults require an additional spinlock as they
		 * run with mmu_lock held for read, not write, and the unsync
		 * logic is not thread safe.  Take the spinklock regardless of
		 * the MMU type to avoid extra conditionals/parameters, there's
		 * no meaningful penalty if mmu_lock is held for write.
		 */
		if (!locked) {
			locked = true;
			spin_lock(&kvm->arch.mmu_unsync_pages_lock);

			/*
			 * Recheck after taking the spinlock, a different vCPU
			 * may have since marked the page unsync.  A false
			 * negative on the unprotected check above is not
			 * possible as clearing sp->unsync _must_ hold mmu_lock
			 * for write, i.e. unsync cannot transition from 1->0
			 * while this CPU holds mmu_lock for read (or write).
			 */
			if (READ_ONCE(sp->unsync))
				continue;
		}

		WARN_ON_ONCE(sp->role.level != PG_LEVEL_4K);
		kvm_unsync_page(kvm, sp);
	}
	if (locked)
		spin_unlock(&kvm->arch.mmu_unsync_pages_lock);

	/*
	 * We need to ensure that the marking of unsync pages is visible
	 * before the SPTE is updated to allow writes because
	 * kvm_mmu_sync_roots() checks the unsync flags without holding
	 * the MMU lock and so can race with this. If the SPTE was updated
	 * before the page had been marked as unsync-ed, something like the
	 * following could happen:
	 *
	 * CPU 1                    CPU 2
	 * ---------------------------------------------------------------------
	 * 1.2 Host updates SPTE
	 *     to be writable
	 *                      2.1 Guest writes a GPTE for GVA X.
	 *                          (GPTE being in the guest page table shadowed
	 *                           by the SP from CPU 1.)
	 *                          This reads SPTE during the page table walk.
	 *                          Since SPTE.W is read as 1, there is no
	 *                          fault.
	 *
	 *                      2.2 Guest issues TLB flush.
	 *                          That causes a VM Exit.
	 *
	 *                      2.3 Walking of unsync pages sees sp->unsync is
	 *                          false and skips the page.
	 *
	 *                      2.4 Guest accesses GVA X.
	 *                          Since the mapping in the SP was not updated,
	 *                          so the old mapping for GVA X incorrectly
	 *                          gets used.
	 * 1.1 Host marks SP
	 *     as unsync
	 *     (sp->unsync = true)
	 *
	 * The write barrier below ensures that 1.1 happens before 1.2 and thus
	 * the situation in 2.4 does not arise.  It pairs with the read barrier
	 * in is_unsync_root(), placed between 2.1's load of SPTE.W and 2.3.
	 */
	smp_wmb();

	return 0;
}

static int mmu_set_spte(struct kvm_vcpu *vcpu, struct kvm_memory_slot *slot,
			u64 *sptep, unsigned int pte_access, gfn_t gfn,
			kvm_pfn_t pfn, struct kvm_page_fault *fault)
{
	struct kvm_mmu_page *sp = sptep_to_sp(sptep);
	int level = sp->role.level;
	int was_rmapped = 0;
	int ret = RET_PF_FIXED;
	bool flush = false;
	bool wrprot;
	u64 spte;

	/* Prefetching always gets a writable pfn.  */
	bool host_writable = !fault || fault->map_writable;
	bool prefetch = !fault || fault->prefetch;
	bool write_fault = fault && fault->write;

	if (unlikely(is_noslot_pfn(pfn))) {
		vcpu->stat.pf_mmio_spte_created++;
		mark_mmio_spte(vcpu, sptep, gfn, pte_access);
		return RET_PF_EMULATE;
	}

	if (is_shadow_present_pte(*sptep)) {
		/*
		 * If we overwrite a PTE page pointer with a 2MB PMD, unlink
		 * the parent of the now unreachable PTE.
		 */
		if (level > PG_LEVEL_4K && !is_large_pte(*sptep)) {
			struct kvm_mmu_page *child;
			u64 pte = *sptep;

			child = spte_to_child_sp(pte);
			drop_parent_pte(vcpu->kvm, child, sptep);
			flush = true;
		} else if (pfn != spte_to_pfn(*sptep)) {
			drop_spte(vcpu->kvm, sptep);
			flush = true;
		} else
			was_rmapped = 1;
	}

	wrprot = make_spte(vcpu, sp, slot, pte_access, gfn, pfn, *sptep, prefetch,
			   true, host_writable, &spte);

	if (*sptep == spte) {
		ret = RET_PF_SPURIOUS;
	} else {
		flush |= mmu_spte_update(sptep, spte);
		trace_kvm_mmu_set_spte(level, gfn, sptep);
	}

	if (wrprot) {
		if (write_fault)
			ret = RET_PF_EMULATE;
	}

	if (flush)
		kvm_flush_remote_tlbs_gfn(vcpu->kvm, gfn, level);

	if (!was_rmapped) {
		WARN_ON_ONCE(ret == RET_PF_SPURIOUS);
		rmap_add(vcpu, slot, sptep, gfn, pte_access);
	} else {
		/* Already rmapped but the pte_access bits may have changed. */
		kvm_mmu_page_set_access(sp, spte_index(sptep), pte_access);
	}

	return ret;
}

static int direct_pte_prefetch_many(struct kvm_vcpu *vcpu,
				    struct kvm_mmu_page *sp,
				    u64 *start, u64 *end)
{
	struct page *pages[PTE_PREFETCH_NUM];
	struct kvm_memory_slot *slot;
	unsigned int access = sp->role.access;
	int i, ret;
	gfn_t gfn;

	gfn = kvm_mmu_page_get_gfn(sp, spte_index(start));
	slot = gfn_to_memslot_dirty_bitmap(vcpu, gfn, access & ACC_WRITE_MASK);
	if (!slot)
		return -1;

	ret = gfn_to_page_many_atomic(slot, gfn, pages, end - start);
	if (ret <= 0)
		return -1;

	for (i = 0; i < ret; i++, gfn++, start++) {
		mmu_set_spte(vcpu, slot, start, access, gfn,
			     page_to_pfn(pages[i]), NULL);
		put_page(pages[i]);
	}

	return 0;
}

static void __direct_pte_prefetch(struct kvm_vcpu *vcpu,
				  struct kvm_mmu_page *sp, u64 *sptep)
{
	u64 *spte, *start = NULL;
	int i;

	WARN_ON_ONCE(!sp->role.direct);

	i = spte_index(sptep) & ~(PTE_PREFETCH_NUM - 1);
	spte = sp->spt + i;

	for (i = 0; i < PTE_PREFETCH_NUM; i++, spte++) {
		if (is_shadow_present_pte(*spte) || spte == sptep) {
			if (!start)
				continue;
			if (direct_pte_prefetch_many(vcpu, sp, start, spte) < 0)
				return;
			start = NULL;
		} else if (!start)
			start = spte;
	}
	if (start)
		direct_pte_prefetch_many(vcpu, sp, start, spte);
}

static void direct_pte_prefetch(struct kvm_vcpu *vcpu, u64 *sptep)
{
	struct kvm_mmu_page *sp;

	sp = sptep_to_sp(sptep);

	/*
	 * Without accessed bits, there's no way to distinguish between
	 * actually accessed translations and prefetched, so disable pte
	 * prefetch if accessed bits aren't available.
	 */
	if (sp_ad_disabled(sp))
		return;

	if (sp->role.level > PG_LEVEL_4K)
		return;

	/*
	 * If addresses are being invalidated, skip prefetching to avoid
	 * accidentally prefetching those addresses.
	 */
	if (unlikely(vcpu->kvm->mmu_invalidate_in_progress))
		return;

	__direct_pte_prefetch(vcpu, sp, sptep);
}

/*
 * Lookup the mapping level for @gfn in the current mm.
 *
 * WARNING!  Use of host_pfn_mapping_level() requires the caller and the end
 * consumer to be tied into KVM's handlers for MMU notifier events!
 *
 * There are several ways to safely use this helper:
 *
 * - Check mmu_invalidate_retry_gfn() after grabbing the mapping level, before
 *   consuming it.  In this case, mmu_lock doesn't need to be held during the
 *   lookup, but it does need to be held while checking the MMU notifier.
 *
 * - Hold mmu_lock AND ensure there is no in-progress MMU notifier invalidation
 *   event for the hva.  This can be done by explicit checking the MMU notifier
 *   or by ensuring that KVM already has a valid mapping that covers the hva.
 *
 * - Do not use the result to install new mappings, e.g. use the host mapping
 *   level only to decide whether or not to zap an entry.  In this case, it's
 *   not required to hold mmu_lock (though it's highly likely the caller will
 *   want to hold mmu_lock anyways, e.g. to modify SPTEs).
 *
 * Note!  The lookup can still race with modifications to host page tables, but
 * the above "rules" ensure KVM will not _consume_ the result of the walk if a
 * race with the primary MMU occurs.
 */
static int host_pfn_mapping_level(struct kvm *kvm, gfn_t gfn,
				  const struct kvm_memory_slot *slot)
{
	int level = PG_LEVEL_4K;
	unsigned long hva;
	unsigned long flags;
	pgd_t pgd;
	p4d_t p4d;
	pud_t pud;
	pmd_t pmd;

	/*
	 * Note, using the already-retrieved memslot and __gfn_to_hva_memslot()
	 * is not solely for performance, it's also necessary to avoid the
	 * "writable" check in __gfn_to_hva_many(), which will always fail on
	 * read-only memslots due to gfn_to_hva() assuming writes.  Earlier
	 * page fault steps have already verified the guest isn't writing a
	 * read-only memslot.
	 */
	hva = __gfn_to_hva_memslot(slot, gfn);

	/*
	 * Disable IRQs to prevent concurrent tear down of host page tables,
	 * e.g. if the primary MMU promotes a P*D to a huge page and then frees
	 * the original page table.
	 */
	local_irq_save(flags);

	/*
	 * Read each entry once.  As above, a non-leaf entry can be promoted to
	 * a huge page _during_ this walk.  Re-reading the entry could send the
	 * walk into the weeks, e.g. p*d_large() returns false (sees the old
	 * value) and then p*d_offset() walks into the target huge page instead
	 * of the old page table (sees the new value).
	 */
	pgd = READ_ONCE(*pgd_offset(kvm->mm, hva));
	if (pgd_none(pgd))
		goto out;

	p4d = READ_ONCE(*p4d_offset(&pgd, hva));
	if (p4d_none(p4d) || !p4d_present(p4d))
		goto out;

	pud = READ_ONCE(*pud_offset(&p4d, hva));
	if (pud_none(pud) || !pud_present(pud))
		goto out;

	if (pud_large(pud)) {
		level = PG_LEVEL_1G;
		goto out;
	}

	pmd = READ_ONCE(*pmd_offset(&pud, hva));
	if (pmd_none(pmd) || !pmd_present(pmd))
		goto out;

	if (pmd_large(pmd))
		level = PG_LEVEL_2M;

out:
	local_irq_restore(flags);
	return level;
}

static int __                          struct kvm *kvm,
			                                                       gfn_t gfn, int max_level, bool is_private)
{
	struct kvm_lpage_info *linfo;
	int host_level;

	max_level = min(max_level, max_huge_page_level);
	for ( ; max_level > PG_LEVEL_4K; max_level--) {
		linfo = lpage_info_slot(gfn, slot, max_level);
		if (!linfo->disallow_lpage)
			break;
	}

	if (is_private)
		return max_level;

	if (max_level == PG_LEVEL_4K)
		return PG_LEVEL_4K;

	host_level = host_pfn_mapping_level(kvm, gfn, slot);
	return min(host_level, max_level);
}

int kvm_mmu_max_mapping_level(struct kvm *kvm,
			      const struct kvm_memory_slot *slot, gfn_t gfn,
			      int max_level)
{
	bool is_private = kvm_slot_can_be_private(slot) &&
			  kvm_mem_is_private(kvm, gfn);

	return __                                     gfn, max_level, is_private);
}

void kvm_mmu_hugepage_adjust(struct kvm_vcpu *vcpu, struct kvm_page_fault *fault)
{
	struct kvm_memory_slot *slot = fault->slot;
	kvm_pfn_t mask;

	fault->huge_page_disallowed = fault->exec && fault->nx_huge_page_workaround_enabled;

	if (unlikely(fault->max_level == PG_LEVEL_4K))
		return;

	if (is_error_noslot_pfn(fault->pfn))
		return;

	if (kvm_slot_dirty_track_enabled(slot))
		return;

	/*
	 * Enforce the iTLB multihit workaround after capturing the requested
	 * level, which will be used to do precise, accurate accounting.
	 */
	fault->req_level = __                          vcpu->kvm, slot,
						       fault->gfn, fault->max_level,
						       fault->is_private);
	if (fault->req_level == PG_LEVEL_4K || fault->huge_page_disallowed)
		return;

	/*
	 * mmu_invalidate_retry() was successful and mmu_lock is held, so
	 * the pmd can't be split from under us.
	 */
	fault->goal_level = fault->req_level;
	mask = KVM_PAGES_PER_HPAGE(fault->goal_level) - 1;
	VM_BUG_ON((fault->gfn & mask) != (fault->pfn & mask));
	fault->pfn &= ~mask;
}

void disallowed_hugepage_adjust(struct kvm_page_fault *fault, u64 spte, int cur_level)
{
	if (cur_level > PG_LEVEL_4K &&
	    cur_level == fault->goal_level &&
	    is_shadow_present_pte(spte) &&
	    !is_large_pte(spte) &&
	    spte_to_child_sp(spte)->nx_huge_page_disallowed) {
		/*
		 * A small SPTE exists for this pfn, but FNAME(fetch),
		 * direct_map(), or kvm_tdp_mmu_map() would like to create a
		 * large PTE instead: just force them to go down another level,
		 * patching back for them into pfn the next 9 bits of the
		 * address.
		 */
		u64 page_mask = KVM_PAGES_PER_HPAGE(cur_level) -
				KVM_PAGES_PER_HPAGE(cur_level - 1);
		fault->pfn |= fault->gfn & page_mask;
		fault->goal_level--;
	}
}

static int direct_map(struct kvm_vcpu *vcpu, struct kvm_page_fault *fault)
{
	struct kvm_shadow_walk_iterator it;
	struct kvm_mmu_page *sp;
	int ret;
	gfn_t base_gfn = fault->gfn;

	kvm_mmu_hugepage_adjust(vcpu, fault);

	trace_kvm_mmu_spte_requested(fault);
	for_each_shadow_entry(vcpu, fault->addr, it) {
		/*
		 * We cannot overwrite existing page tables with an NX
		 * large page, as the leaf could be executable.
		 */
		if (fault->nx_huge_page_workaround_enabled)
			disallowed_hugepage_adjust(fault, *it.sptep, it.level);

		base_gfn = gfn_round_for_level(fault->gfn, it.level);
		if (it.level == fault->goal_level)
			break;

		sp = kvm_mmu_get_child_sp(vcpu, it.sptep, base_gfn, true, ACC_ALL);
		if (sp == ERR_PTR(-EEXIST))
			continue;

		link_shadow_page(vcpu, it.sptep, sp);
		if (fault->huge_page_disallowed)
			account_nx_huge_page(vcpu->kvm, sp,
					     fault->req_level >= it.level);
	}

	if (WARN_ON_ONCE(it.level != fault->goal_level))
		return -EFAULT;

	ret = mmu_set_spte(vcpu, fault->slot, it.sptep, ACC_ALL,
			   base_gfn, fault->pfn, fault);
	if (ret == RET_PF_SPURIOUS)
		return ret;

	direct_pte_prefetch(vcpu, it.sptep);
	return ret;
}

static void kvm_send_hwpoison_signal(struct kvm_memory_slot *slot, gfn_t gfn)
{
	unsigned long hva = gfn_to_hva_memslot(slot, gfn);

	send_sig_mceerr(BUS_MCEERR_AR, (void __user *)hva, PAGE_SHIFT, current);
}

static int kvm_handle_error_pfn(struct kvm_vcpu *vcpu, struct kvm_page_fault *fault)
{
	if (is_sigpending_pfn(fault->pfn)) {
		kvm_handle_signal_exit(vcpu);
		return -EINTR;
	}

	/*
	 * Do not cache the mmio info caused by writing the readonly gfn
	 * into the spte otherwise read access on readonly gfn also can
	 * caused mmio page fault and treat it as mmio access.
	 */
	if (fault->pfn == KVM_PFN_ERR_RO_FAULT)
		return RET_PF_EMULATE;

	if (fault->pfn == KVM_PFN_ERR_HWPOISON) {
		kvm_send_hwpoison_signal(fault->slot, fault->gfn);
		return RET_PF_RETRY;
	}

	return -EFAULT;
}

static int kvm_handle_noslot_fault(struct kvm_vcpu *vcpu,
				   struct kvm_page_fault *fault,
				   unsigned int access)
{
	gva_t gva = fault->is_tdp ? 0 : fault->addr;

	vcpu_cache_mmio_info(vcpu, gva, fault->gfn,
			     access & shadow_mmio_access_mask);

	/*
	 * If MMIO caching is disabled, emulate immediately without
	 * touching the shadow page tables as attempting to install an
	 * MMIO SPTE will just be an expensive nop.
	 */
	if (unlikely(!enable_mmio_caching))
		return RET_PF_EMULATE;

	/*
	 * Do not create an MMIO SPTE for a gfn greater than host.MAXPHYADDR,
	 * any guest that generates such gfns is running nested and is being
	 * tricked by L0 userspace (you can observe gfn > L1.MAXPHYADDR if and
	 * only if L1's MAXPHYADDR is inaccurate with respect to the
	 * hardware's).
	 */
	if (unlikely(fault->gfn > kvm_mmu_max_gfn()p, spte_index(start));
	slo                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              ;
	u64 *sptep                                                                                                                                                                                                                                                                                                                     /*
		 * It's entirely possible for the mapping to have been zapped
		 * by a different task, but the root page should always be
		 * available as the vCPU holds a reference to its root(s).
		                       !sptep))
			spte = REMOVED_SPTE;

		if (!is_shadow_present_pte(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   kvm_arch_nr_memslot_as_ids(kvm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     (root_pgd & __PT_BASE_ADDR_MASK)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       u8 kvm_max_level_for_order(int order)
{
	BUILD_BUG_ON(KVM_MAX_HUGEPAGE_LEVEL > PG_LEVEL_1G);

	KVM_MMU_WARN_ON(order != KVM_HPAGE_GFN_SHIFT(PG_LEVEL_1G) &&
			order != KVM_HPAGE_GFN_SHIFT(PG_LEVEL_2M) &&
			order != KVM_HPAGE_GFN_SHIFT(PG_LEVEL_4K));

	if (order >= KVM_HPAGE_GFN_SHIFT(PG_LEVEL_1G))
		return PG_LEVEL_1G;

	if (order >= KVM_HPAGE_GFN_SHIFT(PG_LEVEL_2M))
		return PG_LEVEL_2M;

	return PG_LEVEL_4K;
}

static void kvm_mmu_prepare_memory_fault_exi                           			      struct kvm_page_fault *fault)
{
	kvm_prepare_memory_fault_exit(vcpu, fault->gfn << PAGE_SHIFT,
				      PAGE_SIZE, fault->write, fault->exec,
				      fault->is_private);
}

static int kvm_faultin_pfn_priv                                  struct kvm_page_fault *fault     int max_order, r;

	if (!kvm_slot_can_be_private(fault->slot)) {
		kvm_mmu_prepare_memory_fault_exit(vcpu, fault);
		return -EFAULT;
	}

	r = kvm_gmem_get_pfn                                   , &fault->pfn,
			     &max_order);
	if (r) {
		kvm_mmu_prepare_memory_fault_exit(vcpu, fault);
		return r;
	}
                     min(kvm_max_level_for_order(max_order),
			       fault->max_level);
                       !(fault->slot->flags & KVM_MEM_READONLY);

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  if (fault->is_private != kvm_mem_is_private(vcpu->kvm, fault->gfn)) {
		kvm_mmu_prepare_memory_fault_exit(vcpu, fault);
		return -EFAULT;
	}

	if (fault->is_private)
		return kvm_faultin_pfn_private(vcpu, fault);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              gfn                                   gfn                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     bool __kvm_mmu_honors_guest_mtrrs(bool vm_has_noncoherent_dma)
{
	/*
	 * If host MTRRs are ignored                                           
	 * VM has non-coherent DMA                                             
	 *                                                                   
	 * to memory that is                                                                                                                                                                                     return vm_has_noncoherent_dma && shadow_memtype_mask;
}                                                                                                                                                                                                                                                             
	 */
	if (kvm_mmu_honors_guest_mtrrs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 kvm_arch_nr_memslot_as_ids(kvm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       range_add                                                                                                                                                                                                                                                                                           );

	write_unlock(&kvm->mmu_lock);
}

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   a                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         kvm_arch_nr_memslot_as_ids(kvm)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           *mmu_shrinker                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                mmu_shrinker = shrinker_alloc(0, "x86-mmu");
	if (!mmu_shrinker)
		goto out_shrinker;

	mmu_shrinker->count                      count;
	mmu_shrinker->                              ;
	mmu_shrinker->                          ;

	shrinker_register(mmu_shrinker)                                                                                                                                                                                                                                                                                                                                                                                                                                       shrinker_free(                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
#ifdef CONFIG_KVM_GENERIC_MEMORY_ATTRIBUTES
bool kvm_arch_pre_set_memory_attributes                       struct kvm_gfn_range *range)
{
	/*
	 * Zap SPTEs even if the slot can't be mapped PRIVATE.  KVM x86 only
	 * supports KVM_MEMORY_ATTRIBUTE_PRIVATE, and so it *seems* like KVM
	 * can simply ignore such slots.  But if userspace is making memory
	 * PRIVATE, then KVM must prevent the guest from accessing the memory
	 * as shared.  And if userspace is making memory SHARED and this point
	 * is reached, then at least one page within the range was previously
	 * PRIVATE, i.e. the slot's possible hugepage ranges are changing.
	 * Zapping SPTEs in this case ensures KVM will reassess whether or not
	 * a hugepage can be used for affected ranges                         !kvm_arch_has_private_mem(kvm)))
		return false;

	       kvm_unmap_gfn_range(kvm, range);
}

static bool hugepage_test_mixed(struct kvm_memory_slot *slot, gfn_t gfn,
				int level)
{
	return lpage_info_slot(gfn, slot, level)->disallow_lpage & KVM_LPAGE_MIXED_FLAG;
}

static void hugepage_clear_mixed(struct kvm_memory_slot *slot, gfn_t gfn,
				 int level)
{
	lpage_info_slot(gfn, slot, level)->disallow_lpage &= ~KVM_LPAGE_MIXED_FLAG;
}

static void hugepage_set_mixed(struct kvm_memory_slot *slot, gfn_t gfn,
			       int level)
{
	lpage_info_slot(gfn, slot, level)->disallow_lpage |= KVM_LPAGE_MIXED_FLAG;
}

static bool hugepage_has_attrs(struct kvm *kvm, struct kvm_memory_slot *slot,
			       gfn_t gfn, int level, unsigned long attrs)
{
	const unsigned long start = gfn;
	const unsigned long end = start + KVM_PAGES_PER_HPAGE(level);

	if (level == PG_LEVEL_2M)
		return kvm_range_has_memory_attributes(kvm, start, end, attrs);

	for (gfn = start; gfn < end; gfn += KVM_PAGES_PER_HPAGE(level - 1)) {
		if (hugepage_test_mixed(slot, gfn, level - 1) ||
		    attrs != kvm_get_memory_attributes(kvm, gfn))
			return false;
	}
	return true;
}

bool kvm_arch_post_set_memory_attributes                                   gfn_range *range)
{
	unsigned long attrs = range->arg.attribute                                 = range->slot;
	int level                                                                                         /*
	 * Calculate which ranges can be mapped with hugepages even if the slot
	 * can't map memory PRIVATE.  KVM mustn't create a SHARED hugepage over
	 * a range that has PRIVATE GFNs, and conversely converting a range to
	 * SHARED may now allow hugepages                         !kvm_arch_has_private_mem(kvm)))
		return false;

	       The sequence matters here: upper levels consume the result of lower
	 * level's scanning                                  level <= KVM_MAX_HUGEPAGE_LEVEL; level++) {
		gfn_t nr_pages = KVM_PAGES_PER_HPAGE(level);
		gfn_t gfn = gfn_round_for_level(range->start, level);

		/* Process the head page if it straddles the range. */
		if (gfn != range->start || gfn + nr_pages > range->end) {
			/*
			 * Skip mixed tracking if the aligned gfn isn't covered
			 * by the memslot, KVM can't use a hugepage due to the
			 * misaligned address regardless of memory attributes.
			 */
			if (gfn >= slot->base_gfn) {
				if (hugepage_has_attrs(kvm, slot, gfn, level, attrs))
					hugepage_clear_mixed(slot, gfn, level);
				else
					hugepage_set_mixed(slot, gfn, level);
			}
			gfn += nr_pages;
		}

		/*
		 * Pages entirely covered by the range are guaranteed to have
		 * only the attributes which were just set.
		 */
		for ( ; gfn + nr_pages <= range->end; gfn += nr_pages)
			hugepage_clear_mixed(slot, gfn, level);

		/*
		 * Process the last tail page if it straddles the range and is
		 * contained by the memslot.  Like the head page, KVM can't
		 * create a hugepage if the slot size is misaligned.
		 */
		if (gfn < range->end &&
		    (gfn + nr_pages) <= (slo                          )) {
			if (hugepage_has_attrs(kvm, slot, gfn, level, attrs))
				hugepage_clear_mixed(slot, gfn, level);
			else
				hugepage_set_mixed(slot, gfn, level);
		}
	}                                init_memslot_memory_attributes                                      memory_slot *slot     int level;

	if (!kvm_arch_has_private_mem(kvm))
		return;

	for (                     level <= KVM_MAX_HUGEPAGE_LEVEL; level++) {
		/*
		 * Don't bother tracking mixed attributes for pages that can't
		 * be huge due to alignment, i.e. process only pages that are
		 * entirely contained by the memslot.
		 */
		gfn_t end = gfn_round_for_level(                             , level);
		gfn_t start = gfn_round_for_level(slot->base_gfn, level);
		gfn_t nr_pages = KVM_PAGES_PER_HPAGE(level);
		gfn_t gfn;

		if (start < slot->base_gfn)
			start += nr_pages;

		/*
		 * Unlike setting attributes, every potential hugepage needs to
		 * be manually checked as the attributes may already be mixed.
		 */
		for (gfn = start; gfn < end; gfn += nr_pages) {
			unsigned long attrs = kvm_get_memory_attributes(kvm, gfn);

			if (hugepage_has_attrs(kvm, slot, gfn, level, attrs))
				hugepage_clear_mixed(slot, gfn, level);
			else
				hugepage_set_mixed(slot, gfn, level);
		}
	}
}
#endif
