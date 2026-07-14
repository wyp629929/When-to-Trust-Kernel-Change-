"""
Final figure generation with 3-seed error bars.
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
import os

OUTPUT = '/Users/wangyaoping/Desktop/ml_paper/figures'

with open(f'{OUTPUT}/seed_results.json') as f:
    D = json.load(f)

CMAP = {'poly': '#4C72B0', 'highfreq': '#DD8452', 'gmm': '#55A868'}
MARKER = {'poly': 'o', 'highfreq': 's', 'gmm': '^'}
WIDTHS = [32, 64, 128, 256, 512, 1024]
DATASETS = ['poly', 'highfreq', 'gmm']

# Build records
records = []
for ds in DATASETS:
    for w in WIDTHS:
        k = f'{ds}_w{w}'
        records.append({
            'ds': ds, 'w': w,
            'cka': D[k]['cka_mean'], 'cka_std': D[k]['cka_std'],
            'frob': D[k]['frob_mean'], 'frob_std': D[k]['frob_std'],
        })

# ============================================================
# Figure 1: Width → Kernel Stability
# ============================================================
def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for ds in DATASETS:
        pts = [r for r in records if r['ds'] == ds]
        x = [r['w'] for r in pts]
        ax1.errorbar(x, [r['cka'] for r in pts],
                     yerr=[r['cka_std'] for r in pts],
                     fmt=f'-{MARKER[ds]}', color=CMAP[ds], label=ds,
                     linewidth=2, markersize=7, capsize=3)
        ax2.errorbar(x, [r['frob'] for r in pts],
                     yerr=[r['frob_std'] for r in pts],
                     fmt=f'-{MARKER[ds]}', color=CMAP[ds], label=ds,
                     linewidth=2, markersize=7, capsize=3)

    ax1.set_xscale('log', base=2); ax1.set_xticks(WIDTHS)
    ax1.set_xticklabels([str(w) for w in WIDTHS])
    ax1.set_xlabel('Width'); ax1.set_ylabel('CKA(K₀, K_T)')
    ax1.set_title('(a) Kernel Similarity (CKA)'); ax1.set_ylim(0.82, 1.005)
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.set_xscale('log', base=2); ax2.set_xticks(WIDTHS)
    ax2.set_xticklabels([str(w) for w in WIDTHS])
    ax2.set_xlabel('Width'); ax2.set_ylabel('‖ΔK‖_F / ‖K₀‖_F (%)')
    ax2.set_title('(b) Frobenius Change'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig1_width_stability.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig1_width_stability.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")

# ============================================================
# Figure 2: ΔK vs ΔError (with err data from experiments)
# ============================================================
def fig2():
    # Use earlier seed_data for err approximation
    # We don't have per-seed err data, so we'll show CKA-Stability vs Frob
    # to demonstrate the dissociation
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: CKA vs Frobenius (all data with error bars)
    ax = axs[0]
    for ds in DATASETS:
        pts = [r for r in records if r['ds'] == ds]
        ax.errorbar([r['cka'] for r in pts], [r['frob'] for r in pts],
                     xerr=[r['cka_std'] for r in pts],
                     yerr=[r['frob_std'] for r in pts],
                     fmt=MARKER[ds], color=CMAP[ds], label=ds,
                     markersize=8, capsize=3, linewidth=0, elinewidth=1)
        for r in pts:
            ax.annotate(str(r['w']), (r['cka'], r['frob']),
                       fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('CKA(K₀, K_T)'); ax.set_ylabel('‖ΔK‖_F / ‖K₀‖_F (%)')
    ax.set_title('CKA vs Frobenius Change')
    ax.legend(); ax.grid(True, alpha=0.2)
    ax.axvspan(0.99, 1.01, ymin=0.05, ymax=0.6, alpha=0.08, color='red')
    ax.text(0.991, 55, 'Blind spot', fontsize=9, color='red', fontstyle='italic')

    # Panel B: CKA stability vs width (show dissociation with error bars)
    ax = axs[1]
    for ds in DATASETS:
        pts = [r for r in records if r['ds'] == ds]
        x = [r['w'] for r in pts]
        ax.errorbar(x, [r['cka'] for r in pts],
                    yerr=[r['cka_std'] for r in pts],
                    fmt=f'-{MARKER[ds]}', color=CMAP[ds], label=ds,
                    linewidth=2, markersize=7, capsize=3)

    ax.set_xscale('log', base=2); ax.set_xticks(WIDTHS)
    ax.set_xticklabels([str(w) for w in WIDTHS])
    ax.set_xlabel('Width')
    ax.set_ylabel('CKA(K₀, K_T)')
    ax.set_title('Kernel Stability Across Width (3 seeds)')
    ax.set_ylim(0.82, 1.005)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig2_dissociation.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig2_dissociation.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")

# ============================================================
# Figure 3: CKA blind spot
# ============================================================
def fig3():
    fig, ax = plt.subplots(figsize=(6, 5))
    for ds in DATASETS:
        pts = [r for r in records if r['ds'] == ds]
        ax.errorbar([r['cka'] for r in pts], [r['frob'] for r in pts],
                     xerr=[r['cka_std'] for r in pts],
                     yerr=[r['frob_std'] for r in pts],
                     fmt=MARKER[ds], color=CMAP[ds], label=ds,
                     markersize=9, capsize=4, linewidth=0, elinewidth=1.5)
        for r in pts:
            ax.annotate(str(r['w']), (r['cka'], r['frob']),
                       fontsize=8, xytext=(4, 4), textcoords='offset points')

    ax.set_xlabel('CKA(K₀, K_T)')
    ax.set_ylabel('‖K_T − K₀‖_F / ‖K₀‖_F (%)')
    ax.set_title('CKA vs Frobenius Change: The Blind Spot')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    ax.axhspan(20, 60, xmin=0.96, xmax=1.0, alpha=0.06, color='red')
    ax.axvspan(0.988, 1.002, ymin=0.18, ymax=0.6, alpha=0.06, color='red')
    ax.text(0.989, 50, 'CKA blind spot:\nCKA > 0.99,\nFrob > 20%',
            fontsize=9, color='red', fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig3_cka_blindspot.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig3_cka_blindspot.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")

# ============================================================
# Print summary table
# ============================================================
def print_table():
    print()
    print("="*95)
    print(f"{'Dataset':<12} {'Width':<8} {'CKA':<16} {'Frob%':<16} {'CKA_std':<10} {'Frob_std':<10}")
    print("="*95)
    for ds in DATASETS:
        for r in [rec for rec in records if rec['ds'] == ds]:
            print(f"{ds:<12} {r['w']:<8} {r['cka']:<16.4f} {r['frob']:<16.1f} {r['cka_std']:<10.4f} {r['frob_std']:<10.2f}")
        print("-"*95)
    print()

if __name__ == '__main__':
    print_table()
    fig1()
    fig2()
    fig3()
    print("\nAll figures saved.")
