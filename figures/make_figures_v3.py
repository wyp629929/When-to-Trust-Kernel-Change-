"""
Figure generation v3 — reads multi-seed JSON output.
Fig 1: Width → CKA & Frob (mean±std over 10 seeds)
Fig 2: All similarity metrics vs ΔError scatter
Fig 3: CKA blind spot with SVCCA overlay
Fig 4: Epoch dynamics (unchanged)
Fig S1: Frequency sweep
"""
import numpy as np, json, os, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
warnings.filterwarnings('ignore')

FIGDIR = '/Users/wangyaoping/Desktop/ml_paper/figures'
OUTDIR = '/Users/wangyaoping/Desktop/ml_paper/jmlr_paper/figures'
os.makedirs(FIGDIR, exist_ok=True)

CMAP = {'poly': '#4C72B0', 'highfreq': '#DD8452', 'gmm': '#55A868'}
MARKER = {'poly': 'o', 'highfreq': 's', 'gmm': '^'}
LABEL = {'poly': 'Poly', 'highfreq': 'High-Freq', 'gmm': 'GMM'}
WIDTHS = [32, 64, 128, 256, 512, 1024]

# ============================================================
# Load data
# ============================================================

def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    # Convert to list of dicts
    return raw  # list of dicts with keys: ds, width, seed, cka, svcca, pwcca, frob, delta_err

# ============================================================
# Figure 1: Width controls kernel stability (multi-seed)
# ============================================================

def fig1(records):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [r for r in records if r['ds'] == ds]
        x = WIDTHS
        cka_means = [np.mean([r['cka'] for r in pts if r['width'] == w]) for w in x]
        cka_stds = [np.std([r['cka'] for r in pts if r['width'] == w]) for w in x]
        frob_means = [np.mean([r['frob'] for r in pts if r['width'] == w]) for w in x]
        frob_stds = [np.std([r['frob'] for r in pts if r['width'] == w]) for w in x]

        ax1.errorbar(x, cka_means, yerr=cka_stds, fmt=f'-{MARKER[ds]}', color=CMAP[ds],
                     label=LABEL[ds], linewidth=2, markersize=7, capsize=3)
        ax2.errorbar(x, frob_means, yerr=frob_stds, fmt=f'-{MARKER[ds]}', color=CMAP[ds],
                     label=LABEL[ds], linewidth=2, markersize=7, capsize=3)

    ax1.set_xscale('log', base=2)
    ax1.set_xticks(WIDTHS)
    ax1.set_xticklabels([str(w) for w in WIDTHS])
    ax1.set_xlabel('Width')
    ax1.set_ylabel('CKA(K₀, K_T)')
    ax1.set_title('(a) Kernel Similarity (CKA)')
    ax1.set_ylim(0.80, 1.005)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale('log', base=2)
    ax2.set_xticks(WIDTHS)
    ax2.set_xticklabels([str(w) for w in WIDTHS])
    ax2.set_xlabel('Width')
    ax2.set_ylabel('‖K_T − K₀‖_F / ‖K₀‖_F (%)')
    ax2.set_title('(b) Frobenius Change')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig1_width_stability.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/fig1_width_stability.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")


# ============================================================
# Figure 2: All similarity metrics vs ΔError
# ============================================================

def fig2(records):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = [('cka', 'CKA(K₀, K_T)'),
               ('svcca', 'SVCCA(H₀, H₁)'),
               ('pwcca', 'PWCCA(H₀, H₁)')]
    widths = sorted(set(r['width'] for r in records))

    for idx, (ax, (key, xlabel)) in enumerate([(axes[0,0], metrics[0]), (axes[0,1], metrics[1]), (axes[1,0], metrics[2])]):
        for ds in ['poly', 'highfreq', 'gmm']:
            # Plot per-seed points for visual scatter
            pts = [r for r in records if r['ds'] == ds]
            xd = [r[key] for r in pts]
            yd = [-r['delta_err'] for r in pts]
            ax.scatter(xd, yd, c=CMAP[ds], marker=MARKER[ds], s=40, alpha=0.7,
                       edgecolors='k', linewidths=0.3, label=LABEL[ds], zorder=3)

        # Annotate with config-mean Spearman (matching Section 3.5 methodology)
        stats_lines = []
        for ds in ['poly', 'highfreq', 'gmm']:
            pts = [r for r in records if r['ds'] == ds]
            x_means = [np.mean([r[key] for r in pts if r['width']==w]) for w in widths]
            de_means = [np.mean([r['delta_err'] for r in pts if r['width']==w]) for w in widths]
            if len(set(x_means)) > 1 and len(set(de_means)) > 1:
                rho, p = spearmanr(x_means, de_means)
                stats_lines.append(f'{LABEL[ds]}: ρ_s={rho:.2f} (p={p:.3f})')
            else:
                stats_lines.append(f'{LABEL[ds]}: constant')

        ax.axhline(0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel(xlabel, fontsize=11)
        if idx == 0:
            ax.set_ylabel('Gen. Improvement −ΔError', fontsize=11)
        ax.legend(fontsize=7, loc='lower right')
        textstr = '\n'.join(stats_lines)
        props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none')
        ax.text(0.05, 0.98, textstr, transform=ax.transAxes, fontsize=7,
                verticalalignment='top', bbox=props)

    # Bottom-right: CKA vs Frobenius (blind spot)
    ax4 = axes[1, 1]
    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [r for r in records if r['ds'] == ds]
        xd = [r['cka'] for r in pts]
        yd = [r['frob'] for r in pts]
        ax4.scatter(xd, yd, c=CMAP[ds], marker=MARKER[ds], s=40, alpha=0.7,
                    edgecolors='k', linewidths=0.3, label=LABEL[ds], zorder=3)

    ax4.axhline(10, color='gray', ls=':', alpha=0.3)
    ax4.axvline(0.99, color='red', ls='--', alpha=0.3)
    ax4.set_xlabel('CKA(K₀, K_T)', fontsize=11)
    ax4.set_ylabel('‖ΔK‖_F / ‖K₀‖_F (%)', fontsize=11)
    ax4.legend(fontsize=7, loc='upper left')
    ax4.text(0.992, 45, 'CKA blind spot:\n>0.99 with\n>40% Frob', fontsize=8, color='red', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTDIR}/fig2.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTDIR}/fig2.pdf', dpi=200, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")


# ============================================================
# Figure 3: CKA vs SVCCA — do they agree?
# ============================================================

def fig3(records):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for ds in ['poly', 'highfreq', 'gmm']:
        cka_vals = [r['cka'] for r in records if r['ds'] == ds]
        svcca_vals = [r['svcca'] for r in records if r['ds'] == ds]
        ax.scatter(cka_vals, svcca_vals, c=CMAP[ds], marker=MARKER[ds], s=50, alpha=0.7,
                   edgecolors='k', linewidths=0.3, label=LABEL[ds], zorder=3)

    # Identity line
    ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', alpha=0.3, label='y=x')

    ax.set_xlabel('CKA(K₀, K_T)', fontsize=12)
    ax.set_ylabel('SVCCA(H₀, H₁)', fontsize=12)
    ax.set_title('CKA vs SVCCA: Representation Similarity Metrics', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Pooled Spearman
    all_cka = [r['cka'] for r in records]
    all_svcca = [r['svcca'] for r in records]
    rho, p = spearmanr(all_cka, all_svcca)
    ax.text(0.05, 0.05, f'Pooled ρ_s={rho:.3f} (p={p:.2e})',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/fig3_cka_vs_svcca.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/fig3_cka_vs_svcca.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 3 saved.")


# ============================================================
# Supplementary: Frequency sweep
# ============================================================

def fig_s1_freq(freq_path):
    with open(freq_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    FREQ_LABEL = {1: 'k=1 (low)', 2: 'k=2', 3: 'k=3', 5: 'k=5 (high)'}
    FREQ_COLOR = {1: '#2E86AB', 2: '#A23B72', 3: '#F18F01', 5: '#C73E1D'}

    # Panel A: CKA vs k
    ax = axes[0]
    for k in sorted(set(r['k'] for r in data)):
        pts = [r for r in data if r['k'] == k]
        ckas = [r['cka'] for r in pts]
        ax.scatter([k]*len(ckas), ckas, s=30, alpha=0.6, color=FREQ_COLOR[k], zorder=3)
        ax.plot(k, np.mean(ckas), 'o', color=FREQ_COLOR[k], markersize=10, markeredgecolor='k')
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('CKA(K₀, K_T)')
    ax.set_title('(a) Kernel Stability vs Frequency')
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xticklabels(['k=1', 'k=2', 'k=3', 'k=5'])

    # Panel B: Gain vs k
    ax = axes[1]
    for k in sorted(set(r['k'] for r in data)):
        pts = [r for r in data if r['k'] == k]
        gains = [-r['delta_err'] for r in pts]
        ax.scatter([k]*len(gains), gains, s=30, alpha=0.6, color=FREQ_COLOR[k], zorder=3)
        ax.plot(k, np.mean(gains), 'o', color=FREQ_COLOR[k], markersize=10, markeredgecolor='k')
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Frequency k')
    ax.set_ylabel('Generalization Improvement −ΔError')
    ax.set_title('(b) Generalization vs Frequency')
    ax.set_xticks([1, 2, 3, 5])
    ax.set_xticklabels(['k=1', 'k=2', 'k=3', 'k=5'])

    # Panel C: CKA vs Gain, colored by k
    ax = axes[2]
    for k in sorted(set(r['k'] for r in data)):
        pts = [r for r in data if r['k'] == k]
        ckas = [r['cka'] for r in pts]
        gains = [-r['delta_err'] for r in pts]
        ax.scatter(ckas, gains, s=40, alpha=0.7, color=FREQ_COLOR[k],
                   edgecolors='k', linewidths=0.3, label=FREQ_LABEL[k], zorder=3)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('CKA(K₀, K_T)')
    ax.set_ylabel('Generalization Improvement')
    ax.set_title('(c) Dissociation persists across frequencies')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{FIGDIR}/figS1_freq_sweep.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{FIGDIR}/figS1_freq_sweep.pdf', bbox_inches='tight')
    plt.close()
    print("Figure S1 (Frequency sweep) saved.")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    # Load expanded experiment data
    scale_path = f'{FIGDIR}/exp_scaleup_results.json'
    freq_path = f'{FIGDIR}/freq_sweep_results.json'

    records = load_data(scale_path)
    print(f"Loaded {len(records)} records from {scale_path}")

    fig1(records)
    fig2(records)
    fig3(records)

    if os.path.exists(freq_path):
        fig_s1_freq(freq_path)
    else:
        print(f"Skipping Fig S1: {freq_path} not found")

    print("\nAll figures generated.")
