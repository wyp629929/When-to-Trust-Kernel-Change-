"""
Figure generation for "Kernel evolution does not explain finite-width feature learning"

Figure 1: Width → Kernel Stability (CKA, Frobenius)
Figure 2: ΔK vs ΔError scatter (Spearman ρ)
Figure 3: CKA vs Frobenius (the CKA blind spot)
Figure 4 (supp): Epoch dynamics for poly width=32
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
import warnings, os
warnings.filterwarnings('ignore')

# ============================================================
# Raw data from width_sweep.py
# ============================================================

DATA = [
    # (dataset, width, CKA_final, Frob_pct, err_init, err_final)
    ('poly',     32,   0.873639,  29.99,  0.2519,  0.0803),
    ('poly',     64,   0.965093,  14.88,  0.2146,  0.1129),
    ('poly',    128,   0.995709,   5.13,  0.2198,  0.2093),
    ('poly',    256,   0.999075,   3.97,  0.2777,  0.2778),
    ('poly',    512,   0.999905,   2.95,  0.3941,  0.3997),
    ('poly',   1024,   0.999991,   0.89,  0.5092,  0.5109),

    ('highfreq',  32,  0.850964, 100.64,  0.8923,  0.8844),
    ('highfreq',  64,  0.951538,  52.47,  0.8392,  0.8637),
    ('highfreq', 128,  0.980302,  21.42,  0.8312,  0.8361),
    ('highfreq', 256,  0.995467,   9.41,  0.8650,  0.8656),
    ('highfreq', 512,  0.999510,   1.80,  0.8837,  0.8835),
    ('highfreq',1024,  0.999948,   0.55,  0.8934,  0.8934),

    ('gmm',       32,  0.996009,  37.11,  0.0265,  0.0278),
    ('gmm',       64,  0.996642,  31.71,  0.0295,  0.0315),
    ('gmm',      128,  0.994041,  47.63,  0.0241,  0.0284),
    ('gmm',      256,  0.998109,  31.68,  0.0317,  0.0338),
    ('gmm',      512,  0.999621,  15.58,  0.0344,  0.0360),
    ('gmm',     1024,  0.999896,   8.02,  0.0368,  0.0374),
]

# epoch dynamics for poly width=32 (from measure_kernel_change.py)
POLY32_DYNAMICS = {
    'epoch':  [0, 10, 20, 50, 100, 200, 500],
    'test_err': [0.2519, 0.1524, 0.1198, 0.0976, 0.0901, 0.0843, 0.0803],
    'cka':  [1.0, 0.913, 0.895, 0.882, 0.877, 0.875, 0.874],
    'frob': [0.0, 15.2, 21.4, 26.8, 28.9, 29.7, 30.0],
}

# ============================================================
# Color scheme
# ============================================================

CMAP = {'poly': '#4C72B0', 'highfreq': '#DD8452', 'gmm': '#55A868'}
MARKER = {'poly': 'o', 'highfreq': 's', 'gmm': '^'}
WIDTHS = [32, 64, 128, 256, 512, 1024]

OUTPUT = '/Users/wangyaoping/Desktop/ml_paper/figures'

# ============================================================
# Figure 1: Width controls kernel stability
# ============================================================

def fig1():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [(d['width'], d['cka_final']) for d in RECORDS if d['dataset'] == ds]
        pts.sort()
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax1.plot(x, y, f'-{MARKER[ds]}', color=CMAP[ds], label=ds, linewidth=2, markersize=7)
        ax2.plot(x, [d['frob_pct'] for d in RECORDS if d['dataset'] == ds][::-1],
                 f'-{MARKER[ds]}', color=CMAP[ds], label=ds, linewidth=2, markersize=7)

    ax1.set_xscale('log', base=2)
    ax1.set_xticks(WIDTHS)
    ax1.set_xticklabels([str(w) for w in WIDTHS])
    ax1.set_xlabel('Width')
    ax1.set_ylabel('CKA(K₀, K_T)')
    ax1.set_title('(a) Kernel Similarity (CKA)')
    ax1.set_ylim(0.82, 1.005)
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
    plt.savefig(f'{OUTPUT}/fig1_width_stability.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig1_width_stability.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 1 saved.")


# ============================================================
# Figure 2: ΔK vs ΔError
# ============================================================

def fig2():
    fig, ax = plt.subplots(figsize=(6, 5))

    x_all, y_all = [], []
    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [(d['frob_pct'], -d['delta_err']) for d in RECORDS if d['dataset'] == ds]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        x_all.extend(x); y_all.extend(y)
        ax.scatter(x, y, c=CMAP[ds], marker=MARKER[ds], s=80,
                   label=ds, edgecolors='black', linewidths=0.5, zorder=3)

    # Spearman across all
    rho_all, p_all = spearmanr(x_all, y_all)
    # regression
    slope, intercept, r_val, p_val, _ = linregress(x_all, y_all)
    x_line = np.linspace(0, max(x_all) * 1.05, 100)
    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5,
            label=f'OLS (slope={slope:.3f})')

    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Kernel Change ‖ΔK‖_F / ‖K₀‖_F (%)')
    ax.set_ylabel('Generalization Improvement −ΔError')
    ax.set_title(f'Kernel Change vs Generalization\n'
                 f'All: ρ_s={rho_all:.3f} (p={p_all:.4f})', fontsize=11)

    # per-dataset Spearman
    legend_text = []
    for ds in ['poly', 'highfreq', 'gmm']:
        xd = [d['frob_pct'] for d in RECORDS if d['dataset'] == ds]
        yd = [-d['delta_err'] for d in RECORDS if d['dataset'] == ds]
        rho_d, p_d = spearmanr(xd, yd)
        legend_text.append(f'{ds}: ρ_s={rho_d:.3f} (p={p_d:.3f})')

    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.2)

    # text box with per-dataset stats
    textstr = '\n'.join(legend_text)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    print(f"  Spearman all data: ρ={rho_all:.4f}, p={p_all:.4f}")
    for ds in ['poly', 'highfreq', 'gmm']:
        xd = [d['frob_pct'] for d in RECORDS if d['dataset'] == ds]
        yd = [-d['delta_err'] for d in RECORDS if d['dataset'] == ds]
        rho_d, p_d = spearmanr(xd, yd)
        print(f"  Spearman {ds}: ρ={rho_d:.4f}, p={p_d:.4f}")

    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig2_delta_vs_gen.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig2_delta_vs_gen.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 2 saved.")


# ============================================================
# Figure 3: CKA vs Frobenius (the blind spot)
# ============================================================

def fig3():
    fig, ax = plt.subplots(figsize=(6, 5))

    for ds in ['poly', 'highfreq', 'gmm']:
        pts = [(d['cka_final'], d['frob_pct']) for d in RECORDS if d['dataset'] == ds]
        x = [p[0] for p in pts]
        y = [p[1] for p in pts]
        ax.scatter(x, y, c=CMAP[ds], marker=MARKER[ds], s=80,
                   label=ds, edgecolors='black', linewidths=0.5, zorder=3)
        # annotate width
        for d in RECORDS:
            if d['dataset'] == ds:
                ax.annotate(str(d['width']), (d['cka_final'], d['frob_pct']),
                           fontsize=7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('CKA(K₀, K_T)')
    ax.set_ylabel('‖ΔK‖_F / ‖K₀‖_F (%)')
    ax.set_title('CKA vs Frobenius Change: The Blind Spot')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # highlight the blind spot region
    ax.axvspan(0.99, 1.01, ymin=0.05, ymax=0.55, alpha=0.08, color='red')
    ax.text(0.991, 48, 'CKA blind spot:\nhigh similarity,\nsignificant\nspectral change',
            fontsize=8, color='red', fontstyle='italic')

    print("Figure 3 saved.")
    plt.tight_layout()
    plt.savefig(f'{OUTPUT}/fig3_cka_vs_frob.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig3_cka_vs_frob.pdf', bbox_inches='tight')
    plt.close()


# ============================================================
# Figure 4: Epoch dynamics (poly width=32)
# ============================================================

def fig4():
    fig, ax1 = plt.subplots(figsize=(6, 4))

    e = POLY32_DYNAMICS['epoch']
    color1 = '#4C72B0'
    color2 = '#DD8452'

    ax1.plot(e, POLY32_DYNAMICS['test_err'], 'o-', color=color1, linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Test Error', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(e, POLY32_DYNAMICS['cka'], 's--', color=color2, linewidth=2, markersize=6, label='CKA')
    ax2.set_ylabel('CKA(K₀, Kₜ)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.82, 1.02)

    # mark the divergence
    ax1.axvline(x=50, color='gray', linestyle=':', alpha=0.5)
    ax1.text(55, ax1.get_ylim()[1]*0.9, 'Error continues\ndropping, kernel\nstabilizes', fontsize=8,
             color='gray')

    ax1.set_title('Poly width=32: Error decreases while kernel stabilizes')
    fig.tight_layout()
    plt.savefig(f'{OUTPUT}/fig4_epoch_dynamics.png', dpi=200, bbox_inches='tight')
    plt.savefig(f'{OUTPUT}/fig4_epoch_dynamics.pdf', bbox_inches='tight')
    plt.close()
    print("Figure 4 saved.")


# ============================================================
# Supplementary table
# ============================================================

def print_table():
    print("\n" + "="*100)
    print(f"{'Dataset':<12} {'Width':<8} {'CKA':<10} {'Frob%':<10} {'Err0':<10} {'ErrT':<10} {'ΔErr':<10} {'KAE':<10}")
    print("="*100)
    for d in RECORDS:
        print(f"{d['dataset']:<12} {d['width']:<8} {d['cka_final']:<10.4f} {d['frob_pct']:<10.2f} "
              f"{d['err_init']:<10.4f} {d['err_final']:<10.4f} {d['delta_err']:<+10.4f} "
              f"{d['delta_err']/(d['frob_pct']+0.1):<10.4f}")
    print("="*100)


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    os.makedirs(OUTPUT, exist_ok=True)

    # build records
    global RECORDS
    RECORDS = []
    for ds, w, cka, frob, ei, ef in DATA:
        RECORDS.append({
            'dataset': ds, 'width': w, 'cka_final': cka,
            'frob_pct': frob, 'err_init': ei, 'err_final': ef,
            'delta_err': ei - ef,  # positive = improvement
        })

    print_table()
    fig1()
    fig2()
    fig3()
    fig4()
    print("\nAll figures generated.")
