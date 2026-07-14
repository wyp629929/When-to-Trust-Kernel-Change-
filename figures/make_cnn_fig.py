"""Generate CNN CIFAR scatter figure (Supplementary Figure S6)."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

d = json.load(open('/Users/wangyaoping/Desktop/ml_paper/figures/cnn_cifar_results.json'))
CMAP = {16: '#4C72B0', 32: '#DD8452', 64: '#55A868', 128: '#C44E52', 256: '#937860'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, xkey, xlabel, title in [
    (ax1, 'frob', 'Frobenius Change ‖ΔK‖/‖K‖ (%)', 'CNN Kernel Change vs Representation Improvement'),
    (ax2, 'cka', 'CKA(K₀, K_T)', 'CKA vs Representation Improvement')]:
    for w in sorted(set(r['width'] for r in d)):
        pts = [r for r in d if r['width'] == w]
        xs = [r[xkey] for r in pts]
        ys = [r['delta_r2'] for r in pts]
        ax.scatter(xs, ys, c=CMAP[w], s=60, label=f'W={w}', edgecolors='k', linewidths=0.3, zorder=3)

    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Linear Probe ΔR²', fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc='best')

    xs_all = [r[xkey] for r in d]
    ys_all = [r['delta_r2'] for r in d]
    rho, p = spearmanr(xs_all, ys_all)
    ax.text(0.05, 0.95, f'ρ={rho:.2f} (p={p:.2e})', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/figS6_cnn_cifar.png', dpi=200, bbox_inches='tight')
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/figS6_cnn_cifar.pdf', bbox_inches='tight')
plt.close()
print("Figure S6 saved.")
