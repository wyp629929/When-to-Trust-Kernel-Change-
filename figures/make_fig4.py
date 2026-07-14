"""Combined Figure 4: Real-data validation.
Panel (a): CNN CIFAR-10 — Frobenius vs ΔR² (linear probe improvement)
Panel (b): FashionMNIST — Frobenius vs ΔError (code: err0-err1, positive = improvement)
"""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

cnn = json.load(open('/Users/wangyaoping/Desktop/ml_paper/figures/cnn_cifar_results.json'))
fashion = json.load(open('/Users/wangyaoping/Desktop/ml_paper/figures/fashion_results.json'))

CMAP_W = {16: '#4C72B0', 32: '#DD8452', 64: '#55A868', 128: '#C44E52', 256: '#937860'}
W_LABEL = {16: 'W=16', 32: 'W=32', 64: 'W=64', 128: 'W=128', 256: 'W=256'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

# === Panel (a): CNN CIFAR-10: Frobenius vs ΔR² ===
for w in sorted(set(r['width'] for r in cnn)):
    pts = [r for r in cnn if r['width'] == w]
    ax1.scatter([r['frob'] for r in pts], [r['delta_r2'] for r in pts],
               c=CMAP_W[w], s=55, label=W_LABEL[w], edgecolors='k', linewidths=0.3, zorder=3)

ax1.axhline(0, color='gray', ls=':', alpha=0.5)
ax1.set_xlabel('Frobenius Change ‖ΔK‖/‖K‖ (%)', fontsize=11)
ax1.set_ylabel('Linear Probe ΔR²', fontsize=11)
ax1.set_title('(a) CNN on CIFAR-10', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, loc='lower right')

xs = [r['frob'] for r in cnn]; ys = [r['delta_r2'] for r in cnn]
rho, p = spearmanr(xs, ys)
ax1.text(0.05, 0.95, f'ρ={rho:.2f} (p={p:.2e})', transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# === Panel (b): FashionMNIST: Frobenius vs ΔError (code delta_err) ===
fw_map = {32: '#4C72B0', 64: '#DD8452', 128: '#55A868', 256: '#C44E52', 512: '#937860', 1024: '#333333'}
for w in sorted(set(r['w'] for r in fashion)):
    pts = [r for r in fashion if r['w'] == w]
    ax2.scatter([r['frob'] for r in pts], [r['delta_err'] for r in pts],
               c=fw_map[w], s=55, label=f'w={w}', edgecolors='k', linewidths=0.3, zorder=3)

ax2.axhline(0, color='gray', ls=':', alpha=0.5)
ax2.set_xlabel('Frobenius Change ‖ΔK‖/‖K‖ (%)', fontsize=11)
ax2.set_ylabel('ΔError (err₀−err₁)', fontsize=11)
ax2.set_title('(b) FashionMNIST Binary', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8, loc='upper left')

xs2 = [r['frob'] for r in fashion]; ys2 = [r['delta_err'] for r in fashion]
rho2, p2 = spearmanr(xs2, ys2)
ax2.text(0.05, 0.95, f'ρ={rho2:.2f} (p={p2:.2e})', transform=ax2.transAxes, fontsize=9,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax2.text(0.95, 0.55, 'Larger ΔK → LESS\nimprovement', transform=ax2.transAxes,
         fontsize=8, color='#B85450', fontweight='bold', ha='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, ec='#B85450'))

plt.tight_layout()
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/fig4_real_data.png', dpi=200, bbox_inches='tight')
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/fig4_real_data.pdf', bbox_inches='tight')
plt.close()
print("Figure 4 saved.")
