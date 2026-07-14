"""CNN eigenvalue spectrum evolution plot for CIFAR-10.
Shows top-20 eigenvalues before and after training at W=16 and W=128."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

d = json.load(open('/Users/wangyaoping/Desktop/ml_paper/figures/cnn_spectrum_results.json'))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

for ax, entry in zip(axes, d):
    w = entry['W']
    e0 = entry['evals0_init']
    eT = entry['evalsT_init']
    x = np.arange(1, len(e0) + 1)
    ax.plot(x, e0, 'o-', label='Initial', color='#4C72B0', markersize=4)
    ax.plot(x, eT, 's--', label='Trained', color='#DD8452', markersize=4)
    ax.set_xlabel('Eigenvalue index')
    ax.set_ylabel('Eigenvalue magnitude')
    ax.set_title(f'CNN W={w} on CIFAR-10')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/fig_cnn_spectrum.png', dpi=200, bbox_inches='tight')
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/fig_cnn_spectrum.pdf', bbox_inches='tight')
plt.close()
print("CNN spectrum figure saved.")
