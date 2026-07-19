"""
Fixed-width seed-wise analysis.
For each (dataset, width) combination, how much does Frobenius change
vary across seeds, and does it correlate with gain variation?

If width is a confound, then WITHIN each fixed width, the correlation
between ΔK and gain should disappear or be near-zero.
"""
import json, numpy as np
from scipy.stats import spearmanr

d = json.load(open('./figures/exp_scaleup_results.json'))
WIDTHS = [32, 64, 128, 256, 512, 1024]

print("=" * 90)
print(f"{'Dataset':<10} {'Width':<6} {'Seeds':<6} {'ρ(ΔK,Gain)':<14} {'p-value':<10} {'Mean ΔK':<10} {'Std ΔK':<8} {'Mean Gain':<10}")
print("=" * 90)

for ds in ['poly', 'highfreq', 'gmm']:
    for w in WIDTHS:
        pts = [r for r in d if r['ds'] == ds and r['width'] == w]
        frobs = [r['frob'] for r in pts]
        gains = [r['delta_err'] for r in pts]
        rho, p = spearmanr(frobs, gains)
        print(f"{ds:<10} {w:<6} {len(pts):<6} {rho:<+14.4f} {p:<10.4f} "
              f"{np.mean(frobs):<8.1f}±{np.std(frobs):<6.1f} {np.mean(gains):<+8.4f}")
    print("-" * 90)

# Pooled within-width analysis (remove width confound)
# Partial Spearman: ΔK vs Gain controlling for width
print("\n=== Partial Spearman (controlling for width) ===")
from scipy.stats import spearmanr
import pandas as pd

for ds in ['poly', 'highfreq', 'gmm']:
    pts = [r for r in d if r['ds'] == ds]
    frobs = np.array([r['frob'] for r in pts])
    gains = np.array([r['delta_err'] for r in pts])
    widths = np.array([r['width'] for r in pts])

    # Partial Spearman: residualize both ΔK and Gain by width
    def partial_spearman(x, y, z):
        # Spearman partial: rank, then residualize
        rx = np.argsort(np.argsort(x))  # rank transform
        ry = np.argsort(np.argsort(y))
        rz = np.argsort(np.argsort(z))
        from scipy import stats
        # Partial of x,y|z
        rho_xy, _ = spearmanr(x, y)
        rho_xz, _ = spearmanr(x, z)
        rho_yz, _ = spearmanr(y, z)
        r_partial = (rho_xy - rho_xz * rho_yz) / (np.sqrt((1 - rho_xz**2) * (1 - rho_yz**2)) + 1e-12)
        return r_partial

    r_raw, p_raw = spearmanr(frobs, gains)
    r_partial = partial_spearman(frobs, gains, widths)

    # Fixed-width pooled (all seeds within same width, then combine)
    within_rhos = []
    for w in WIDTHS:
        wp = [r for r in pts if r['width'] == w]
        fw = [r['frob'] for r in wp]
        gw = [r['delta_err'] for r in wp]
        if len(fw) >= 5:
            rw, _ = spearmanr(fw, gw)
            within_rhos.append(rw)

    print(f"{ds}: raw ρ={r_raw:.4f} (p={p_raw:.4e}), partial ρ|width={r_partial:.4f}, "
          f"mean within-width ρ={np.mean(within_rhos):.4f}")
