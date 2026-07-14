"""Reliability Map summary figure for the diagnostic framework."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, 10.5)
ax.set_ylim(-0.4, 8)
ax.axis('off')

# Colors
C_BOX = '#E8F4F8'
C_EDGE = '#2C7FB8'
C_GOOD = '#D4F0D4'
C_GOOD_EDGE = '#2E8B2E'
C_WARN = '#FFF3CD'
C_WARN_EDGE = '#CC9900'
C_FAIL = '#F8D7DA'
C_FAIL_EDGE = '#B85450'

def box(ax, x, y, w, h, text, color=C_BOX, edge=C_EDGE, fs=11, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor=edge, linewidth=2, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', zorder=3)

def arrow(ax, x1, y1, x2, y2, label=''):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#555'), zorder=1)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + 0.15
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=8, color='#555',
                style='italic', zorder=1)

def arrow_split(ax, x1, y1, x2a, y2a, x2b, y2b, label_a='', label_b=''):
    """Arrow from (x1,y1) splitting to two targets."""
    mx = (x1 + x2a) / 2
    my = (y1 + y2a) / 2
    ax.annotate('', xy=(mx, my), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='#555'), zorder=1)
    ax.annotate('', xy=(x2a, y2a), xytext=(mx, my),
                arrowprops=dict(arrowstyle='->', lw=2, color='#555'), zorder=1)
    ax.annotate('', xy=(x2b, y2b), xytext=(mx, my),
                arrowprops=dict(arrowstyle='->', lw=2, color='#555'), zorder=1)
    if label_a:
        ax.text((mx + x2a)/2, (my + y2a)/2 + 0.15, label_a, ha='center', fontsize=8, color='#555', style='italic')
    if label_b:
        ax.text((mx + x2b)/2, (my + y2b)/2 + 0.15, label_b, ha='center', fontsize=8, color='#555', style='italic')

# ---- Title ----
ax.text(5, 7.6, 'Diagnostic Reliability Map', ha='center', va='center',
        fontsize=16, fontweight='bold')

# Row 1: Observable diagnostic
box(ax, 3, 6.3, 4, 0.7, 'Observable Kernel Diagnostic\n(CKA, Frobenius Change)',
    color=C_BOX, edge=C_EDGE, fs=11, bold=True)

# Row 2: Split into CKA high vs low
arrow_split(ax, 5, 6.3, 2, 5, 8, 5, label_a='CKA ≈ 1', label_b='CKA < 1, ΔK varies')

# Row 3 left: CKA high → Stability without learning
box(ax, 0.6, 4, 3.2, 0.8, 'Regime I: Stability\nwithout Learning',
    color=C_WARN, edge=C_WARN_EDGE, fs=10, bold=True)
ax.text(2.2, 3.9, 'High CKA does NOT imply\nsufficient representation',
        ha='center', va='top', fontsize=8.5, color='#666')

# Row 3 right: ΔK → Change without predictable learning
box(ax, 6.8, 4, 3.2, 0.8, 'Regime II: Change without\nPredictable Learning',
    color=C_WARN, edge=C_WARN_EDGE, fs=10, bold=True)
ax.text(8.4, 3.9, 'Large ΔK does NOT imply\nuseful task adaptation',
        ha='center', va='top', fontsize=8.5, color='#666')

# Arrow down from right
arrow(ax, 8, 4, 8, 2.8)

# Row 4 right: CKA blind spot
box(ax, 6.8, 2, 3.2, 0.8, 'Regime III: CKA Blind Spot',
    color=C_FAIL, edge=C_FAIL_EDGE, fs=10, bold=True)
ax.text(8.4, 1.9, 'CKA>0.99 but Frobenius>40%\nSpectral quantities evolve',
        ha='center', va='top', fontsize=8.5, color='#B85450')

# Arrow from left to center
arrow(ax, 2.2, 4, 5, 2.8)

# Row 5: Central insight
box(ax, 2.8, 1.4, 4.4, 1.1,
    'Diagnostic reliability depends on\ntask-aligned spectral evolution,\nnot global kernel magnitude',
    color=C_BOX, edge=C_EDGE, fs=10, bold=True)

# Arrow down
arrow(ax, 5, 1.4, 5, 0.6)

# Row 6: Validated diagnostic
box(ax, 3, 0.1, 4, 0.5, 'Validated Task-Aware Diagnostic',
    color=C_GOOD, edge=C_GOOD_EDGE, fs=11, bold=True)

# Down arrow to validated
arrow(ax, 5, 1.4, 5, 0.6)

# Right side: Path to spectral analysis
arrow(ax, 8.4, 2.8, 10.5, 1.5)
ax.text(10.5, 1.5, 'Need spectral\nanalysis\n(λ, vᵢ)', ha='center', va='center',
        fontsize=9, color='#555', bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/fig_reliability_map.png', dpi=200, bbox_inches='tight')
plt.savefig('/Users/wangyaoping/Desktop/ml_paper/figures/fig_reliability_map.pdf', bbox_inches='tight')
plt.close()
print("Reliability Map figure saved.")
