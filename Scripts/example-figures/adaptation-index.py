"""
Adaptation index pipeline schematic.

Shows the construction of a composite adaptation index
from longitudinal head-fixed stress acclimation measures.

Pipeline:
  measure trajectories → per-animal slopes → sign-align → z-score → mean = AI
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── style ────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.linewidth': 0,
})

fig, ax = plt.subplots(figsize=(7.5, 4.0), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.5)
ax.set_aspect('equal')
ax.axis('off')

# ── colors ───────────────────────────────────────────────────────
ASTRO   = '#1D9E75'
CONN    = '#534AB7'
PUPIL   = '#D85A30'
FCM     = '#3B8BD4'
GREY    = '#5F5E5A'
LIGHT   = '#E8E7E3'
INDEX   = '#2C2C2C'

# ── helpers ──────────────────────────────────────────────────────
def draw_box(x, y, w, h, label, color, fontsize=7, bold=False):
    rect = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.08', facecolor=color, edgecolor='none', alpha=0.18)
    ax.add_patch(rect)
    border = mpatches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle='round,pad=0.08', facecolor='none', edgecolor=color, lw=0.8, alpha=0.5)
    ax.add_patch(border)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            color=color, fontweight=weight, linespacing=1.3)

def draw_arrow(x0, y0, x1, y1, color=GREY):
    arrow = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle='->', mutation_scale=8,
        lw=0.8, color=color, alpha=0.5)
    ax.add_patch(arrow)

# ── column positions ─────────────────────────────────────────────
col_measure = 1.3
col_slope   = 3.6
col_zscore  = 5.9
col_index   = 8.5

# ── COLUMN 1: input measures ────────────────────────────────────
ax.text(col_measure, 5.2, 'Measures', fontsize=8, fontweight='bold',
        ha='center', va='center', color=GREY)
ax.text(col_measure, 4.85, '(sessions 1–N)', fontsize=6,
        ha='center', va='center', color='#999')

measures = [
    ('Astrocyte Ca²⁺\namplitude',  ASTRO, 4.1),
    ('Connectivity\ndifferentiation',  CONN, 3.2),
    ('Pupil\ndilation',  PUPIL, 2.3),
    ('Fecal\ncorticosterone', FCM, 1.4),
]

for label, color, y in measures:
    draw_box(col_measure, y, 2.0, 0.7, label, color)

# ── COLUMN 2: per-animal slopes ──────────────────────────────────
ax.text(col_slope, 5.2, 'Fit slope', fontsize=8, fontweight='bold',
        ha='center', va='center', color=GREY)
ax.text(col_slope, 4.85, r'per animal ($\hat{\beta}_{ik}$)', fontsize=6.5,
        ha='center', va='center', color='#999')

signs = [('−1', ASTRO, 4.1), ('+1', CONN, 3.2), ('−1', PUPIL, 2.3), ('−1', FCM, 1.4)]
for sign_label, color, y in signs:
    draw_box(col_slope, y, 1.6, 0.7,
             f'slope × ({sign_label})\n→ sign-aligned', color, fontsize=6.5)

# arrows: measures → slopes
for _, color, y in measures:
    draw_arrow(col_measure + 1.05, y, col_slope - 0.85, y, color)

# ── COLUMN 3: z-score ────────────────────────────────────────────
ax.text(col_zscore, 5.2, 'Z-score', fontsize=8, fontweight='bold',
        ha='center', va='center', color=GREY)
ax.text(col_zscore, 4.85, 'across animals', fontsize=6.5,
        ha='center', va='center', color='#999')

for _, color, y in measures:
    draw_box(col_zscore, y, 1.6, 0.7, 'z-scored slope\n(+  = adapted)', color, fontsize=6.5)

# arrows: slopes → z-scores
for _, color, y in measures:
    draw_arrow(col_slope + 0.85, y, col_zscore - 0.85, y, color)

# ── COLUMN 4: composite index ────────────────────────────────────
index_y = 2.75
draw_box(col_index, index_y, 1.8, 1.6, '', INDEX)
ax.text(col_index, index_y + 0.35, 'Adaptation\nIndex', fontsize=8,
        ha='center', va='center', color=INDEX, fontweight='bold', linespacing=1.3)
ax.text(col_index, index_y - 0.45,
        r'$\mathrm{AI}_i = \frac{1}{K}\sum_k z_{ik}$',
        fontsize=8, ha='center', va='center', color=GREY)

# arrows: z-scores → index (converge)
for _, color, y in measures:
    draw_arrow(col_zscore + 0.85, y, col_index - 0.95, index_y, color)

# ── bottom annotation ────────────────────────────────────────────
ax.text(5.0, 0.55,
        'Positive AI = greater adaptation  •  '
        'Locomotion analyzed independently, excluded from index',
        fontsize=6.5, ha='center', va='center', color='#999', style='italic')

plt.savefig('outputs/adaptation_index_pipeline.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.savefig('outputs/adaptation_index_pipeline.pdf',
            bbox_inches='tight', facecolor='white')
print('Saved.')

plt.savefig('outputs/adaptation_index_pipeline.png', dpi=200,
            bbox_inches='tight', facecolor='white')
plt.savefig('outputs/adaptation_index_pipeline.pdf',
            bbox_inches='tight', facecolor='white')
print('Saved.')