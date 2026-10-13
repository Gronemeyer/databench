# Publication-ready matplotlib — concise tips

Distilled from building the 4 Hz timing-geometry figures. The goal is figures
that drop straight into a manuscript: correct physical size, a single legible
font hierarchy, no overlapping text, vector output, and *the data, not the
prose*, on the figure face.

---

## 1. Size the figure in inches, to the column

Journals specify column widths, not pixels. Set `figsize` to the **final
printed size** and never scale afterward — scaling a figure rescales every font
with it.

| Target | width |
|---|---|
| single column | ~3.4 in |
| 1.5 column | ~5.0 in |
| double / full width | ~7.0 in |
| square supporting panel | ~3.4 × 3.4 in |

```python
PANEL_IN = 3.4
fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN), constrained_layout=True)
```

## 2. One font hierarchy, applied via `rc_context`

Define the type scale **once** and wrap every figure in it, so the whole set is
consistent and you never hand-set `fontsize=` per element. Sub-axes elements
(colorbar ticks, polar ticks) still need explicit small sizes.

```python
PUB_RC = {
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "axes.linewidth": 0.7, "legend.fontsize": 6.5,
    "savefig.dpi": 300, "figure.dpi": 150,
}
with plt.rc_context(PUB_RC):
    fig, ax = plt.subplots(figsize=(PANEL_IN, PANEL_IN), constrained_layout=True)
    ...
```
Rule of thumb at print size: **6–8 pt** everywhere. Title need not dwarf labels.

## 3. `constrained_layout=True`, not `tight_layout()`

`constrained_layout` reserves space for colorbars, shared axes, and gridspecs
without the "Axes not compatible with tight_layout" warning. Set it at figure
creation.

- **Insets are the exception**: `constrained_layout` ignores `inset_axes`, and
  `tight_layout` *warns* on them. If you must use `tight_layout`, **add insets
  after** calling it. With `constrained_layout` just add them last anyway.

## 4. Always export vector (SVG **and** PDF)

Raster is for previews only. Emit a vector companion next to every figure.

```python
SAVE_FORMATS = ("pdf",)            # alongside the .svg
fig.savefig("fig.svg"); [fig.savefig(f"fig.{e}") for e in SAVE_FORMATS]
```
Keep raster fallbacks (PNG) at **≥300 dpi**. Vector keeps text selectable and
lines crisp at any zoom.

## 5. Put the statistics in a caption, not on the canvas

A baked-in stats paragraph is the #1 source of text collisions and clutter.
The figure shows data; the numbers go in a caption string you save beside it
and paste into the manuscript.

```python
caption = f"... slope {slope:+.2f} ms/mm, r={r:+.2f}; planar R²={r2:.2f} ..."
(run.plots_dir / "fig_caption.txt").write_text(caption)
```
Exception: a *single* number that names the panel (e.g. `r = +0.93`) is fine in
the title.

## 6. Declutter dense labels: threshold + halo

Do not label every element on a crowded map.

- **Drop small items**: compute polygon/marker area and label only those above a
  percentile cutoff; rely on the colorbar for the rest.
- **Strip redundant text**: `L_VISp → VISp` when the layout already shows side.
- **Halo for legibility** over colored fills, via `path_effects`:

```python
import matplotlib.patheffects as pe
ax.text(x, y, label, fontsize=5, ha="center", va="center",
        path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])
```
If you need auto-repel and can add a dependency, `adjustText` does it; otherwise
threshold + halo gets you most of the way with zero deps.

## 7. Show all the data, but make the untrustworthy recede

Don't silently drop categories — a reader can't tell missing-because-excluded
from missing-because-absent. Draw everything; encode *confidence* in the style:

- high-confidence → full color
- measured but low-confidence → desaturated grey **hatch** (`hatch="////"`)
- no data → white outline only

Iterate the **superset** of geometry ∪ measured so nothing vanishes, and state
the encoding in the caption.

## 8. Markers: small, thin-edged

Default scatter markers are too big and their black rims dominate at print size.

```python
ax.scatter(x, y, s=22, edgecolor="0.25", linewidths=0.4)   # not s=70, lw=1
ax.scatter(x_edge, y_edge, clip_on=False)                  # show points on the axis edge
```

## 9. Colorbars: clip to the data, shrink, size the text

- Clip the norm to the **range you actually trust**, not global outliers — more
  of the colormap maps to visible contrast.
- `fraction=0.046, pad=0.02` keeps the bar from eating the axes; set label and
  tick sizes explicitly (rc doesn't reach them).

```python
from matplotlib.colors import Normalize
norm = Normalize(-vmax, vmax)                  # symmetric for signed data
cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
cb.set_label("relative timing (ms)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
```

## 10. Colormap choice encodes the data type

- **Diverging** (`RdBu_r`, `coolwarm`) for signed values around a meaningful
  zero — keep it **symmetric** (`vmin=-vmax, vmax=+vmax`) so white = 0.
- **Sequential** (`viridis`) for magnitudes / [0,1] scores.
- Never `jet`/`rainbow` (perceptually non-uniform, not colorblind-safe).

## 11. Link axes that mean the same thing

If a quantity is a color on one panel and an x-axis on its neighbor, give them
**identical limits** and say so in the caption — the reader reads across panels
for free.

```python
ax_map_norm = Normalize(-vmax, vmax)
ax_profile.set_xlim(-vmax, vmax)               # same range as the colorbar
```
Marginal panels that share a spatial axis should `sharey`/`sharex` the main axes
(`fig.add_subplot(gs[0,1], sharey=ax)`).

## 12. Preserve geometry; indicate direction without implying origin

- `ax.set_aspect("equal")` for any spatial/anatomical map **and** any
  identity-line/agreement plot — distortion misleads.
- Show a **direction** (e.g. a gradient axis) as a small **corner compass** via
  `inset_axes([x, y, w, h])` in an empty corner, *not* an arrow planted in the
  data — a planted arrow's tail reads as an origin/source you may not be claiming.

```python
comp = ax.inset_axes([0.82, 0.80, 0.17, 0.17]); comp.axis("off"); comp.set_aspect("equal")
comp.annotate("", xy=(ux, uy), xytext=(-ux, -uy), arrowprops=dict(arrowstyle="-|>", lw=1.5))
```

## 13. Reference geometry for orientation

- Identity/agreement plots: dashed `y=x` (`color="0.6", ls="--", lw=0.8`) and
  faint zero lines (`color="0.9", lw=0.6`).
- Equal limits on both axes (`set_xlim == set_ylim`) so the diagonal is at 45°.

## 14. Circular data: show dispersion, not just the mean

A mean-direction arrow alone overstates certainty. Add a **±1 circular-SD
wedge** and report the resultant length R.

```python
sd = np.degrees(np.sqrt(-2*np.log(R)))                       # circular SD
ax.fill_between(np.linspace(np.radians(m-sd), np.radians(m+sd), 60), 0, rmax, alpha=0.1)
```

## 15. Be numerically honest in the text you print

- Don't let formatting fake precision: `r=+1.00` from `:.2f` hides `0.9979` —
  use enough decimals (`:.3f`) for near-ceiling values.
- Pair every central value with a spread (SEM, IQR, circular SD).
- Prefer reporting a **range of local estimates** over one global number when a
  fit's R² shows the global model is an approximation (e.g. local k-NN speeds
  vs one planar speed).

---

### Minimal template

```python
PUB_RC = {"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
          "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.7,
          "savefig.dpi": 300}

with plt.rc_context(PUB_RC):
    fig, ax = plt.subplots(figsize=(3.4, 3.4), constrained_layout=True)
    ax.scatter(x, y, s=22, edgecolor="0.25", linewidths=0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (unit)"); ax.set_ylabel("y (unit)")
    ax.set_title(f"thing · r={r:+.2f}")
    for ext in ("svg", "pdf"):
        fig.savefig(f"thing.{ext}", bbox_inches="tight")
(out_dir / "thing_caption.txt").write_text(detailed_stats_caption)
```
