import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io

# --- Parameters ---
NROWS = 12
NCOLS = 16
ROW_DELAY = 2
EXPOSURE_ROWS = NROWS
NCYCLES = 4

# --- Theme definitions ---
THEMES = {
    'dark': {
        'fig_bg': '#1a1a1e',
        'ax_bg': '#1a1a1e',
        'inactive': np.array([0.12, 0.12, 0.14]),
        'rolling': np.array([0.25, 0.25, 0.28]),
        'cell_edge': '#2a2a2e',
        'title_color': '#d0d0d0',
        'row_label_color': '#666666',
        'led_off_fill': '#2a2a2e',
        'led_off_edge': '#444444',
        'led_off_text': '#555555',
        'rolling_text': '#888888',
        'global_text': '#ffffff',
        'legend_edge': '#555555',
        'legend_text': '#aaaaaa',
        'cycle_text': '#666666',
        'led_label_on': '#aaaaaa',
        'suffix': 'dark',
    },
    'light': {
        'fig_bg': '#ffffff',
        'ax_bg': '#ffffff',
        'inactive': np.array([0.88, 0.88, 0.90]),
        'rolling': np.array([0.72, 0.72, 0.75]),
        'cell_edge': '#a9abb3',
        'title_color': '#111111',
        'row_label_color': '#6e7380',
        'led_off_fill': '#d1d4db',
        'led_off_edge': '#9ea4b0',
        'led_off_text': '#5f6470',
        'rolling_text': '#3d434f',
        'global_text': '#111111',
        'legend_edge': '#8b91a0',
        'legend_text': '#2f3440',
        'cycle_text': '#4b5160',
        'led_label_on': '#2f3440',
        'suffix': 'light',
    },
}

# Shared LED colors
COLOR_488 = np.array([0.1, 0.6, 0.95])
COLOR_408 = np.array([0.55, 0.2, 0.85])
COLOR_GLOBAL_488 = np.array([0.15, 0.75, 1.0])
COLOR_GLOBAL_408 = np.array([0.7, 0.3, 1.0])

# --- Build per-row exposure windows ---
cycle_length = (NROWS + EXPOSURE_ROWS) * ROW_DELAY
total_frames = cycle_length * NCYCLES

frames_data = []
for f in range(total_frames):
    cycle_idx = f // cycle_length
    t = f % cycle_length
    exposing = set()
    for row in range(NROWS):
        row_start = row * ROW_DELAY
        row_end = row_start + EXPOSURE_ROWS * ROW_DELAY
        if row_start <= t < row_end:
            exposing.add(row)
    is_global = (len(exposing) == NROWS)
    led_color = 0 if cycle_idx % 2 == 0 else 1
    frames_data.append({
        'exposing': exposing,
        'is_global': is_global,
        'led': led_color,
        'cycle': cycle_idx,
        't': t,
    })

# --- Render for each theme ---
pad = 0.08
cell_w = 1.0
cell_h = 1.0

for theme_name, T in THEMES.items():
    fig, ax = plt.subplots(figsize=(8.5, 5.5), facecolor=T['fig_bg'])
    pil_frames = []
    grid_center_x = NCOLS * cell_w / 2
    grid_center_y = NROWS * cell_h / 2

    for fi, fd in enumerate(frames_data):
        ax.clear()
        ax.set_facecolor(T['ax_bg'])
        ax.set_xlim(-1.8, NCOLS * cell_w + 4.5)
        ax.set_ylim(-3.8, NROWS * cell_h + 3.5)
        ax.set_aspect('equal')
        ax.axis('off')

        # Title
        ax.text(grid_center_x, NROWS * cell_h + 2.8,
                'sCMOS Rolling Shutter — Virtual LED Shutter',
                ha='center', va='center', fontsize=13, fontweight='bold',
                color=T['title_color'], family='monospace')

        # Pixel grid
        for row in range(NROWS):
            for col in range(NCOLS):
                y = (NROWS - 1 - row) * cell_h
                x = col * cell_w
                if row in fd['exposing']:
                    if fd['is_global']:
                        c = COLOR_GLOBAL_488 if fd['led'] == 0 else COLOR_GLOBAL_408
                    else:
                        c = T['rolling']
                else:
                    c = T['inactive']
                rect = Rectangle((x + pad, y + pad),
                                  cell_w - 2*pad, cell_h - 2*pad,
                                  facecolor=c, edgecolor=T['cell_edge'],
                                  linewidth=0.5, zorder=2)
                ax.add_patch(rect)

        # Row labels
        for row in range(NROWS):
            y = (NROWS - 1 - row) * cell_h + cell_h / 2
            ax.text(-1.2, y, f'R{row}', ha='center', va='center',
                    fontsize=6, color=T['row_label_color'], family='monospace')

        # --- Right-side status panel ---
        panel_x = NCOLS * cell_w + 1.8
        led_circle_y = grid_center_y + 3.0
        led_wavelength_y = grid_center_y + 1.8
        led_state_y = grid_center_y + 0.8
        mode_label_y = grid_center_y - 0.2

        if fd['is_global']:
            led_label = "488 nm" if fd['led'] == 0 else "408 nm"
            led_c = COLOR_GLOBAL_488 if fd['led'] == 0 else COLOR_GLOBAL_408

            # LED circle
            circ = plt.Circle((panel_x, led_circle_y), 0.55,
                              facecolor=led_c, edgecolor='white', linewidth=1.5,
                              alpha=0.95, zorder=3)
            ax.add_patch(circ)

            # LED wavelength label
            ax.text(panel_x, led_wavelength_y, led_label,
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color=led_c, family='monospace')

            # LED ON label
            ax.text(panel_x, led_state_y, 'LED ON',
                    ha='center', va='center', fontsize=8,
                    color=T['led_label_on'], family='monospace')

            # Global exposure label
            ax.text(panel_x, mode_label_y, 'GLOBAL\nEXPOSURE',
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color=T['global_text'], family='monospace',
                    linespacing=1.5)
        else:
            # LED off
            circ = plt.Circle((panel_x, led_circle_y), 0.55,
                              facecolor=T['led_off_fill'],
                              edgecolor=T['led_off_edge'], linewidth=1.0,
                              alpha=0.7, zorder=3)
            ax.add_patch(circ)

            ax.text(panel_x, led_wavelength_y, '---',
                    ha='center', va='center', fontsize=10,
                    color=T['led_off_text'], family='monospace')

            ax.text(panel_x, led_state_y, 'LED OFF',
                    ha='center', va='center', fontsize=8,
                    color=T['led_off_text'], family='monospace')

            n_exp = len(fd['exposing'])
            ax.text(panel_x, mode_label_y,
                    f'ROLLING\n{n_exp}/{NROWS} rows',
                    ha='center', va='center', fontsize=9,
                    color=T['rolling_text'], family='monospace',
                    linespacing=1.5)

        # --- Bottom legend (2x2 centered relative to grid) ---
        legend_items = [
            ('Inactive', T['inactive']),
            ('Exposing (partial)', T['rolling']),
            ('488 nm (blue)', COLOR_488),
            ('408 nm (violet)', COLOR_408),
        ]
        legend_centers_x = [grid_center_x - 3.4, grid_center_x + 3.4]
        legend_row_ys = [-1.55, -2.35]

        for idx, (label, color) in enumerate(legend_items):
            row = idx // 2
            col = idx % 2
            cx = legend_centers_x[col]
            ly = legend_row_ys[row]
            rect = Rectangle((cx - 1.45, ly), 0.6, 0.5,
                              facecolor=color, edgecolor=T['legend_edge'],
                              linewidth=0.5)
            ax.add_patch(rect)
            ax.text(cx - 0.6, ly + 0.25, label,
                    ha='left', va='center', fontsize=7,
                    color=T['legend_text'], family='monospace')

        # Cycle counter
        ax.text(grid_center_x, -3.35,
                f'Frame cycle {fd["cycle"]+1}/{NCYCLES}',
                ha='center', va='center', fontsize=8,
                color=T['cycle_text'], family='monospace')

        # Render
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor=fig.get_facecolor(), pad_inches=0.3)
        buf.seek(0)
        pil_frames.append(Image.open(buf).copy())
        buf.close()

    plt.close(fig)

    # Save GIF
    out = f'outputs/rolling_shutter_virtual_led_{T["suffix"]}.gif'
    pil_frames[0].save(out, save_all=True, append_images=pil_frames[1:],
                       duration=80, loop=0)
    print(f"[{theme_name}] Saved {len(pil_frames)} frames → {out}")