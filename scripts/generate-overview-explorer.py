from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_plotly():
    try:
        import plotly.graph_objects as go  # type: ignore[import]
        from plotly.subplots import make_subplots  # type: ignore[import]
    except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
        raise ModuleNotFoundError(
            "plotly is required for the standalone explorer builder",
            name=exc.name,
        ) from exc
    return go, make_subplots


def _build_explorer_figure(
    meso_traces: np.ndarray,
    roi_names: Sequence[str],
    time_axis: np.ndarray,
    *,
    pupil: Optional[Tuple[np.ndarray, np.ndarray]],
    treadmill: Optional[Tuple[np.ndarray, np.ndarray]],
    title: str,
) -> object:
    go, make_subplots = _require_plotly()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
    )

    for trace, name in zip(meso_traces, roi_names):
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=trace,
                name=name,
                mode="lines",
                line=dict(width=1.4),
                hovertemplate="<b>%{meta}</b><br>%{x:.3f} : %{y:.4f}<extra></extra>",
                meta=name,
            ),
            row=1,
            col=1,
        )

    if pupil is not None:
        t_pupil, y_pupil = pupil
        fig.add_trace(
            go.Scatter(
                x=t_pupil,
                y=y_pupil,
                name="Pupil diameter (mm)",
                mode="lines",
                line=dict(color="#EF553B", width=1.6),
                hovertemplate="<b>Pupil</b><br>%{x:.3f} : %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    if treadmill is not None:
        t_treadmill, y_treadmill = treadmill
        fig.add_trace(
            go.Scatter(
                x=t_treadmill,
                y=y_treadmill,
                name="Treadmill speed (mm)",
                mode="lines",
                line=dict(color="#00CC96", width=1.6),
                hovertemplate="<b>Treadmill</b><br>%{x:.3f} : %{y:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    fig.update_layout(
        title=None,
        template="plotly_white",
        height=860,
        width=1280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0, xanchor="left"),
        margin=dict(t=60, r=40, l=60, b=40),
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="ΔF/F", row=1, col=1)
    fig.update_yaxes(title_text="Pupil (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (mm)", row=3, col=1)
    return fig


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_index(entries: List[Dict[str, str]], output_dir: Path) -> Path:
    manifest = json.dumps(entries, indent=2)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Mesomap Explorer Index</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }}
    label {{ font-weight: 600; }}
    select {{ min-width: 140px; padding: 4px 6px; }}
    iframe {{ width: 100%; height: 80vh; border: 1px solid #ccc; border-radius: 4px; }}
    .warning {{ color: #b33; font-weight: 600; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Mesomap Explorer Browser</h1>
  <div class=\"controls\">
    <label for=\"subject\">Subject</label>
    <select id=\"subject\"></select>
    <label for=\"session\">Session</label>
    <select id=\"session\"></select>
    <label for=\"task\">Task</label>
    <select id=\"task\"></select>
  </div>
  <div id=\"status\" class=\"warning\"></div>
  <iframe id=\"viewer\" title=\"Mesomap Explorer\" src=\"\"></iframe>

  <script>
    const manifest = {manifest};

    function uniq(values) {{ return [...new Set(values)]; }}

    const bySubject = new Map();
    manifest.forEach(entry => {{
      const subj = entry.subject || 'unknown';
      if (!bySubject.has(subj)) bySubject.set(subj, []);
      bySubject.get(subj).push(entry);
    }});

    const subjectSel = document.getElementById('subject');
    const sessionSel = document.getElementById('session');
    const taskSel = document.getElementById('task');
    const iframe = document.getElementById('viewer');
    const status = document.getElementById('status');

    function setStatus(msg) {{ status.textContent = msg || ''; }}

    function populateSubjects() {{
      subjectSel.innerHTML = '';
      uniq(Array.from(bySubject.keys())).forEach(subj => {{
        const opt = document.createElement('option');
        opt.value = subj; opt.textContent = subj; subjectSel.appendChild(opt);
      }});
    }}

    function populateSessions(subj) {{
      sessionSel.innerHTML = '';
      const entries = bySubject.get(subj) || [];
      const sessions = uniq(entries.map(e => e.session || 'unknown'));
      sessions.forEach(sess => {{
        const opt = document.createElement('option');
        opt.value = sess; opt.textContent = sess; sessionSel.appendChild(opt);
      }});
    }}

    function populateTasks(subj, sess) {{
      taskSel.innerHTML = '';
      const entries = (bySubject.get(subj) || []).filter(e => (e.session || 'unknown') === sess);
      entries.forEach(e => {{
        const opt = document.createElement('option');
        opt.value = e.html; opt.textContent = e.task;
        taskSel.appendChild(opt);
      }});
    }}

    function updateViewer() {{
      const html = taskSel.value;
      if (!html) {{
        iframe.src = '';
        setStatus('No run selected.');
        return;
      }}
      setStatus('');
      iframe.src = html;
    }}

    subjectSel.addEventListener('change', () => {{
      populateSessions(subjectSel.value);
      populateTasks(subjectSel.value, sessionSel.value);
      updateViewer();
    }});

    sessionSel.addEventListener('change', () => {{
      populateTasks(subjectSel.value, sessionSel.value);
      updateViewer();
    }});

    taskSel.addEventListener('change', updateViewer);

    populateSubjects();
    if (subjectSel.options.length) {{
      subjectSel.selectedIndex = 0;
      populateSessions(subjectSel.value);
      if (sessionSel.options.length) {{
        sessionSel.selectedIndex = 0;
        populateTasks(subjectSel.value, sessionSel.value);
      }}
    }}
    if (taskSel.options.length) {{
      taskSel.selectedIndex = 0;
      updateViewer();
    }} else {{
      setStatus('No explorer HTML files found.');
    }}
  </script>
</body>
</html>
"""
    index_path = output_dir / "mesomap_explorers.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def build_explorers_from_pickle(
    dataset_path: Path,
    *,
    output_dir: Optional[Path] = None,
    roi_limit: Optional[int] = None,
) -> List[Path]:
    dataset = pd.read_pickle(dataset_path)

    out_dir = output_dir or dataset_path.parent / "mesomap_explorers"
    _ensure_dir(out_dir)

    generated: List[Path] = []
    manifest: List[Dict[str, str]] = []

    for key in dataset.index:
        subject, session, task = (str(key[0]), str(key[1]), str(key[2]))

        meso_block = dataset.loc[key, "mesomap"]
        t_meso = np.asarray(meso_block["time_elapsed_s"])
        roi_names = [
            name
            for name in meso_block.index
            if name != "time_elapsed_s" and np.asarray(meso_block[name]).ndim == 1
        ]
        if roi_limit is not None:
            roi_names = roi_names[: int(roi_limit)]
        meso_traces_list = [np.asarray(meso_block[name]) for name in roi_names]
        meso_lengths = {trace.shape for trace in meso_traces_list}
        if len(meso_lengths) != 1:
            raise ValueError(f"mesomap ROI traces have inconsistent shapes: {sorted(meso_lengths)}")
        meso_traces = np.stack(meso_traces_list)

        t_pupil = np.asarray(dataset.loc[key, ("pupil", "time_elapsed_s")])
        y_pupil = np.asarray(dataset.loc[key, ("pupil", "pupil_diameter_mm")])
        pupil = (t_pupil, y_pupil)

        t_treadmill = np.asarray(dataset.loc[key, ("treadmill", "time_elapsed_s")])
        y_treadmill = np.asarray(dataset.loc[key, ("treadmill", "speed_mm")])
        if t_treadmill.ndim != 1 or y_treadmill.ndim != 1 or t_treadmill.size == 0 or y_treadmill.size == 0:
            treadmill = (np.asarray([]), np.asarray([]))
        else:
            order = np.argsort(t_treadmill)
            t_treadmill = t_treadmill[order]
            y_treadmill = y_treadmill[order]
            if t_treadmill.size:
                _, unique_idx = np.unique(t_treadmill, return_index=True)
                t_treadmill = t_treadmill[unique_idx]
                y_treadmill = y_treadmill[unique_idx]
            indices = np.searchsorted(t_treadmill, t_meso, side="right") - 1
            indices = np.clip(indices, 0, len(y_treadmill) - 1)
            treadmill = (t_meso, y_treadmill[indices])

        title = f"Subject {subject} | Session {session} | Task {task}"
        fig = _build_explorer_figure(
            meso_traces,
            roi_names,
            t_meso,
            pupil=pupil,
            treadmill=treadmill,
            title=title,
        )

        file_label = f"{subject}_{session}_{task}_explorer.html".replace(" ", "_")
        html_path = out_dir / file_label
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        generated.append(html_path)
        manifest.append(
            {
                "subject": subject,
                "session": session,
                "task": task,
                "html": html_path.name,
            }
        )
        print(f"[OK] {subject} / {session} / {task} -> {html_path}")

    if manifest:
        _build_index(manifest, out_dir)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Plotly explorer HTML files without interpolating auxiliary traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-pickle",
        type=Path,
        required=True,
        help="Dataset pickle containing mesomap, pupil, and treadmill arrays",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for explorer HTML files (defaults next to the pickle)",
    )
    parser.add_argument(
        "--roi-limit",
        type=int,
        default=None,
        help="Limit the number of ROIs plotted per explorer",
    )
    args = parser.parse_args()

    try:
        generated = build_explorers_from_pickle(
            args.dataset_pickle,
            output_dir=args.output_dir,
            roi_limit=args.roi_limit,
        )
    except Exception as exc:
        print(f"[FAIL] {exc}")
        return 1

    if not generated:
        print("No explorers generated. Verify that the dataset contains mesomap traces.")
        return 1

    print("\nGenerated explorers:")
    for path in generated:
        try:
            rel = path.relative_to(args.output_dir) if args.output_dir else path.name
        except Exception:
            rel = path
        print(f" - {rel}")
    if args.output_dir is not None:
        print(f"Index written to {args.output_dir / 'mesomap_explorers.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
