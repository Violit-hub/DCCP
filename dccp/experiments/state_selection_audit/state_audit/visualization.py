"""生成无需 Web 服务的离线 HTML、接触图和选择时间轴。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .stage_gates import stage_status


METHOD_COLORS = {
    "random": "#64748b",
    "curvature": "#ef4444",
    "entropy": "#2563eb",
    "joint": "#16a34a",
    "oracle": "#9333ea",
}
SERIES_COLORS = {
    "progress": "#111827",
    "curvature": "#ef4444",
    "entropy": "#2563eb",
    "joint score": "#16a34a",
}


def _read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) if mask is None else (np.asarray(mask, dtype=bool) & np.isfinite(values))
    output = np.zeros_like(values)
    if not np.any(valid):
        return output
    low = float(np.min(values[valid]))
    high = float(np.max(values[valid]))
    if high > low:
        output[valid] = (values[valid] - low) / (high - low)
    return output


def _save_contact_sheet(
    store: ArtifactStore,
    trajectory: dict[str, Any],
    selected_by_index: dict[int, list[str]],
    target: Path,
) -> None:
    steps = trajectory.get("steps", [])
    cell_width, image_height, label_height, columns = 176, 144, 36, 6
    rows = max(1, (len(steps) + columns - 1) // columns)
    sheet = Image.new("RGB", (cell_width * columns, (image_height + label_height) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for position, step in enumerate(steps):
        decision_index = int(step["decision_index"])
        with Image.open(store.run_dir / step["frame_path"]) as source:
            frame = source.convert("RGB")
            frame.thumbnail((cell_width - 8, image_height - 8))
        x = (position % columns) * cell_width
        y = (position // columns) * (image_height + label_height)
        selected_methods = selected_by_index.get(decision_index, [])
        border = METHOD_COLORS.get("joint", "#16a34a") if selected_methods else "#cbd5e1"
        draw.rectangle((x + 1, y + 1, x + cell_width - 2, y + image_height - 2), outline=border, width=4)
        sheet.paste(frame, (x + (cell_width - frame.width) // 2, y + (image_height - frame.height) // 2))
        method_text = ",".join(method[:3] for method in selected_methods) or "-"
        draw.text((x + 5, y + image_height + 2), f"t={decision_index}  {method_text}", fill="black")
        draw.text((x + 5, y + image_height + 17), f"H={float(step['entropy']):.3f}", fill="#334155")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    sheet.save(temporary, format="JPEG", quality=90)
    temporary.replace(target)
    target.chmod(0o660)


def _selection_svg(
    store: ArtifactStore,
    trajectory_id: str,
    selections: list[dict[str, Any]],
    target: Path,
) -> bool:
    score_path = store.selection_dir / f"{trajectory_id}_scores.npz"
    if not score_path.exists():
        return False
    with np.load(score_path, allow_pickle=False) as data:
        starts = data["valid_start_indices"].astype(int)
        progress = data["progress_scores"]
        curvature = data["curvature_scores"]
        entropy = data["entropy_scores"]
        decision = data["decision_scores"]
        valid_mask = data["valid_mask"].astype(bool)
    if len(starts) == 0:
        return False

    width, height = 1080, 360
    left, right, top, bottom = 60, 25, 55, 55
    plot_width, plot_height = width - left - right, height - top - bottom

    def x_at(position: int) -> float:
        return left + (plot_width * position / max(len(starts) - 1, 1))

    def points(values: np.ndarray) -> str:
        normalised = _normalise(values, valid_mask)
        return " ".join(
            f"{x_at(index):.1f},{top + (1.0 - value) * plot_height:.1f}"
            for index, value in enumerate(normalised)
        )

    series = {
        "progress": progress,
        "curvature": curvature,
        "entropy": entropy,
        "joint score": decision,
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#f8fafc" stroke="#cbd5e1"/>',
    ]
    for grid in range(5):
        y = top + grid * plot_height / 4
        parts.append(f'<line x1="{left}" x2="{left + plot_width}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e2e8f0"/>')
    for name, values in series.items():
        parts.append(
            f'<polyline points="{points(values)}" fill="none" stroke="{SERIES_COLORS[name]}" '
            'stroke-width="2.5" stroke-linejoin="round"/>'
        )
    for row_number, selection in enumerate(selections):
        method = str(selection["method"])
        color = METHOD_COLORS.get(method, "#475569")
        for decision_index in selection.get("selected_indices", []):
            matches = np.flatnonzero(starts == int(decision_index))
            if not len(matches):
                continue
            x = x_at(int(matches[0]))
            parts.append(
                f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{top + plot_height}" '
                f'stroke="{color}" stroke-width="2" stroke-dasharray="5,4"/>'
            )
            parts.append(
                f'<text x="{x + 3:.1f}" y="{18 + row_number * 13}" fill="{color}" '
                f'font-size="11">{html.escape(method)}@{int(decision_index)}</text>'
            )
    for index, decision_index in enumerate(starts):
        if index % max(1, len(starts) // 10) == 0 or index == len(starts) - 1:
            x = x_at(index)
            parts.append(f'<text x="{x:.1f}" y="{height - 25}" text-anchor="middle" font-size="11">{int(decision_index)}</text>')
    legend_x = left
    for name, color in SERIES_COLORS.items():
        parts.append(f'<line x1="{legend_x}" x2="{legend_x + 20}" y1="{height - 8}" y2="{height - 8}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x + 25}" y="{height - 4}" font-size="11">{html.escape(name)}</text>')
        legend_x += 145
    parts.append("</svg>")
    ArtifactStore.write_text_atomic(target, "\n".join(parts))
    return True


def _table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return '<p class="muted">尚无数据</p>'
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields)
        body.append(f"<tr>{cells}</tr>")
    return f"<div class=table-wrap><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def generate_inspection_site(cfg: AuditConfig) -> Path:
    """可在任意阶段重复运行；页面只展示当时已经存在的产物。"""
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    inspection_dir = store.run_dir / "inspection"
    assets_dir = inspection_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    trajectories = _read_json(store.nominal_dir / "trajectories.json", [])
    selection_rows = ArtifactStore.read_jsonl(store.selection_dir / "selected_states.jsonl")
    branch_rows = ArtifactStore.read_jsonl(store.branch_dir / "branch_results.jsonl")
    summary_rows = _read_json(store.report_dir / "summary.json", [])
    statuses = stage_status(cfg)

    selections_by_traj: dict[str, list[dict[str, Any]]] = {}
    for row in selection_rows:
        selections_by_traj.setdefault(str(row["trajectory_id"]), []).append(row)

    trajectory_sections = []
    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        selections = selections_by_traj.get(trajectory_id, [])
        selected_by_index: dict[int, list[str]] = {}
        for row in selections:
            for decision_index in row.get("selected_indices", []):
                selected_by_index.setdefault(int(decision_index), []).append(str(row["method"]))
        contact_path = assets_dir / f"{trajectory_id}_contact.jpg"
        _save_contact_sheet(store, trajectory, selected_by_index, contact_path)
        svg_path = assets_dir / f"{trajectory_id}_selection.svg"
        has_svg = _selection_svg(store, trajectory_id, selections, svg_path)
        method_rows = [
            {
                "method": row["method"],
                "selected_indices": row.get("selected_indices", []),
                "eligible_count": len(row.get("eligible_indices", [])),
            }
            for row in selections
        ]
        frame_cards = []
        for step in trajectory.get("steps", []):
            decision_index = int(step["decision_index"])
            frame_path = html.escape(str(step["frame_path"]))
            method_badges = " ".join(
                f'<span class="badge" style="background:{METHOD_COLORS.get(method, "#475569")}">{html.escape(method)}</span>'
                for method in selected_by_index.get(decision_index, [])
            )
            frame_cards.append(
                f'<a class="frame" href="../{frame_path}"><img src="../{frame_path}" loading="lazy">'
                f'<span>t={decision_index} · H={float(step["entropy"]):.4f}</span>{method_badges}</a>'
            )
        trajectory_sections.append(
            f"""
            <section>
              <h2>{html.escape(trajectory_id)}</h2>
              <p>success={trajectory.get('success')} · decisions={trajectory.get('num_decisions')} · finish_step={trajectory.get('finish_low_level_step')}</p>
              <a href="assets/{contact_path.name}"><img class="wide" src="assets/{contact_path.name}" alt="contact sheet"></a>
              {f'<img class="wide" src="assets/{svg_path.name}" alt="selection timeline">' if has_svg else '<p class="muted">选择阶段未完成，暂无曲线。</p>'}
              {_table(method_rows, ['method', 'selected_indices', 'eligible_count'])}
              <details><summary>逐张查看原始决策帧</summary><div class="frames">{''.join(frame_cards)}</div></details>
            </section>
            """
        )

    latest_branches: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in branch_rows:
        key = (
            row.get("trajectory_id"),
            row.get("decision_index"),
            row.get("candidate_index"),
            row.get("evaluation_seed"),
        )
        latest_branches[key] = row
    branch_cards = []
    for key, row in sorted(latest_branches.items()):
        video_path = row.get("video_path")
        video = (
            f'<video controls preload="metadata" width="320" src="../{html.escape(video_path)}"></video>'
            if video_path and (store.run_dir / video_path).exists()
            else '<span class="muted">未录制视频</span>'
        )
        branch_cards.append(
            f'<div class="branch"><b>{html.escape(str(key))}</b><br>'
            f'status={html.escape(str(row.get("status")))} · success={row.get("success")}<br>{video}</div>'
        )

    status_cards = "".join(
        f'<div class="status {html.escape(str(item["state"]).lower())}"><b>{html.escape(item["stage"])}</b>'
        f'<span>{html.escape(str(item["state"]))}</span><small>{html.escape(str(item["detail"]))}</small></div>'
        for item in statuses
    )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(cfg.run_name)} 状态选择检查</title>
<style>
body{{font-family:system-ui,sans-serif;margin:0;background:#f1f5f9;color:#0f172a}}main{{max-width:1180px;margin:auto;padding:24px}}
h1,h2{{margin-bottom:8px}}section{{background:white;border-radius:12px;padding:18px;margin:18px 0;box-shadow:0 1px 4px #cbd5e1}}
.statuses,.branches{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}}.status,.branch{{background:white;border:1px solid #cbd5e1;border-radius:9px;padding:12px}}
.status span{{display:block;font-weight:700;color:#2563eb;margin:5px 0}}.status small{{display:block;word-break:break-all;color:#64748b}}.passed span,.done span,.frozen span{{color:#16a34a}}
.wide{{max-width:100%;height:auto;border:1px solid #cbd5e1;border-radius:7px}}.muted{{color:#64748b}}.table-wrap{{overflow:auto}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #e2e8f0;text-align:left;padding:7px;white-space:nowrap}}video{{margin-top:8px;max-width:100%}}
.frames{{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-top:12px}}.frame{{color:#0f172a;text-decoration:none;border:1px solid #cbd5e1;border-radius:7px;padding:6px;background:#f8fafc}}.frame img{{width:100%;height:130px;object-fit:cover;display:block;margin-bottom:5px}}.frame span{{display:inline-block;margin-right:4px}}.badge{{color:white;border-radius:10px;padding:2px 6px;font-size:10px}}
</style></head><body><main>
<h1>{html.escape(cfg.run_name)} · Coffee 状态选择审计</h1>
<p>此页面可在每个阶段后重新生成；先人工验收当前产物，再执行下一阶段。</p>
<div class="statuses">{status_cards}</div>
{''.join(trajectory_sections) or '<section><p class="muted">尚未采集 nominal 轨迹。</p></section>'}
<section><h2>分支结果与视频</h2><div class="branches">{''.join(branch_cards) or '<p class="muted">尚未评估分支。</p>'}</div></section>
<section><h2>最终汇总</h2>{_table(summary_rows, ['method','num_trajectories','mean_improvement','opportunity_rate','macro_capture_ratio','micro_capture_ratio'])}</section>
</main></body></html>"""
    index_path = inspection_dir / "index.html"
    ArtifactStore.write_text_atomic(index_path, document)
    ArtifactStore.write_json_atomic(
        inspection_dir / "inspection_manifest.json",
        {
            "run_name": cfg.run_name,
            "stage_status": statuses,
            "num_trajectories": len(trajectories),
            "num_selection_rows": len(selection_rows),
            "num_branch_rows": len(branch_rows),
            "index": str(index_path),
        },
    )
    return index_path
