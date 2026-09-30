"""Offline inspection site for every stage and every candidate branch."""

from __future__ import annotations

import html
import json
from collections import defaultdict

import numpy as np

from .artifacts import AuditStore
from .config import AuditConfig
from .stages import stage_status


def _load_json(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _video(path: str | None, label: str) -> str:
    if not path:
        return f'<span class="missing">{html.escape(label)} missing</span>'
    escaped = html.escape(path)
    return f'<div class="clip"><small>{html.escape(label)}</small><video controls preload="metadata" src="../{escaped}"></video></div>'


def _table(rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return '<p class="missing">No records yet.</p>'
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _scatter_svg(pair_rows: list[dict]) -> str:
    if not pair_rows:
        return '<p class="missing">Metrics not available.</p>'
    width, height, pad = 620, 300, 42
    xs = [float(row["predicted_abs_margin"]) for row in pair_rows]
    ys = [float(row["actual_success_gap"]) for row in pair_rows]
    xmax = max(max(xs), 0.1)
    parts = [f'<svg viewBox="0 0 {width} {height}" class="plot">', '<rect width="100%" height="100%" fill="white"/>']
    zero_y = pad + (1.0 - (0.0 + 1.0) / 2.0) * (height - 2 * pad)
    parts.append(f'<line x1="{pad}" x2="{width-pad}" y1="{zero_y}" y2="{zero_y}" stroke="#94a3b8"/>')
    for x, y in zip(xs, ys):
        px = pad + x / xmax * (width - 2 * pad)
        py = pad + (1.0 - (y + 1.0) / 2.0) * (height - 2 * pad)
        color = "#16a34a" if y > 0 else ("#dc2626" if y < 0 else "#64748b")
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" opacity=".8"/>')
    parts.extend([
        f'<text x="{width/2}" y="{height-7}" text-anchor="middle">predicted |margin|</text>',
        '<text x="14" y="150" transform="rotate(-90 14 150)" text-anchor="middle">actual success gap</text>',
        '</svg>',
    ])
    return "".join(parts)


def generate_inspection(cfg: AuditConfig):
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    state_payload = _load_json(store.states / "manifest.json", {"states": []})
    candidate_payload = _load_json(store.candidates / "manifest.json", {"candidates": []})
    states = state_payload["states"]
    candidates = candidate_payload["candidates"]
    predictions = AuditStore.read_jsonl(store.world_model / "predictions.jsonl")
    scores = AuditStore.read_jsonl(store.labels / "prediction_scores.jsonl")
    labels = {str(row["state_id"]): row for row in AuditStore.read_jsonl(store.labels / "labels.jsonl")}
    outcomes = AuditStore.read_jsonl(store.simulator / "outcomes.jsonl")
    report = _load_json(store.reports / "validity_report.json", {})

    latest_predictions = {}
    for row in predictions:
        if row.get("status") == "OK":
            latest_predictions[(str(row["state_id"]), int(row["candidate_index"]), int(row["wm_seed"]))] = row
    latest_scores = {}
    for row in scores:
        if row.get("status") == "OK":
            latest_scores[(str(row["state_id"]), int(row["candidate_index"]), int(row["wm_seed"]))] = row
    latest_outcomes = {}
    for row in outcomes:
        if row.get("status") == "OK":
            latest_outcomes[(str(row["state_id"]), int(row["candidate_index"]), int(row["evaluation_seed"]))] = row
    candidates_by_state = defaultdict(list)
    for row in candidates:
        candidates_by_state[str(row["state_id"])].append(row)
    candidate_metrics = {
        (str(row["state_id"]), int(row["candidate_index"])): row
        for row in report.get("candidate_metrics", [])
    }

    sections = []
    for state in states:
        state_id = str(state["state_id"])
        label = labels.get(state_id, {})
        pairs = label.get("dccp_emitted_pairs", [])
        winner_indices = {int(row["winner_candidate_index"]) for row in pairs}
        loser_indices = {int(row["loser_candidate_index"]) for row in pairs}
        cards = []
        for candidate in sorted(candidates_by_state[state_id], key=lambda row: int(row["candidate_index"])):
            index = int(candidate["candidate_index"])
            badges = []
            if index == 0: badges.append('<span class="badge nominal">nominal</span>')
            if index in label.get("predicted_top_candidates", []): badges.append('<span class="badge top">predicted top</span>')
            if index in winner_indices: badges.append('<span class="badge winner">pair winner</span>')
            if index in loser_indices: badges.append('<span class="badge loser">pair loser</span>')
            action_summary = "not generated"
            action_file = store.run_dir / candidate["action_path"]
            if action_file.exists():
                with np.load(action_file, allow_pickle=False) as data:
                    first = np.asarray(data["actions"])[0]
                action_summary = np.array2string(first, precision=3, suppress_small=True)
            wm_clips = []
            for seed in cfg.world_model.seeds:
                prediction = latest_predictions.get((state_id, index, int(seed)))
                score = latest_scores.get((state_id, index, int(seed)))
                score_text = f" · LRM={float(score['score']):.4f}" if score else ""
                wm_clips.append(_video(prediction.get("video_path") if prediction else None, f"WM seed {seed}{score_text}"))
            sim_clips = []
            for seed in cfg.simulator.evaluation_seeds:
                outcome = latest_outcomes.get((state_id, index, int(seed)))
                suffix = ""
                if outcome:
                    suffix = f" · success={outcome['success']} short={float(outcome['short_progress']):.1f}"
                sim_clips.append(_video(outcome.get("video_path") if outcome else None, f"SIM seed {seed}{suffix}"))
            metric = candidate_metrics.get((state_id, index), {})
            cards.append(f'''
              <article class="candidate">
                <h3>Candidate {index:02d} {''.join(badges)}</h3>
                <code>{html.escape(action_summary)}</code>
                <p>label score={html.escape(str(label.get('label_scores', label.get('candidate_mean_scores', {})).get(str(index), '—')))} · WM-seed mean={html.escape(str(label.get('candidate_mean_scores', {}).get(str(index), '—')))} · simulator success={html.escape(str(metric.get('success_rate', '—')))}</p>
                <details open><summary>World-model prediction → LRM</summary><div class="clips">{''.join(wm_clips)}</div></details>
                <details><summary>Real simulator rollouts</summary><div class="clips">{''.join(sim_clips)}</div></details>
              </article>
            ''')
        pair_display = [{
            "alternative": row["alternative_candidate_index"],
            "winner": row["winner_candidate_index"], "loser": row["loser_candidate_index"],
            "margin": round(float(row["margin_alt_minus_nominal"]), 4),
        } for row in pairs]
        sections.append(f'''
          <section><h2>{html.escape(state_id)}</h2>
            <p>trajectory={html.escape(str(state['trajectory_id']))} · decision={state['decision_index']} · low-level={state['low_level_step']}</p>
            <a href="../{html.escape(state['frame_path'])}"><img class="start" src="../{html.escape(state['frame_path'])}"></a>
            <h3>Frozen DCCP nominal-vs-alternative labels</h3>
            <p>aggregation={html.escape(str(label.get('label_aggregation', '—')))} · primary WM seed={html.escape(str(label.get('primary_wm_seed', '—')))}</p>
            {_table(pair_display, ['alternative','winner','loser','margin'])}
            <div class="candidate-grid">{''.join(cards) or '<p class="missing">Candidates not generated.</p>'}</div>
          </section>
        ''')
    statuses = stage_status(cfg)
    status_html = "".join(
        f'<div class="status {html.escape(str(row["state"]).lower())}"><b>{html.escape(row["stage"])}</b><strong>{html.escape(row["state"])}</strong><small>{html.escape(row["detail"])}</small></div>'
        for row in statuses
    )
    summary = report.get("summary", {})
    summary_rows = [{"metric": key, "value": value} for key, value in summary.items() if not isinstance(value, dict)]
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(cfg.run_name)} branch validity</title><style>
body{{font-family:system-ui,sans-serif;margin:0;background:#eef2f7;color:#0f172a}}main{{max-width:1500px;margin:auto;padding:22px}}section{{background:white;padding:18px;margin:18px 0;border-radius:12px;box-shadow:0 1px 4px #cbd5e1}}
.statuses{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:8px}}.status{{background:white;border:1px solid #cbd5e1;padding:10px;border-radius:8px}}.status strong,.status small{{display:block;margin-top:5px;word-break:break-all}}.done strong,.frozen strong{{color:#16a34a}}.started strong{{color:#d97706}}
.start{{width:256px;max-width:100%;border-radius:8px}}.candidate-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:12px}}.candidate{{border:1px solid #cbd5e1;border-radius:10px;padding:12px;background:#f8fafc}}.candidate code{{display:block;white-space:pre-wrap;font-size:11px}}
.clips{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:6px}}.clip video{{width:100%;display:block;margin-top:3px}}.clip small{{display:block}}.badge{{font-size:10px;color:white;border-radius:10px;padding:2px 6px}}.nominal{{background:#475569}}.top{{background:#7c3aed}}.winner{{background:#16a34a}}.loser{{background:#dc2626}}
.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:6px;border-bottom:1px solid #e2e8f0;text-align:left;white-space:nowrap}}.missing{{color:#64748b}}.plot{{max-width:100%;border:1px solid #cbd5e1}}
</style></head><body><main><h1>{html.escape(cfg.run_name)} · DCCP 分支标签有效性审计</h1>
<p>每个阶段运行后都可重建此离线页面。先检查状态与动作，再检查 WM 视频和 LRM 标签，最后打开模拟器视频。</p>
<div class="statuses">{status_html}</div>
<section><h2>最终摘要</h2>{_table(summary_rows, ['metric','value'])}<h3>预测 margin 与真实成功率差</h3>{_scatter_svg(report.get('pair_metrics', []))}</section>
{''.join(sections) or '<section><p class="missing">States not prepared.</p></section>'}
</main></body></html>'''
    index = store.inspection / "index.html"
    AuditStore.write_text(index, document)
    AuditStore.write_json(store.inspection / "manifest.json", {
        "run_name": cfg.run_name, "stage_status": statuses, "index": str(index),
        "num_states": len(states), "num_candidates": len(candidates),
        "num_wm_predictions": len(latest_predictions), "num_simulator_outcomes": len(latest_outcomes),
    })
    return index
