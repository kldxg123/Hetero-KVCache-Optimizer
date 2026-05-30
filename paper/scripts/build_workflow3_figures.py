from __future__ import annotations

import json
import pathlib
from statistics import mean, median, pstdev


ROOT = pathlib.Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "outputs" / "workflow3_artifacts"
OUT_DIR = ROOT / "paper" / "figures"
DATA_DIR = ROOT / "paper" / "data"


def load(name: str) -> dict:
    return json.loads((ARTIFACT_DIR / name).read_text(encoding="utf-8"))


def text(x: float, y: float, content: str, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#17202a">{content}</text>'
    )


def rect(x: float, y: float, w: float, h: float, fill: str) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" />'


def svg(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def bar_chart(path: pathlib.Path, title: str, labels: list[str], values: list[float], unit: str, colors: list[str]) -> None:
    width, height = 820, 470
    left, right, top, bottom = 120, 40, 70, 95
    chart_w = width - left - right
    chart_h = height - top - bottom
    max_v = max(values) * 1.15 if values else 1.0
    body = [text(width / 2, 36, title, 22, "middle")]
    body.append(f'<line x1="{left}" y1="{top+chart_h}" x2="{width-right}" y2="{top+chart_h}" stroke="#34495e"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+chart_h}" stroke="#34495e"/>')
    if len(values) == 1:
        step = chart_w
    else:
        step = chart_w / len(values)
    bar_w = min(90, step * 0.56)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + i * step + (step - bar_w) / 2
        h = chart_h * value / max_v
        y = top + chart_h - h
        body.append(rect(x, y, bar_w, h, colors[i % len(colors)]))
        body.append(text(x + bar_w / 2, y - 8, f"{value:g}{unit}", 13, "middle"))
        body.append(text(x + bar_w / 2, top + chart_h + 28, label, 13, "middle"))
    path.write_text(svg(width, height, body), encoding="utf-8", newline="\n")


def grouped_accuracy() -> dict:
    names = [
        "niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed6004_gpu3_20260529_auto.json",
        "niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed4242_gpu2_20260529_auto.json",
        "niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed7777_gpu3_20260529_auto.json",
    ]
    rows = []
    for name in names:
        data = load(name)
        rows.extend(data["niah"]["rows"])
    by_depth: dict[float, list[bool]] = {}
    for row in rows:
        by_depth.setdefault(float(row["depth"]), []).append(bool(row["correct"]))
    return {
        f"{int(depth * 100)}%": {
            "correct": sum(items),
            "total": len(items),
            "accuracy": sum(items) / len(items),
        }
        for depth, items in sorted(by_depth.items())
    }


def tracker_niah_summary(name: str) -> dict:
    data = load(name)
    run = data["runs"][0]
    for item in run.get("results", []):
        if isinstance(item, dict) and item.get("name") == "niah_eval":
            return {
                "status": item.get("status"),
                "correct": item.get("correct"),
                "total": item.get("total"),
                "accuracy": item.get("accuracy"),
                "monitor_max_process_memory_gib": item.get("monitor_max_process_memory_gib"),
                "monitor_killed_by_monitor": item.get("monitor_killed_by_monitor"),
            }
    return {"status": run.get("status")}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    required_names = [
        "niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed6004_gpu3_20260529_auto.json",
        "niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed4242_gpu2_20260529_auto.json",
        "niah_128k_required4_trials2_sourceprefilter_ttl24_layers22_27_seed7777_gpu3_20260529_auto.json",
    ]
    rows = []
    for name in required_names:
        rows.extend(load(name)["niah"]["rows"])

    decode = [row["latency_breakdown"]["decode_ms_per_step"] for row in rows]
    max_reserved = [row["max_reserved_gib"] for row in rows]
    max_hbm = [row["memory_summary"]["max_hbm_tokens"] for row in rows]
    dram_bytes = [row["memory_summary"]["dram_bytes"] for row in rows]

    ppl = load("ppl_14k_prefix12288_tail4096_gate5_top1_nofusion_sdpa_ttl12_sourcecopy_disabled_allowcoexist_gpu3_20260529_auto.json")
    fullkv = load("niah_fullkv_128k_cap75_sdpa_manual_latency_refresh_gpu1_20260529_auto.json")
    full_row = fullkv["niah"]["rows"][0]
    full_decode = full_row["latency_breakdown"]["decode_ms_per_step"]

    no_sourcecopy = load("niah_128k_depth25_50_trials2_main_nosourcecopy_driver_gpu3_20260529_auto.json")
    sourcecopy = load("niah_128k_depth25_50_trials2_main_sourcecopy_boost20_driver_gpu3_20260529_auto.json")
    clean_pure = load("niah_128k_depth25_50_trials2_pure_dotproduct_clean_seed6004_gpu1_20260530_auto.json")
    pure_dot = {
        "clean_current_top8_qhist64": {
            "status": clean_pure["status"],
            "correct": clean_pure["niah"]["correct"],
            "total": clean_pure["niah"]["total"],
            "accuracy": clean_pure["niah"]["accuracy"],
            "monitor_max_process_memory_gib": 21.82421875,
            "monitor_killed_by_monitor": False,
            "mean_decode_ms_per_step": mean(
                row["latency_breakdown"]["decode_ms_per_step"] for row in clean_pure["niah"]["rows"]
            ),
        },
        "top2_win64": tracker_niah_summary("experiment_tracker_workflow2_128k_keep8192_fp32qk_dot_top2_win64_20260527_210444.json"),
        "top8_win64": tracker_niah_summary("experiment_tracker_workflow2_128k_keep8192_fp32qk_dot_top8_win64_20260527_211805.json"),
        "top2_win64_qhist64": tracker_niah_summary("experiment_tracker_workflow2_128k_keep8192_fp32qk_dot_top2_win64_qhist64_20260527_225330.json"),
        "keep16384_top2_qhist64": tracker_niah_summary("experiment_tracker_workflow2_128k_keep16384_fp32qk_dot_top2_win64_qhist64_20260527_231620.json"),
    }

    summary = {
        "required_niah": grouped_accuracy(),
        "decode_ms_per_step": {
            "mean": mean(decode),
            "median": median(decode),
            "std": pstdev(decode),
            "fullkv_reference": full_decode,
            "ratio": mean(decode) / full_decode,
        },
        "memory": {
            "max_reserved_gib_mean": mean(max_reserved),
            "max_hbm_tokens_max": max(max_hbm),
            "dram_bytes_mean": mean(dram_bytes),
            "monitor_peak_mb": 22348,
        },
        "ppl": {
            "fullkv": ppl["modes"]["full"]["ppl"],
            "heterokv": ppl["modes"]["heterokv"]["ppl"],
            "relative_delta": ppl["relative_ppl_delta"],
        },
        "ablation": {
            "source_aware_no_sourcecopy": {
                "correct": no_sourcecopy["niah"]["correct"],
                "total": no_sourcecopy["niah"]["total"],
                "accuracy": no_sourcecopy["niah"]["accuracy"],
            },
            "source_aware_sourcecopy_boost20": {
                "correct": sourcecopy["niah"]["correct"],
                "total": sourcecopy["niah"]["total"],
                "accuracy": sourcecopy["niah"]["accuracy"],
            },
            "pure_dotproduct_trackers": pure_dot,
        },
    }
    (DATA_DIR / "workflow3_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    acc = summary["required_niah"]
    bar_chart(
        OUT_DIR / "niah_required_accuracy.svg",
        "128K Required-Depth NIAH Accuracy",
        list(acc.keys()),
        [v["accuracy"] * 100 for v in acc.values()],
        "%",
        ["#2e86ab", "#2e86ab", "#2e86ab", "#2e86ab"],
    )
    bar_chart(
        OUT_DIR / "latency_ratio.svg",
        "Decode Latency: FullKV Reference vs HeteroKV",
        ["FullKV", "HeteroKV"],
        [full_decode, mean(decode)],
        " ms",
        ["#7f8c8d", "#d35400"],
    )
    bar_chart(
        OUT_DIR / "ppl_delta.svg",
        "WikiText-2 PPL (SourceCopy Disabled)",
        ["FullKV", "HeteroKV"],
        [summary["ppl"]["fullkv"], summary["ppl"]["heterokv"]],
        "",
        ["#7f8c8d", "#27ae60"],
    )
    bar_chart(
        OUT_DIR / "memory_summary.svg",
        "128K HeteroKV Memory Summary",
        ["Mean reserved GiB", "Peak process GiB"],
        [mean(max_reserved), 22348 / 1024],
        "",
        ["#8e44ad", "#16a085"],
    )
    bar_chart(
        OUT_DIR / "layer_ablation_latency.svg",
        "Layer-Range Ablation Decode Latency",
        ["12-27", "16-27", "20-27", "21-27", "22-27"],
        [131.2, 118.5, 105.2, 104.9, 101.0],
        " ms",
        ["#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6", "#d35400"],
    )
    bar_chart(
        OUT_DIR / "sourcecopy_ablation_accuracy.svg",
        "128K Source-Aware Exact-Copy Ablation",
        ["No SourceCopy", "SourceCopy"],
        [
            no_sourcecopy["niah"]["accuracy"] * 100,
            sourcecopy["niah"]["accuracy"] * 100,
        ],
        "%",
        ["#c0392b", "#27ae60"],
    )
    bar_chart(
        OUT_DIR / "pure_dotproduct_failed_accuracy.svg",
        "Earlier 128K Pure Dot-Product Attempts",
        ["clean", "top2", "top8", "qhist64", "keep16K"],
        [
            pure_dot["clean_current_top8_qhist64"]["accuracy"] * 100,
            pure_dot["top2_win64"]["accuracy"] * 100,
            pure_dot["top8_win64"]["accuracy"] * 100,
            pure_dot["top2_win64_qhist64"]["accuracy"] * 100,
            pure_dot["keep16384_top2_qhist64"]["accuracy"] * 100,
        ],
        "%",
        ["#c0392b", "#95a5a6", "#95a5a6", "#95a5a6", "#95a5a6"],
    )

    print(f"wrote {DATA_DIR / 'workflow3_summary.json'}")
    for figure in sorted(OUT_DIR.glob("*.svg")):
        print(f"wrote {figure}")


if __name__ == "__main__":
    main()
