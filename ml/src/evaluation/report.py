"""Result-formatting helpers shared by scripts/run_baselines.py (Phase 2)
and scripts/finetune_sbert.py (Phase 3), so baseline and fine-tuned results
render as one directly comparable table instead of two independent ones.
"""
from typing import Dict, List, Sequence


def format_metrics_table(results: List[Dict], ks: Sequence[int]) -> str:
    header = ["matcher", "n_queries", "n_candidates"] + [f"P@{k}" for k in ks] + ["MRR"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        o = r["overall"]
        row = [r["matcher"], str(o["n_queries"]), str(o["n_candidates"])]
        row += [f"{o[f'precision_at_{k}']:.4f}" for k in ks]
        row.append(f"{o['reciprocal_rank']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def format_category_table(results: List[Dict], ks: Sequence[int]) -> str:
    header = ["matcher", "category", "n_queries"] + [f"P@{k}" for k in ks] + ["MRR"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        for category, m in sorted(r["by_category"].items(), key=lambda kv: -kv[1]["n_queries"]):
            row = [r["matcher"], category, str(m["n_queries"])]
            row += [f"{m[f'precision_at_{k}']:.4f}" for k in ks]
            row.append(f"{m['reciprocal_rank']:.4f}")
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)
