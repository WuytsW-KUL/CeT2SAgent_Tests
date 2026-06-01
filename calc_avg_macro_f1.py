from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent

EXPECTED_RUNS = {
    "Ablation": 3,
    "Baselines_English_Only": 3,
    "Translation": 2,
}


def load_macro_f1(path: Path) -> float | None:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        value = data.get("macro_f1_so_far")
        if value is None:
            print(f"  [WARN] 'macro_f1_so_far' missing in {path.relative_to(ROOT)}")
            return None
        return float(value)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"  [WARN] Could not read {path.relative_to(ROOT)}: {exc}")
        return None


def classify(rel_parts: tuple[str, ...]) -> tuple[str, str] | None:
    if len(rel_parts) < 2:
        return None
    category = rel_parts[0]
    if category == "Ablation":
        if len(rel_parts) == 4 and rel_parts[-1] == "summary.json":
            return ("Ablation", rel_parts[1])
    elif category == "Baselines_English_Only":
        if len(rel_parts) == 4 and rel_parts[-1] == "summary.json":
            return ("Baselines_English_Only", rel_parts[1])
    elif category == "Translation":
        if len(rel_parts) == 5 and rel_parts[-1] == "summary.json":
            return ("Translation", f"{rel_parts[1]} / {rel_parts[3]}")
    return None


results: dict[str, dict[str, list[float]]] = {
    "Ablation": defaultdict(list),
    "Baselines_English_Only": defaultdict(list),
    "Translation": defaultdict(list),
}

for summary_path in ROOT.rglob("summary.json"):
    rel_parts = summary_path.relative_to(ROOT).parts
    classification = classify(rel_parts)
    if classification is None:
        continue
    category, group_key = classification
    value = load_macro_f1(summary_path)
    if value is not None:
        results[category][group_key].append(value)


SEP_MAJOR = "=" * 72
SEP_MINOR = "-" * 72


def print_category(category: str, groups: dict[str, list[float]]) -> None:
    print(SEP_MAJOR)
    print(f"  {category}")
    print(SEP_MAJOR)
    if not groups:
        print("  (no data found)\n")
        return
    expected = EXPECTED_RUNS[category]
    all_avgs: list[float] = []
    for group_key in sorted(groups):
        vals = groups[group_key]
        n = len(vals)
        avg = sum(vals) / n if n else float("nan")
        all_avgs.append(avg)
        flag = f"  [INCOMPLETE: got {n}/{expected} runs]" if n != expected else ""
        runs_str = "  ".join(f"{v:.4f}" for v in vals)
        print(f"  {group_key}")
        print(f"    runs ({n}): {runs_str}")
        print(f"    average  : {avg:.4f}{flag}")
        print()
    if all_avgs:
        grand_avg = sum(all_avgs) / len(all_avgs)
        print(SEP_MINOR)
        print(f"  Category average across {len(all_avgs)} groups: {grand_avg:.4f}")
    print()


print()
print_category("Ablation", results["Ablation"])
print_category("Baselines_English_Only", results["Baselines_English_Only"])
print_category("Translation", results["Translation"])
