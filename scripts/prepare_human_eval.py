"""Build a blind human-eval sheet for 30 posts (both members fill choices)."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summaries", default="results/eval_summaries.jsonl")
    p.add_argument("--out_csv", default="results/human_eval_blind.csv")
    p.add_argument("--out_key", default="results/human_eval_key.json")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rows = []
    with open(args.summaries, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    rng = random.Random(args.seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    idxs = idxs[: args.n]

    key = {}
    out_rows = []
    for j, i in enumerate(idxs):
        methods = ["SFT", "PPO", "DPO"]
        rng.shuffle(methods)
        key[j] = {"eval_index": i, "order": methods}
        out_rows.append(
            {
                "post_id": j,
                "prompt_tail": rows[i]["prompt"][-500:],
                "A": rows[i][methods[0]],
                "B": rows[i][methods[1]],
                "C": rows[i][methods[2]],
                "member1_best": "",  # fill A/B/C
                "member2_best": "",
            }
        )

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    with open(args.out_key, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)

    print(f"Wrote blind sheet → {args.out_csv}")
    print(f"Wrote answer key  → {args.out_key}  (do NOT show to raters)")


if __name__ == "__main__":
    main()
