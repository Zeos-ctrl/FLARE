"""Reference searcher — proves the experiment loop end-to-end without an LLM.

Randomly samples a few model designs from the search space, submits them as
experiments (which run one at a time), and prints the leaderboard by mean_match.
An LLM agent replaces `propose_random` with its own reasoning; everything else
(submit / wait / leaderboard) stays the same. See agent/AGENT.md.

Usage:
    # start the API first, e.g. `flare-api` (defaults to :8420)
    python agent/search_example.py --base http://localhost:8420 --n 3 --group demo
"""
from __future__ import annotations

import argparse
import random

from src.agent.client import FlareClient


def propose_random(space: dict) -> dict:
    """Sample one design from the advertised search space."""
    def pick(field):
        spec = space[field]
        if "choices" in spec:
            return random.choice(spec["choices"])
        lo, hi = spec["range"]
        return random.randint(lo, hi) if spec["type"] == "int" else round(random.uniform(lo, hi), 4)

    return {
        "amp_hidden_size": pick("amp_hidden_size"),
        "amp_layers": pick("amp_layers"),
        "amp_banks": pick("amp_banks"),
        "amp_dropout": pick("amp_dropout"),
        "phase_hidden_size": pick("phase_hidden_size"),
        "phase_layers": pick("phase_layers"),
        "phase_banks": pick("phase_banks"),
        "phase_dropout": pick("phase_dropout"),
        "fourier_bands": pick("fourier_bands"),
    }


def main():
    ap = argparse.ArgumentParser(description="Random-search reference agent for FLARE")
    ap.add_argument("--base", default="http://localhost:8420")
    ap.add_argument("--n", type=int, default=3, help="number of experiments")
    ap.add_argument("--group", default="demo")
    ap.add_argument("--samples", type=int, default=64, help="num_samples override (small = fast)")
    ap.add_argument("--epochs", type=int, default=20, help="num_epochs per run")
    ap.add_argument("--eval-samples", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    client = FlareClient(args.base)
    space = client.search_space()

    for i in range(args.n):
        design = propose_random(space)
        design["num_epochs"] = args.epochs
        spec = {
            "name": f"{args.group}-{i}",
            "group": args.group,
            "source": "searcher",
            "settings_overrides": {"num_samples": args.samples, "device": args.device},
            "model_design": design,
            "evaluate": True,
            "eval_n_samples": args.eval_samples,
        }
        rec = client.submit_experiment(spec)
        print(f"[{i+1}/{args.n}] submitted {rec['name']} ({rec['id']}) — waiting…")
        rec = client.wait(rec["id"])
        print(f"    -> {rec['status']}  mean_match={rec['metrics'].get('mean_match')}")

    print("\nLeaderboard:")
    for rank, r in enumerate(client.leaderboard(group=args.group), 1):
        print(f"  {rank}. {r['name']:<16} match={r['metrics']['mean_match']:.4f}")


if __name__ == "__main__":
    main()
