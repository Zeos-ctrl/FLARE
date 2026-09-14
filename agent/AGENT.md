# FLARE experiment agent

You are running an automated search over neural-network **surrogate models** for
gravitational waveforms. Your goal is to find a model design that reconstructs
waveforms as accurately as possible.

## Objective

**Maximize evaluation `mean_match`** (0..1): the overlap between the surrogate's
predicted waveforms and ground-truth waveforms, averaged over a held-out test set.
Higher is better. It is recorded automatically after each run's training, when the
experiment's `evaluate` flag is true.

Secondary signal: `best_val_loss_amp` / `best_val_loss_phase` (lower is better),
available live during training before the eval metric exists.

## The loop

Start the dashboard API first (`flare-api` / `uvicorn src.api.main:app`), then:

```python
from src.agent.client import FlareClient
client = FlareClient("http://localhost:8420")

GROUP = "sweep-1"                       # tag this campaign so runs group together
for _ in range(BUDGET):
    board = client.leaderboard(group=GROUP)          # best designs so far
    space = client.search_space()                    # fields + ranges you may vary
    spec = propose_next(board, space)                # <-- your reasoning goes here
    rec = client.submit_experiment(spec)
    rec = client.wait(rec["id"])                      # runs one at a time (queued)
    print(rec["name"], rec["metrics"].get("mean_match"))
```

`propose_next` is where an LLM (or a sampler) reads what has worked and proposes
the next design. Runs are **serialized** (one GPU at a time); submit many and they
queue — you do not need to throttle.

## What you may vary (the spec)

An `ExperimentSpec` (see `src/api/schemas.py`):

- `model_design`: an inline design dict — the main knob. Vary the fields in
  `client.search_space()`: hidden sizes, layers, banks, dropout, learning rates,
  Fourier bands/frequency, and `num_epochs` (your training budget per run).
  Alternatively start from a saved design with `model_name`.
- `settings_overrides`: patch the global data/system settings, e.g.
  `{"num_samples": 2000}` for more training data, `{"waveform": "SEOBNRv4"}`,
  or `{"device": "cpu"}`. Omit to use the saved dashboard settings.
- `evaluate` (default true) and `eval_n_samples` (default 500): the eval pass that
  produces `mean_match`. Keep it on so runs are comparable.
- `group`, `tags`, `name`, `notes`: bookkeeping. Always set `group` for a sweep.

Example spec:

```python
spec = {
    "name": "wider-phase",
    "group": GROUP,
    "settings_overrides": {"num_samples": 2000},
    "model_design": {
        "amp_hidden_size": 256, "amp_layers": 3, "amp_banks": 3,
        "phase_hidden_size": 512, "phase_layers": 4, "phase_banks": 6,
        "fourier_bands": 16, "num_epochs": 200,
    },
    "evaluate": True, "eval_n_samples": 500,
}
```

## Strategy guidance

- Begin with a couple of cheap probes (small `num_samples`, low `num_epochs`) to
  confirm the pipeline, then scale up the budget for promising designs.
- Change one region at a time (amp net, phase net, Fourier features) so you can
  attribute improvements. Phase is usually the harder net.
- Stop when `mean_match` plateaus across several runs, or after `BUDGET` runs.

## Watching progress

Everything you submit appears live in the dashboard's **Experiments** tab
(leaderboard + status). You can also `client.list_experiments(group=GROUP)` or
`client.get_experiment(id)` at any time.
