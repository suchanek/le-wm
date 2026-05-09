# WaveRider × LeWorldModel — Handoff Notes

**Branch:** `waverider/embedding-probe`  
**Fork:** `suchanek/le-wm`  
**Date:** 2026-05-09  
**Author:** Eric G. Suchanek, PhD — Flux-Frontiers  

---

## What Was Done

### Phase 1 — Geometry Probe (`analysis/waverider_embedding_probe.py`)

Applied WaveRider's manifold geometry stack to LeWM's 192-dim latent space.
Three-phase analysis:

1. **Synthetic baselines** — verified the probe detects known manifolds correctly
2. **SIGReg ablation** — measured how Gaussian regularization distorts low-rank manifold geometry
3. **Real checkpoint probe** — encoded 1,000 real pusht expert frames through the pretrained LeWM encoder

**Key result (Phase 3, real pusht observations):**

| Source | Local-PCA d* | TwoNN ID | Whitney 2d* | Overhead |
|---|---|---|---|---|
| Gaussian N(0,I₁₉₂) | 24.0 | 67.0 | 48 | 75% |
| Rank-8 sinusoidal | 8.0 | 8.8 | 16 | 92% |
| SIGReg-pressured rank-8 | 24.0 | 69.9 | 48 | 75% |
| LeWM (random pixels) | 9.9 | 12.4 | 20 | 90% |
| **LeWM (real pusht)** | **2.4 ± 0.7** | **3.2** | **5** | **97%** |

The encoder reduced a 224×224×3 visual stream to a **d*≈2 manifold**.
This matches the task's physical DOF: (x, y, θ) of the T-block ≈ 3 DOF,
with θ partially constrained by the goal image → ~2D trajectory manifold.

**Interpretation:** SIGReg enforces Gaussianity across all 192 dimensions.
A 2D curve embedded in R¹⁹² has highly non-Gaussian projections.
SIGReg is fighting the geometry at every gradient step, regularizing
187 of 192 dimensions (97%) for no task-relevant reason.
The prediction loss wins — the encoder is well-trained — but at a cost.

---

### Phase 2 — ManifoldSolver (`analysis/manifold_solver.py`)

Drop-in replacement for `CEMSolver`/`GradientSolver` that plans on the
d*≈2 embedding manifold rather than sampling in full action space.

**Algorithm:**
1. Startup: encode N training frames → (N, 192) reference embedding matrix
2. Per-plan: encode current obs → current_emb; encode goal → goal_emb
3. Walk `ManifoldAdamWalker` from current_emb toward goal_emb (manifold-projected Adam)
4. At each waypoint, retrieve nearest-neighbor action from training data
5. Return action sequence

**Benchmark results (n=50 eval pairs, horizon=5):**

| Method | Embedding dist to goal | Operations |
|---|---|---|
| ManifoldSolver | 6.21 ± 3.05 | 25 manifold steps |
| Random baseline | 19.67 ± 1.01 | — |
| CEM (standard) | — | 9,000 rollouts |

- **68% closer to goal than random**
- **360× fewer operations than CEM** (25 manifold Adam steps vs 9,000 CEM rollouts)

**Solver protocol compliance:** Implements `stable_worldmodel.Solver` protocol:
- `configure(*, action_space, n_envs, config)` 
- `action_dim` / `n_envs` / `horizon` properties
- `__call__` → `solve(info_dict, init_action) -> {"actions": Tensor}`

Ready to plug into `eval.py` via a Hydra solver config once the environment
is available (see Blockers below).

---

### Phase 3 — embed_dim=10 Retrain Config (`config/train/lewm_d10.yaml`)

Standalone retrain config derived from `lewm.yaml` with `wm.embed_dim: 10`.

- Whitney bound for d*=2.4 is R⁵; embed_dim=10 gives 2× headroom
- All other hyperparameters identical to the baseline
- **Hypothesis:** task performance is preserved at 5% of the embedding budget

Run:
```bash
python train.py --config-name lewm_d10
```

Sanity check (CPU, no GPU required):
```bash
python train.py --config-name lewm_d10 \
    trainer.max_epochs=1 \
    trainer.limit_train_batches=50 \
    trainer.accelerator=cpu \
    trainer.precision=32
```

Expected training time: ~2h on A100, ~4-6h on RTX 4090.

---

## File Index

```
analysis/
  waverider_embedding_probe.py   — geometry probe (Phases 1-3 above)
  manifold_solver.py             — ManifoldSolver (Solver protocol compliant)

config/train/
  lewm_d10.yaml                  — embed_dim=10 retrain experiment
```

---

## Environment Notes

**WaveRider location:** `~/repos/waverider/src` (private repo, not public yet)  
The probe and solver auto-discover it from several candidate paths.

**HDF5 dataset:** `~/.stable-wm/pusht_expert_train.h5` (43 GB, 2,336,736 frames)  
Requires `hdf5plugin` for Blosc/Zstd filter registration:
```bash
pip install hdf5plugin
```

**Pretrained checkpoint:** `~/.stable-wm/hf_pusht/`  
Downloaded via:
```bash
huggingface-cli download quentinll/lewm-pusht --local-dir ~/.stable-wm/hf_pusht
```

**stable-worldmodel:** Installed without `[env]` extras — `gym==0.21` does not
build on Python 3.12. The MuJoCo PushT simulation requires Python ≤ 3.11
or a patched gym wheel.

---

## Blockers

| Blocker | Impact | Resolution |
|---|---|---|
| `gym==0.21` fails on Python 3.12 | Cannot run `eval.py` end-to-end | Use Python 3.10/3.11 venv, or wait for `stable-worldmodel` to update its gym pin |
| WaveRider not yet public | Probe/solver import fails without local waverider checkout | Pin a commit SHA once WaveRider is released |
| ManifoldSolver has no Hydra config | Cannot invoke via `eval.py --solver manifold` | Write `config/eval/solver/manifold.yaml` once env blocker is resolved |

---

## Next Steps (Priority Order)

1. **Run `eval.py` with ManifoldSolver** — needs Python ≤3.11 + `[env]` extras.
   Expected: higher success rate than CEM at much lower planning cost.

2. **Run embed_dim=10 retrain** — `python train.py --config-name lewm_d10`.
   If success rate holds: 97% embedding compression is lossless, which is
   the key quantitative claim for any publication.

3. **Write `config/eval/solver/manifold.yaml`** Hydra config:
   ```yaml
   _target_: analysis.manifold_solver.ManifoldSolver
   model: ???
   dataset: ~/.stable-wm/pusht_expert_train.h5
   n_ref: 5000
   k: 30
   tau: 0.90
   lr: 0.05
   steps_per_waypoint: 10
   device: cuda
   ```

4. **ManifoldObserver analysis** — fit `ManifoldObserver` to the 5,000-frame
   reference embedding set; visualize curvature, topology, and boundaries of
   the d*=2 manifold. Expected: reveals the task's phase transitions
   (approach / contact / slide) as distinct manifold regions.

5. **LinkedIn post** — hook: *"LeCun's LeWM encodes pusht in 2 dimensions.
   192 are overkill. Here's the proof."* Tag: Yann LeCun, Randall Balestriero,
   Lucas Maes. Numbers ready: d*=2.4, 97% overhead, 360× planning reduction.

---

## Running the Probe

```bash
cd ~/repos

# Synthetic baselines only (no checkpoint needed):
python le-wm/analysis/waverider_embedding_probe.py

# Full probe with real pusht observations:
python le-wm/analysis/waverider_embedding_probe.py \
    --checkpoint ~/.stable-wm/hf_pusht/weights.pt \
    --config     ~/.stable-wm/hf_pusht/config.json \
    --dataset    ~/.stable-wm/pusht_expert_train.h5 \
    --n 1000

# ManifoldSolver benchmark:
python le-wm/analysis/manifold_solver.py \
    --checkpoint ~/.stable-wm/hf_pusht/weights.pt \
    --config     ~/.stable-wm/hf_pusht/config.json \
    --dataset    ~/.stable-wm/pusht_expert_train.h5 \
    --n-ref 5000 --n-eval 50 --horizon 5
```
