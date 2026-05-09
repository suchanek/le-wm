"""
WaveRider Embedding Probe — LeWorldModel
========================================

Analyses the geometry of LeWorldModel's 192-dimensional latent space using
WaveRider's ManifoldModel.  Answers three questions:

  1. What is the intrinsic dimensionality d* of the embedding space?
  2. Is the Gaussian assumption of SIGReg compatible with the data manifold,
     or does it hide low-dimensional structure?
  3. Is embed_dim=192 justified, or would 2×d* (Whitney bound) suffice?

Usage
-----
    # Synthetic baselines — no checkpoint required:
    python analysis/waverider_embedding_probe.py

    # With a pretrained LeWM checkpoint from HuggingFace:
    python analysis/waverider_embedding_probe.py \\
        --checkpoint $STABLEWM_HOME/hf_pusht/weights.pt \\
        --config    $STABLEWM_HOME/hf_pusht/config.json

    # Download checkpoint first:
    #   pip install huggingface_hub
    #   huggingface-cli download quentinll/lewm-pusht --local-dir $STABLEWM_HOME/hf_pusht

Authors
-------
    Eric G. Suchanek, PhD  —  Flux-Frontiers
    WaveRider integration with LeWorldModel (LeCun et al., 2026)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Locate WaveRider — try several candidate paths
# ---------------------------------------------------------------------------
_WR_CANDIDATES = [
    Path(__file__).parent.parent.parent / "waverider" / "src",
    Path(__file__).parent.parent / "waverider" / "src",
    Path.home() / "repos" / "waverider" / "src",
]
for _p in _WR_CANDIDATES:
    if _p.exists():
        sys.path.insert(0, str(_p))
        break

try:
    from waverider.manifold_model import ManifoldModel
    _WAVERIDER_OK = True
except ImportError:
    _WAVERIDER_OK = False

EMBED_DIM = 192   # LeWM default (ViT-tiny hidden size → projector → 192-dim)


# ---------------------------------------------------------------------------
# Synthetic embedding generators
# ---------------------------------------------------------------------------

def gaussian_embeddings(n: int, dim: int = EMBED_DIM, seed: int = 42) -> np.ndarray:
    """Isotropic Gaussian — the distribution SIGReg is designed to enforce."""
    return np.random.default_rng(seed).standard_normal((n, dim))


def low_rank_embeddings(n: int, rank: int, dim: int = EMBED_DIM,
                        noise: float = 0.05, seed: int = 42) -> np.ndarray:
    """Rank-r linear manifold embedded in dim-dimensional ambient space."""
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((rank, dim))
    basis /= np.linalg.norm(basis, axis=1, keepdims=True)
    coords = rng.standard_normal((n, rank))
    X = coords @ basis
    X += noise * rng.standard_normal((n, dim))
    return X


def nonlinear_manifold_embeddings(n: int, rank: int = 8, dim: int = EMBED_DIM,
                                  seed: int = 42) -> np.ndarray:
    """Nonlinear (sinusoidal) manifold: rank-r latent → sinusoidal lifting."""
    rng = np.random.default_rng(seed)
    t = rng.standard_normal((n, rank))
    half = dim // 2
    basis_lin = rng.standard_normal((rank, half))
    basis_sin = rng.standard_normal((rank, dim - half))
    X = np.hstack([t @ basis_lin, np.sin(t @ basis_sin)])
    X += 0.05 * rng.standard_normal((n, dim))
    return X


def sigreg_pressured(X: np.ndarray, steps: int = 100, weight: float = 0.09,
                     seed: int = 42) -> np.ndarray:
    """Simulate SIGReg gradient pressure: blend X iteratively toward Gaussian.

    Approximates the effect of minimising the SIGReg term on the embeddings
    directly (without the prediction loss), to show how it distorts geometry.
    """
    rng = np.random.default_rng(seed)
    X = X.copy().astype(np.float64)
    target_std = float(np.std(X))
    for _ in range(steps):
        G = rng.standard_normal(X.shape) * target_std
        X = X - weight * (X - G)
    return X


# ---------------------------------------------------------------------------
# Fast standalone dimensionality estimators (no ManifoldModel overhead)
# ---------------------------------------------------------------------------

def fast_local_pca_dim(X: np.ndarray, k: int = 30, tau: float = 0.90,
                       n_samples: int = 200, seed: int = 42) -> dict:
    """Local-PCA intrinsic dimensionality over random subsampled neighbourhoods."""
    rng = np.random.default_rng(seed)
    n = len(X)
    probe_idx = rng.choice(n, size=min(n_samples, n), replace=False)

    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), algorithm="auto").fit(X)
    _, indices = nbrs.kneighbors(X[probe_idx])

    dims: list[int] = []
    for row_idx, nbr_idx in zip(probe_idx, indices):
        hood = X[nbr_idx[1:]]   # exclude self
        centered = hood - hood.mean(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, sv, _ = np.linalg.svd(centered, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
        ev = sv ** 2
        total = ev.sum()
        if total < 1e-12:
            continue
        cumvar = np.cumsum(ev) / total
        d = int(np.searchsorted(cumvar, tau) + 1)
        dims.append(min(d, len(ev)))

    arr = np.array(dims, dtype=float)
    return {
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "median": float(np.median(arr)),
        "n_probes": len(arr),
    }


def twonn_id(X: np.ndarray) -> float:
    """Two-Nearest-Neighbour (Facco et al., 2017) global intrinsic dim estimate."""
    nbrs = NearestNeighbors(n_neighbors=3).fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:, 1]
    r2 = dists[:, 2]
    mask = r1 > 1e-12
    mu = r2[mask] / r1[mask]
    return float(1.0 / np.mean(np.log(mu)))


# ---------------------------------------------------------------------------
# Probe runner
# ---------------------------------------------------------------------------

def probe(X: np.ndarray, label: str, k_pca: int = 30, k_graph: int = 15,
          tau: float = 0.90) -> None:
    n, dim = X.shape
    bar = "─" * 64

    print(f"\n{bar}")
    print(f"  {label}")
    print(f"  n={n}  ambient_dim={dim}")
    print(bar)

    # Fast local-PCA estimate
    fast = fast_local_pca_dim(X, k=k_pca, tau=tau)
    twonn = twonn_id(X)
    mean_d = fast["mean"]
    whitney = int(np.ceil(2 * mean_d))

    print(f"\n  Local-PCA (τ={tau}, k={k_pca}, n_probes={fast['n_probes']})")
    print(f"    mean d*  = {mean_d:.1f}  ±  {fast['std']:.1f}")
    print(f"    range    = [{fast['min']}, {fast['max']}]  "
          f"median {fast['median']:.1f}")
    print(f"\n  TwoNN global ID   = {twonn:.1f}")
    print(f"\n  Whitney bound (2×d*) = {whitney}")

    if whitney < dim:
        overhead_pct = (dim - whitney) / dim * 100
        ratio = dim / mean_d
        print(f"  >> embed_dim={dim} is {ratio:.1f}× the mean intrinsic dim.")
        print(f"     SIGReg regularises {dim - whitney} potentially redundant "
              f"dims ({overhead_pct:.0f}% overhead).")
    else:
        print(f"  >> embed_dim={dim} is fully justified by the data geometry.")

    # Full ManifoldModel (richer per-node geometry)
    if _WAVERIDER_OK:
        print(f"\n  ManifoldModel (k_graph={k_graph}, k_pca={k_pca}, τ={tau})")
        mm = ManifoldModel(k_graph=k_graph, k_pca=k_pca, variance_threshold=tau)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mm.fit(X)
        gs = mm.geometry_summary()
        print(f"    global mean d*     = {mm.intrinsic_dim:.2f}")
        print(f"    local mean d*      = {gs['mean_intrinsic_dim']:.1f}  "
              f"±  {gs['std_intrinsic_dim']:.1f}")
        print(f"    [min, max] local d* = [{gs['min_intrinsic_dim']}, "
              f"{gs['max_intrinsic_dim']}]")
        print(f"    n_nodes / n_edges   = {gs['n_nodes']} / {gs['n_edges']}")
    else:
        print("\n  [ManifoldModel skipped — waverider not on PYTHONPATH]")

    print()


# ---------------------------------------------------------------------------
# HDF5 dataset loader
# ---------------------------------------------------------------------------

# ImageNet normalisation constants (matches LeWM training pipeline)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_decompressed(h5_path: Path) -> Path:
    """If only the .zst exists, decompress it with the system zstd binary."""
    zst_path = h5_path.with_suffix(h5_path.suffix + ".zst")
    if h5_path.exists():
        return h5_path
    if not zst_path.exists():
        raise FileNotFoundError(f"Neither {h5_path} nor {zst_path} found.")
    print(f"  Decompressing {zst_path.name} → {h5_path.name} ...")
    import subprocess
    subprocess.run(["zstd", "-d", str(zst_path), "-o", str(h5_path)], check=True)
    return h5_path


def load_hdf5_pixels(h5_path: Path, n: int, img_size: int = 224,
                     seed: int = 42) -> "torch.Tensor":
    """Sample n frames from the pusht HDF5 dataset, normalised for LeWM.

    :param h5_path: Path to pusht_expert_train.h5 (decompressed).
    :param n:       Number of frames to sample.
    :param img_size: Target spatial size (default 224).
    :returns: Float32 tensor (n, 3, img_size, img_size), ImageNet-normalised.
    """
    import torch
    import h5py
    from torchvision.transforms.functional import resize, to_tensor

    h5_path = _ensure_decompressed(h5_path)

    with h5py.File(h5_path, "r") as f:
        # Discover the pixels dataset — handle both 'pixels' and 'observations'
        pix_key = next((k for k in ("pixels", "observations", "obs") if k in f), None)
        if pix_key is None:
            print(f"  Available keys: {list(f.keys())}")
            raise KeyError("Cannot find pixel observations in HDF5 (expected 'pixels')")

        pix = f[pix_key]
        total = pix.shape[0]
        print(f"  HDF5 '{pix_key}': shape={pix.shape}  dtype={pix.dtype}")

        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(total, size=min(n, total), replace=False))
        raw = pix[idx]   # (n, ...) — could be (n, H, W, C) or (n, T, H, W, C)

    # Flatten time dimension if present
    if raw.ndim == 5:
        raw = raw[:, 0]          # take first frame of each clip: (n, H, W, C)

    # Convert uint8 HWC → float CHW, resize, normalise
    frames: list[torch.Tensor] = []
    for img in raw:
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        t = to_tensor(img)                          # (3, H, W) float in [0,1]
        if t.shape[-1] != img_size or t.shape[-2] != img_size:
            t = resize(t, [img_size, img_size], antialias=True)
        mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
        t = (t - mean) / std
        frames.append(t)

    return torch.stack(frames)   # (n, 3, img_size, img_size)


# ---------------------------------------------------------------------------
# LeWM checkpoint loader
# ---------------------------------------------------------------------------

def _build_lewm_model(config_path: str):
    """Reconstruct JEPA from config.json; return (model, img_size)."""
    import torch
    repo_root = str(Path(__file__).parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    cfg = json.loads(Path(config_path).read_text())

    def _clean(d: dict) -> dict:
        return {k: v for k, v in d.items() if not k.startswith("_")}

    enc_cfg = _clean(cfg["encoder"])
    img_size = enc_cfg.get("image_size", 224)

    try:
        import stable_pretraining as spt
        encoder = spt.backbone.utils.vit_hf(
            enc_cfg["size"],
            patch_size=enc_cfg["patch_size"],
            image_size=img_size,
            pretrained=False,
            use_mask_token=False,
        )
    except ImportError:
        from transformers import ViTConfig, ViTModel
        vcfg = ViTConfig(
            hidden_size=192,
            num_hidden_layers=12,
            num_attention_heads=3,
            intermediate_size=768,
            image_size=img_size,
            patch_size=enc_cfg.get("patch_size", 14),
            num_channels=3,
        )
        encoder = ViTModel(vcfg, add_pooling_layer=False)

    def _mlp(key: str) -> MLP:
        c = _clean(cfg[key])
        return MLP(input_dim=c["input_dim"], output_dim=c["output_dim"],
                   hidden_dim=c["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(**_clean(cfg["predictor"])),
        action_encoder=Embedder(**_clean(cfg["action_encoder"])),
        projector=_mlp("projector"),
        pred_proj=_mlp("pred_proj"),
    )
    return model, img_size, cfg


def load_lewm_embeddings(weights_path: str, config_path: str,
                         n: int = 500, batch: int = 32,
                         dataset_path: str | None = None) -> np.ndarray:
    """Extract 192-dim embeddings from a pretrained LeWM checkpoint.

    :param weights_path:  Path to weights.pt.
    :param config_path:   Path to config.json.
    :param n:             Number of embeddings to extract.
    :param batch:         Forward-pass batch size.
    :param dataset_path:  Path to pusht_expert_train.h5[.zst] for real frames.
                          If None, falls back to random Gaussian pixels.
    :returns: np.ndarray of shape (n, 192).
    """
    import torch

    model, img_size, _ = _build_lewm_model(config_path)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.eval()

    if dataset_path is not None:
        h5 = Path(dataset_path)
        # Accept both the .h5 and the .h5.zst path
        if str(h5).endswith(".zst"):
            h5 = Path(str(h5)[:-4])
        print(f"  Loading real pusht observations from {h5.name} ...")
        pixels = load_hdf5_pixels(h5, n=n, img_size=img_size)
        source = "real pusht frames"
    else:
        print(f"  No dataset supplied — using random Gaussian pixels.")
        pixels = torch.randn(n, 3, img_size, img_size)
        source = "random Gaussian pixels"

    print(f"  Extracting {len(pixels)} embeddings (source: {source}, batch={batch})...")
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(pixels), batch):
            chunk = pixels[i : i + batch]
            info = {"pixels": chunk.unsqueeze(1)}   # (B, 1, C, H, W)
            info = model.encode(info)
            embeddings.append(info["emb"][:, 0].numpy())

    return np.concatenate(embeddings, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=None,
                        help="Path to LeWM weights.pt (HuggingFace download)")
    parser.add_argument("--config", default=None,
                        help="Path to LeWM config.json (same download)")
    parser.add_argument("--n", type=int, default=500,
                        help="Number of embeddings to probe (default 500)")
    parser.add_argument("--k-pca", type=int, default=30,
                        help="Local-PCA neighbourhood size (default 30)")
    parser.add_argument("--k-graph", type=int, default=15,
                        help="ManifoldModel graph degree (default 15)")
    parser.add_argument("--tau", type=float, default=0.90,
                        help="Variance threshold for intrinsic dim (default 0.90)")
    parser.add_argument("--rank", type=int, default=8,
                        help="Synthetic manifold rank hypothesis (default 8)")
    parser.add_argument("--dataset", default=None,
                        help="Path to pusht_expert_train.h5[.zst] for real observations")
    args = parser.parse_args()

    print("=" * 64)
    print("  WaveRider Embedding Probe — LeWorldModel")
    print("=" * 64)
    print(f"  WaveRider available : {_WAVERIDER_OK}")
    print(f"  Ambient embed_dim   : {EMBED_DIM}  (LeWM default)")
    print(f"  Variance threshold τ: {args.tau}")
    print(f"  Synthetic rank hyp. : {args.rank}")

    # ---- Phase 1: Synthetic baselines ----------------------------------------
    print("\n\n" + "=" * 64)
    print("  Phase 1  —  Synthetic baselines")
    print("=" * 64)

    probe(
        gaussian_embeddings(args.n),
        "Gaussian N(0, I₁₉₂)  — what SIGReg enforces",
        args.k_pca, args.k_graph, args.tau,
    )
    probe(
        low_rank_embeddings(args.n, rank=args.rank),
        f"Rank-{args.rank} linear manifold in {EMBED_DIM}D  — control",
        args.k_pca, args.k_graph, args.tau,
    )
    probe(
        nonlinear_manifold_embeddings(args.n, rank=args.rank),
        f"Rank-{args.rank} nonlinear (sinusoidal) manifold in {EMBED_DIM}D",
        args.k_pca, args.k_graph, args.tau,
    )

    # ---- Phase 2: SIGReg ablation --------------------------------------------
    print("\n" + "=" * 64)
    print("  Phase 2  —  SIGReg pressure on a low-rank manifold")
    print("=" * 64)

    X_low = low_rank_embeddings(args.n, rank=args.rank)
    X_pressed = sigreg_pressured(X_low, steps=100, weight=0.09)

    probe(X_low,     f"Rank-{args.rank} manifold — before SIGReg",
          args.k_pca, args.k_graph, args.tau)
    probe(X_pressed, f"Rank-{args.rank} manifold — after SIGReg pressure (simulated)",
          args.k_pca, args.k_graph, args.tau)

    # ---- Phase 3: Pretrained checkpoint --------------------------------------
    print("\n" + "=" * 64)
    print("  Phase 3  —  Pretrained LeWM checkpoint")
    print("=" * 64)

    if args.checkpoint and args.config:
        try:
            # Random-pixel baseline (already shown above, kept for direct comparison)
            X_rand = load_lewm_embeddings(
                args.checkpoint, args.config, n=args.n,
            )
            probe(X_rand,
                  "LeWM encoder  —  random Gaussian pixels (baseline)",
                  args.k_pca, args.k_graph, args.tau)

            # Real observations (if dataset available)
            if args.dataset:
                X_real = load_lewm_embeddings(
                    args.checkpoint, args.config, n=args.n,
                    dataset_path=args.dataset,
                )
                probe(X_real,
                      "LeWM encoder  —  real pusht expert observations",
                      args.k_pca, args.k_graph, args.tau)
            else:
                h5_default = Path.home() / ".stable-wm" / "pusht_expert_train.h5"
                zst_default = Path(str(h5_default) + ".zst")
                if h5_default.exists() or zst_default.exists():
                    X_real = load_lewm_embeddings(
                        args.checkpoint, args.config, n=args.n,
                        dataset_path=str(h5_default),
                    )
                    probe(X_real,
                          "LeWM encoder  —  real pusht expert observations",
                          args.k_pca, args.k_graph, args.tau)
                else:
                    print("\n  Real-observations probe skipped.")
                    print("  Dataset still downloading, or pass --dataset <path>.")
                    print("  Expected: ~/.stable-wm/pusht_expert_train.h5[.zst]")
        except Exception as exc:
            import traceback
            print(f"\n  [error]: {exc}")
            traceback.print_exc()
    else:
        print("\n  Skipped — pass --checkpoint and --config to probe real embeddings.")
        print("\n  Download a pretrained checkpoint:")
        print("    pip install huggingface_hub")
        print("    export STABLEWM_HOME=~/.stable-wm")
        print("    huggingface-cli download quentinll/lewm-pusht \\")
        print("        --local-dir $STABLEWM_HOME/hf_pusht")
        print("\n  Then run:")
        print("    python analysis/waverider_embedding_probe.py \\")
        print("        --checkpoint $STABLEWM_HOME/hf_pusht/weights.pt \\")
        print("        --config    $STABLEWM_HOME/hf_pusht/config.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
