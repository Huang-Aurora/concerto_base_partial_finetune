"""Experiment config: Concerto base (official pretrained PTv3 encoder) on the
rock-block binary segmentation task. Kaggle-adapted standalone version.

Training protocol is IDENTICAL to the original frozen probe
(concerto-rock-probe, exp01): same official checkpoint (concerto_base),
same custom_config (enc_patch_size=256, enable_flash=False), same data
splits, same class weights, same seed, same evaluation protocol.
Only paths are adapted for Kaggle:
  - data is read from a Kaggle INPUT dataset (/kaggle/input/<slug>),
    auto-detected (structure data/ + label/ + splits/); override with the
    DATA_ROOT env var. A sibling RandLA-Net repo is used as local fallback
    so the folder also runs off-Kaggle.
  - checkpoints / logs / results -> /kaggle/working/outputs (OUTPUT_ROOT)
  - encoder checkpoint -> /kaggle/working/ckpt (CKPT_DIR), auto-downloaded
    by bootstrap.py if absent
  - coord/color npy cache + estimated normals -> <OUTPUT_ROOT>/cache
"""
import os

WORK_ROOT = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------- kaggle ----
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/kaggle/working/outputs")
CKPT_DIR = os.environ.get("CKPT_DIR", "/kaggle/working/ckpt")
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.join(OUTPUT_ROOT, "cache"))
RAW_CACHE_DIR = os.path.join(CACHE_DIR, "raw")          # coord/color npy cache
NORMAL_CACHE_DIR = os.path.join(CACHE_DIR, "normals")   # estimated normals


def _find_data_root():
    env = os.environ.get("DATA_ROOT")
    if env:
        return env
    kaggle_input = "/kaggle/input"
    if os.path.isdir(kaggle_input):
        for slug in sorted(os.listdir(kaggle_input)):
            cand = os.path.join(kaggle_input, slug)
            if all(os.path.isdir(os.path.join(cand, d))
                   for d in ("data", "label", "splits")):
                return cand
        raise FileNotFoundError(
            f"no Kaggle input dataset with data/+label/+splits/ under "
            f"{kaggle_input}; upload the RandLA rock data as a dataset and "
            f"attach it, or set DATA_ROOT")
    # off-Kaggle fallback: sibling RandLA-Net repository
    local = os.environ.get(
        "LOCAL_RANDLA_ROOT",
        os.path.join(WORK_ROOT, os.pardir, "RandLA-Net-Pytorch-New-master"))
    if os.path.isdir(os.path.join(local, "data")):
        return local
    raise FileNotFoundError(
        f"data root not found; set DATA_ROOT (got fallback {local})")


DATA_ROOT = _find_data_root()
DATA_DIR = os.path.join(DATA_ROOT, "data")     # <id>.txt  xyz+rgb, header line
LABEL_DIR = os.path.join(DATA_ROOT, "label")   # <id>.npy  0 bg / 1 rock / 2 ignore
SPLIT_DIR = os.path.join(DATA_ROOT, "splits")  # train.txt / val.txt / test.txt

# --------------------------------------------------------------- Concerto ---
# Official pre-trained weight (Pointcept/Concerto on HuggingFace). Real
# pretrained checkpoint; verified in the frozen probe: config in_channels=9,
# enc_channels=(48,96,192,384,512), enc_mode=True, 453 state_dict keys,
# 108.46M params. bootstrap.py downloads it here if missing.
CONCERTO_CKPT = os.path.join(CKPT_DIR, "concerto_base.pth")
# Official default preprocessing (concerto.transform.default(), byte-identical
# to sonata's): CenterShift, GridSample(0.02, fnv), NormalizeColor,
# Collect feat=(coord, color, normal).
GRID_SIZE = 0.02
# enable_flash=False because flash_attn is not available. Non-flash
# SerializedAttention materializes (N_patch, H, K, K) attention maps; with
# ~1M grid points the official K=1024 needs >20 GiB per layer (OOM), so per
# the README's own advice ("reduce patch size if necessary") K is 256
# (peak attention tensor ~3 GiB/layer). SAME deviation as the frozen probe.
CONCERTO_CUSTOM_CONFIG = dict(enc_patch_size=[128] * 5, enable_flash=False)

# ----------------------------------------------------------------- model ----
# Point-level feature dim after the official README upcast:
# enc_channels[2] + enc_channels[3] + enc_channels[4] = 192 + 384 + 512 = 1088
UPCAST_DIM = 1088
HEAD_HIDDEN = 128
HEAD_DROPOUT = 0.1
NUM_CLASSES = 2
IGNORE_LABEL = 2  # 2 = ignore (excluded from loss and metrics)

# ------------------------------------------------------------------ loss ----
# Class imbalance on the TRAIN split: bg 81.12% / rock 18.88%. Weighted CE
# with the RandLA-Net project scheme w_c = 1 / (frac_c + 0.02) -> [1.203, 4.789].
CLASS_WEIGHTS = [1.203, 4.789]

# -------------------------------------------------------------- optimize ----
# Head-only values used by the FROZEN experiment (train.py); the fine-tune
# experiment overrides/extends them in config_finetune.py.
LR = 1e-3
WEIGHT_DECAY = 1e-4
LR_MIN = 1e-5
BATCH_SCENES = 1  # one whole scene per step (Pointcept-style full-scene batch)

# ----------------------------------------------------------------- splits ---
def load_split(name):
    with open(os.path.join(SPLIT_DIR, f"{name}.txt")) as f:
        return [int(x) for x in f.read().split()]


TRAIN_SCENES = load_split("train")  # 14 scenes: 1 2 4 5 6 7 8 9 10 11 12 13 17 18
VAL_SCENES = load_split("val")      # 3 scenes: 3 14 16
TEST_SCENES = load_split("test")    # 3 scenes: 15 19 20 (NOT used for tuning)

# --------------------------------------------------------------- normals ----
# The rock data has xyz+rgb only, while the official weight requires a
# 9-channel input (coord+color+normal). Normals are ESTIMATED from geometry:
# per-scene kNN (scipy cKDTree) + local PCA, oriented so z >= 0 (terrain
# convention), cached as npy under the working cache dir.
NORMAL_K = 16

# ------------------------------------------------------------------ runs ----
VAL_SEED_BASE = 20260  # fixed GridSample seed per val scene -> deterministic val
SMOKE = dict(epochs=2, train_scenes=[1, 2, 4], val_scenes=[3], export_scenes=[3])
EXP01 = dict(epochs=15, train_scenes=TRAIN_SCENES, val_scenes=VAL_SCENES,
             export_scenes=VAL_SCENES)

# ------------------------------------------- exploratory encoder variant ----
# Set env PROBE_ENCODER=concerto_large_outdoor to switch to the outdoor-specific
# large model (208M; enc_channels=(64,128,256,512,768)). EXPLORATORY only
# (differs in BOTH capacity and pretraining data; not controlled vs exp01).
if os.environ.get("PROBE_ENCODER", "") == "concerto_large_outdoor":
    CONCERTO_CKPT = os.path.join(CKPT_DIR, "concerto_large_outdoor.pth")
    UPCAST_DIM = 256 + 512 + 768  # 1536, from enc_channels[2..4]

SEED = 53124
