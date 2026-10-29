"""Config: Concerto base PARTIAL FINE-TUNE experiment.

Follow-up to the frozen probe. Encoder = the SAME official Concerto base
checkpoint and the SAME custom_config as the frozen experiment; only the
last encoder stage is unfrozen, plus a freshly initialized head.

Everything shared with the frozen experiment (data dirs, splits, grid size,
class weights, ignore label, seed, val protocol, model/head architecture,
ckpt path, patch-256 deviation) is IMPORTED from config.py, not redefined.

Selection metric / evaluation protocol: identical to the frozen probe
(validation scenes 3/14/16, FULL-SIZE rock IoU via GridSample inverse
mapping; grid-level metrics recorded alongside; per-epoch confusion
matrices). Test scenes 15/19/20 stay unused.
"""
import config as C  # noqa: F401  (shared, verified settings; see its docstring)

# ------------------------------------------------------------ unfreezing ----
# Stage-level partial fine-tune. Encoder stage modules (official
# PointTransformerV3, enc_mode=True, depths (3,3,3,12,3)):
#   embedding, enc.enc0 .. enc.enc4; enc.enc{s} contains `down` (s>0,
#   GridPooling) and `block0..block{depth-1}` (Block).
# Default (4,) = last stage only: enc.enc4.down + enc.enc4.block0..2 at
# 512 channels (~37M params incl. cpe SubMConv3d, ~34% of the 108.46M
# encoder). Stage 3 would add 12 blocks at 384 ch (~73M) -> ~full-model
# fine-tuning, too aggressive for a first partial-FT run.
UNFREEZE_STAGES = (4,)

# --------------------------------------------------------------- head -------
# Same architecture as the frozen probe (MLP 1088->128->GELU->Dropout->2).
# Random re-initialization (the frozen probe's head was fit to FROZEN
# features; features drift under fine-tuning, so a fresh head is the honest
# starting point). Init is governed by SEED, set before model construction.
HEAD_INIT = "random"  # only "random" implemented

# -------------------------------------------------------------- optimize ----
# Two AdamW param groups (frozen params never enter the optimizer).
ENC_LR = 1e-5       # unfrozen encoder stage
ENC_LR_MIN = 2e-7   # cosine floor, 2% of peak -- same floor ratio as exp01
HEAD_LR = 1e-4      # segmentation head (10x encoder lr, standard for partial FT)
HEAD_LR_MIN = 2e-6  # 2% of peak
WEIGHT_DECAY = 1e-4  # same as the frozen probe, applied to both groups
BATCH_SCENES = 1     # one whole scene per step
EPOCHS = 30          # max budget
EARLY_STOP_PATIENCE = 10  # epochs without val full-size rock-IoU improvement

# ------------------------------------------------------------------ runs ----
SEED = C.SEED  # 53124, same as the frozen probe
RUN_NAME = "concerto_base_partial_finetune"
SMOKE = dict(epochs=2, train_scenes=[1, 2, 4], val_scenes=[3], export_scenes=[3])
FINETUNE = dict(epochs=EPOCHS, train_scenes=C.TRAIN_SCENES,
                val_scenes=C.VAL_SCENES, export_scenes=C.VAL_SCENES)
