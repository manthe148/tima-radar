"""Single source of truth for v8 paths. v7 + raw data are READ-ONLY inputs."""
import os

V8   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # /workspace/v8
TEST = os.path.join(os.path.dirname(V8), "test")                     # /workspace/test (frozen v7 + data)

# frozen v7 inputs (do NOT edit in place)
CKPT        = os.path.join(TEST, "tornet_resnet_v7_windboost.pt")
TRAIN_MOD   = TEST                      # train_resnet.py lives here (model class + RANGES/MOMENTS)
MANIFEST    = os.path.join(TEST, "dataset_manifest.csv")
VOL_ROOT    = os.path.join(TEST, "radar_volumes")
NPZ_POS     = os.path.join(TEST, "npz_out")
NPZ_NEG     = os.path.join(TEST, "npz_negatives")

# v8 outputs
DATA        = os.path.join(V8, "data")
MORPH_CSV   = os.path.join(DATA, "mode_morphology.csv")
