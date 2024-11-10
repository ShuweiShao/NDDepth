import os
import sys

file_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_directory, "../../../DepthAnything/metric_depth"))

import torch

from depth_anything_v2.dpt import DepthAnythingV2

# These functions aim to fit based on disparity map

def get_model(DEVICE, MODEL_PATH, encoder='vitl', max_depth=20.0):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})

    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE).eval()
    return model
