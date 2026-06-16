import torch
import torch.nn as nn
from networks.afnonet import AFNONet
from collections import OrderedDict

# -----------------------------
# 1. Load model architecture
# -----------------------------
params = {
        'patch_size': 8,
        'embed_dim': 768,
        'depth': 12,
        'num_blocks': 8,
        'N_in_channels': 20,
        'N_out_channels': 20,
        'img_size': (720, 1440)
    }

class ParamsObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.in_channels, self.out_channels = 20, 20
        self.patch_size, self.num_blocks = 8, 8

params_obj = ParamsObject(**params)

model = AFNONet(params_obj)

# -----------------------------
# 2. Load checkpoint
# -----------------------------
checkpoint = torch.load("weights/backbone.ckpt", map_location="cpu", weights_only=False)
state_dict = checkpoint.get('model_state', checkpoint)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=False)

model.eval()

# -----------------------------
# 3. Apply INT8 Dynamic Quantization
# -----------------------------
quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},   # Quantize Linear layers
        dtype=torch.qint8
)

# -----------------------------
# 4. Save INT8 model
# -----------------------------
torch.save(quantized_model.state_dict(), "weights/backbone_int8.ckpt")

print("INT8 model saved successfully!")