import torch
from networks.afnonet import AFNONet
from collections import OrderedDict

# 1. Load your 12-layer backbone
params = {
        'patch_size': 8, 'embed_dim': 768, 'depth': 6, 'num_blocks': 8,
        'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
    }

class ParamsObject:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.in_channels, self.out_channels = 20, 20
        self.patch_size, self.num_blocks = 8, 8

params_obj = ParamsObject(**params)

model = AFNONet(params_obj)
checkpoint = torch.load("weights/fine-tuned models/pruned_6_layers_epoch_6.ckpt", weights_only=False)
state_dict = checkpoint.get('model_state', checkpoint)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
model.load_state_dict(new_state_dict, strict=False)

# 2. Convert to FP16
model_half = model.half() 

# 3. Save the new, smaller checkpoint
torch.save(model_half.state_dict(), "weights/quantized models/pruned_6_fp16.ckpt")