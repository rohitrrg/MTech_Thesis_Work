import torch
import os
from networks.afnonet import AFNONet

def get_model_size_and_params(model):
    """
    Calculates the total parameters and the memory footprint in MB.
    """
    # 1. Count the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # 2. Identify the size of a single parameter (standard is 4 bytes for float32)
    # We check the first parameter to be accurate
    param_size = next(model.parameters()).element_size() 
    
    # 3. Calculate total size in bytes and convert to Megabytes (MB)
    total_size_bytes = total_params * param_size
    total_size_mb = total_size_bytes / (1024 ** 2)
    
    return total_params, total_size_mb

def calculate_backbone_size(weight_path=None):
    """
    Instantiates the backbone and calculates its size.
    """
    # Define architecture manually as it is not in metadata
    class ParamsObject:
        def __init__(self, depth, **entries):
            self.__dict__.update(entries)
            self.depth = depth
            self.in_channels, self.out_channels = 20, 20

    base_params = {
        'patch_size': 8, 
        'embed_dim': 768, 
        'num_blocks': 8, 
        'img_size': (720, 1440)
    }

    # Recreate the 12-layer backbone
    backbone_params = ParamsObject(depth=12, **base_params)
    model = AFNONet(backbone_params)

    # Calculate metrics
    params_count, size_mb = get_model_size_and_params(model)

    print("-" * 40)
    print(f"Model: FourCastNet Backbone (12 Layers)")
    print(f"Total Parameters: {params_count:,}") # Should be ~74.6 Million
    print(f"Memory Size:      {size_mb:.2f} MB") # Should be ~285-300 MB
    
    # Optional: Compare with the physical file size on disk if path provided
    if weight_path and os.path.exists(weight_path):
        file_size_mb = os.path.getsize(weight_path) / (1024 ** 2)
        print(f"Physical File:    {file_size_mb:.2f} MB")
    print("-" * 40)

if __name__ == "__main__":
    # Point this to your actual backbone checkpoint
    calculate_backbone_size("weights/quantized models/backbone_fp16.ckpt")