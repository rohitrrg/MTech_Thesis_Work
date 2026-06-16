# # import torch
# # from networks.afnonet import AFNONet
# # import matplotlib.pyplot as plt

# # def analyze_model_breakdown(params_obj):
# #     model = AFNONet(params_obj)
    
# #     transformer_params = 0
# #     projection_params = 0
# #     pos_embed_params = 0
# #     other_params = 0

# #     for name, param in model.named_parameters():
# #         num_params = param.numel()
        
# #         # Category: Transformer Blocks (The main backbone)
# #         if "blocks" in name:
# #             transformer_params += num_params
        
# #         # Category: Patch Embedding and Head (The "rest")
# #         elif "patch_embed" in name or "head" in name:
# #             projection_params += num_params
            
# #         # Category: Positional Embeddings
# #         elif "pos_embed" in name:
# #             pos_embed_params += num_params
            
# #         else:
# #             other_params += num_params

# #     total_params = transformer_params + projection_params + pos_embed_params + other_params
    
# #     # Calculate Percentages
# #     transformer_pct = (transformer_params / total_params) * 100
# #     rest_pct = 100 - transformer_pct

# #     print(f"--- FourCastNet Model Breakdown ---")
# #     print(f"Total Parameters: {total_params:,}")
# #     print(f"{'-'*35}")
# #     print(f"Transformer Blocks:  {transformer_params:,} ({transformer_pct:.2f}%)")
# #     print(f"Projection Layers:   {projection_params:,}")
# #     print(f"Positional Embeds:   {pos_embed_params:,}")
# #     print(f"Other:               {other_params:,}")
# #     print(f"{'-'*35}")
# #     print(f"Final Ratio: {transformer_pct:.1f}% Transformer / {rest_pct:.1f}% Rest")

# #     # Optional: Visualization for MTP Report
# #     labels = ['Transformer Blocks', 'Projection/Rest', 'Positional Embeds']
# #     sizes = [transformer_params, projection_params, pos_embed_params]
# #     colors = ['#ff9999','#66b3ff','#99ff99']
    
# #     plt.figure(figsize=(8, 6))
# #     plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
# #     plt.title("FourCastNet Parameter Distribution")
# #     plt.show()

# # if __name__ == "__main__":
# #     # Standard 12-layer parameters
# #     params = {
# #         'patch_size': 8, 'embed_dim': 768, 'depth': 12, 'num_blocks': 8,
# #         'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
# #     }

# #     class ParamsObject:
# #         def __init__(self, **entries):
# #             self.__dict__.update(entries)
# #             self.in_channels, self.out_channels = 20, 20

# #     analyze_model_breakdown(ParamsObject(**params))


# import torch
# from networks.afnonet import AFNONet
# from collections import OrderedDict

# def analyze_actual_model(params_obj, checkpoint_path):
#     # 1. Instantiate the architecture
#     model = AFNONet(params_obj)
    
#     # 2. Load your actual weights
#     # This ensures we are counting exactly what is in your file
#     checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
#     state_dict = checkpoint.get('model_state', checkpoint)
    
#     # Handling the 'module.' prefix if saved with DDP
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         name = k[7:] if k.startswith('module.') else k
#         new_state_dict[name] = v
    
#     model.load_state_dict(new_state_dict, strict=False)
    
#     transformer_params = 0
#     projection_params = 0 # Patch embed and Head
#     other_params = 0

#     for name, param in model.named_parameters():
#         n = param.numel()
#         if "blocks" in name:
#             transformer_params += n
#         elif "patch_embed" in name or "head" in name:
#             projection_params += n
#         else:
#             other_params += n

#     total = transformer_params + projection_params + other_params
    
#     print(f"Analysis for: {checkpoint_path}")
#     print(f"Total Parameters: {total:,}")
#     print(f"Transformer Blocks: {transformer_params:,} ({100*transformer_params/total:.1f}%)")
#     print(f"Rest (Embed/Head/Pos): {projection_params + other_params:,} ({100*(projection_params+other_params)/total:.1f}%)")

# if __name__ == "__main__":
#     # Standard 12-layer backbone params
#     params = {
#         'patch_size': 8, 'embed_dim': 768, 'num_blocks': 8,
#         'N_in_channels': 20, 'N_out_channels': 20, 'img_size': (720, 1440)
#     }

#     class ParamsObject:
#         def __init__(self, **entries):
#             self.__dict__.update(entries)
#             self.in_channels, self.out_channels = 20, 20

#     # Pass your backbone path here
#     analyze_actual_model(ParamsObject(depth=11, **params), "weights/pruned models/pruned_1_layers.ckpt")


import torch

def count_params_in_file(checkpoint_path):
    # 1. Load the raw dictionary from the disk
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 2. Get the state_dict (handles both raw dicts and lightning-style checkpoints)
    state_dict = checkpoint.get('model_state', checkpoint)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']

    total_params = 0
    block_params = 0
    other_params = 0
    layers_found = set()

    for name, param in state_dict.items():
        # param is a tensor, param.numel() gives the count
        count = param.numel()
        total_params += count
        
        if "blocks." in name:
            block_params += count
            # Identify which layer index this belongs to (e.g., blocks.0, blocks.1)
            layer_idx = name.split('.')[1]
            layers_found.add(layer_idx)
        else:
            other_params += count

    print(f"--- Checkpoint File Analysis ---")
    print(f"File: {checkpoint_path}")
    print(f"Detected Depth (Transformer Layers): {len(layers_found)}")
    print(f"Total Parameters in File: {total_params:,}")
    print(f"  - Transformer Params: {block_params:,}")
    print(f"  - Embedding/Head Params: {other_params:,}")
    print("-" * 35)

if __name__ == "__main__":
    # Compare both files side-by-side
    # count_params_in_file("weights/backbone.ckpt")
    count_params_in_file("weights/pruned models/pruned_6_layers.ckpt")