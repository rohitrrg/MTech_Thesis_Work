# Appendix

This appendix provides supplementary material to ensure the reproducibility and transparency of the research presented in this booklet. It includes details about the computational environment and key source code snippets that define the model architecture and the training procedure.

## A.1. Computational Environment

All experiments were conducted in the following environment:

*   **Hardware:**
    *   **GPU:** NVIDIA A100-SXM4-40GB
    *   **CPU:** AMD EPYC 7742 64-Core Processor
*   **Software:**
    *   **Operating System:** Ubuntu 20.04.4 LTS
    *   **CUDA Version:** 11.6
    *   **PyTorch Version:** 1.12.1+cu116
    *   **Torchvision Version:** 0.13.1+cu116
    *   **Other Key Libraries:** `numpy`, `h5py`, `wandb`, `timm`, `einops`

## A.2. Source Code: Model Architecture

The core AFNO model architecture is defined in `FourCastNet/networks/afnonet.py`. The following snippet shows the `AFNONet` class, which is the main building block of our `backbone` model, and the `Block` class, which defines a single layer of the network.

```python
# From: /home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/networks/afnonet.py

class AFNONet(nn.Module):
    def __init__(
            self,
            params,
            img_size=(720, 1440),
            patch_size=(16, 16),
            in_chans=2,
            out_chans=2,
            embed_dim=768,
            depth=12,
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.params = params
        self.img_size = img_size
        self.patch_size = (params.patch_size, params.patch_size)
        self.in_chans = params.in_channels
        self.out_chans = params.out_channels
        self.num_features = self.embed_dim = embed_dim
        self.num_blocks = params.num_blocks
        depth = params.depth 
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        print('Dept Inside AFNONet:', depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.h = img_size[0] // self.patch_size[0]
        self.w = img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            num_blocks=self.num_blocks, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w (p1 p2 c_out) -> b c_out (h p1) (w p2)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            h=self.img_size[0] // self.patch_size[0],
            w=self.img_size[1] // self.patch_size[1],
        )
        return x

class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x
```

## A.3. Source Code: Training Script

The main training and evaluation logic is contained in `FourCastNet/train.py`. The `Trainer` class handles data loading, model initialization, the training loop, and validation. The `train_one_epoch` method is the core of the training process.

```python
# From: /home/gpuuser0/gpuuser0_a/rohit/rohitg/weather-models/FourCastNet/FourCastNet/train.py

class Trainer():
  def __init__(self, params, world_rank):
    
    self.params = params
    self.world_rank = world_rank
    self.device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # ... (WandB and Data Loader Initialization) ...

    if params.nettype == 'afno':
      self.model = AFNONet(params).to(self.device) 
    else:
      raise Exception("not implemented")
     
    # ... (Optimizer, Scheduler, and DDP Initialization) ...

  def train(self):
    if self.params.log_to_screen:
      logging.info("Starting Training Loop...")

    best_valid_loss = 1.e6
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_time, valid_logs = self.validate_one_epoch()
      
      # ... (Scheduler and Checkpointing Logic) ...

  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    self.model.train()
    
    for i, data in enumerate(self.train_data_loader, 0):
      self.iters += 1
      inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)      
      
      # ... (Data Handling Logic) ...

      tr_start = time.time()

      self.model.zero_grad()
      with amp.autocast(self.params.enable_amp):
        gen = self.model(inp).to(self.device, dtype = torch.float)
        loss = self.loss_obj(gen, tar)

      if self.params.enable_amp:
        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
        self.gscaler.update()
      else:
        loss.backward()
        self.optimizer.step()

      tr_time += time.time() - tr_start
    
    # ... (Logging Logic) ...

    return tr_time, data_time, logs

  def validate_one_epoch(self):
    self.model.eval()
    # ... (Validation Logic with RMSE/ACC calculation) ...
    return valid_time, logs

if __name__ == '__main__':
  # ... (Argument Parsing and Trainer Initialization) ...
  trainer = Trainer(params, world_rank)
  trainer.train()
```
