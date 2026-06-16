# 2. Methodology

This chapter provides a comprehensive blueprint of the experimental methods employed in this investigation. We detail the architecture of our baseline deep learning model, the specific optimization techniques applied, the dataset used for training and evaluation, and the metrics by which we measure both forecast skill and computational performance. Every choice was made to ensure a systematic, reproducible, and rigorous analysis.

## 2.1. Baseline Model Architecture: AFNO-based FourCastNet

The foundation of our research is a deep learning model inspired by NVIDIA's FourCastNet, a state-of-the-art architecture for data-driven weather prediction. The model's core innovation is its use of the **Adaptive Fourier Neural Operator (AFNO)**, a specialized type of Vision Transformer (ViT) that is exceptionally efficient for high-resolution, grid-like data such as global weather maps.

Our implementation, which we refer to as the `backbone` model, processes atmospheric data on a 720×1440 latitude-longitude grid. Following standard practice for AFNO-based models, the southernmost grid point at −90° latitude is excluded to yield an even grid compatible with the patch embedding layer.

The architecture can be broken down into three main stages:

1.  **Patching and Embedding:** The high-resolution input grid is first divided into a grid of smaller, non-overlapping patches. In our configuration, we use a patch size of 16×16 pixels. Each 16x16 patch, containing data for all input variables, is flattened into a one-dimensional vector. This vector is then projected into a high-dimensional embedding space via a linear transformation. This process converts the spatial grid into a sequence of patch embeddings, which is the standard input format for Transformer-based models.

2.  **AFNO Blocks:** The sequence of patch embeddings is then processed through a deep stack of identical AFNO blocks. These blocks are the workhorse of the model, responsible for learning the complex spatio-temporal relationships in the data. Each AFNO block contains two key sub-layers:

    *   **Fourier Transform Mixer:** This is the AFNO's signature component, replacing the computationally expensive multi-head self-attention mechanism found in standard Transformers. The mixer operates in the frequency domain to achieve a global receptive field with high efficiency. It first applies a 2D Fast Fourier Transform (FFT) to the patch embeddings. It then learns a sparse, block-diagonal weight matrix that it uses to mix information across all patches simultaneously. This sparse representation is key to its efficiency. Finally, it applies an Inverse FFT to transform the data back to the spatial domain. This allows the model to learn global interactions without the quadratic complexity of self-attention.
    *   **Feed-Forward Network (MLP):** Following the mixer, each patch embedding is passed through a standard multi-layer perceptron (MLP). This consists of two linear layers with a GELU (Gaussian Error Linear Unit) activation function in between. The MLP provides non-linear transformations and allows the model to learn complex features within each individual patch.

    Residual connections and layer normalization are used around both sub-layers to ensure stable training for this deep architecture.

3.  **Unpatching and Projection:** After passing through the final AFNO block, the sequence of processed patch embeddings is transformed back into a grid structure. A final linear projection layer then maps these embeddings to the desired output, producing the forecast for the next time step on the 720×1440 grid.

The full `backbone` model consists of a stack of 12 of these AFNO blocks, forming a deep neural network capable of learning highly complex weather dynamics.

## 2.2. Model Optimization Techniques

Our primary research goal is to make the powerful `backbone` model more efficient. To this end, we applied two well-established optimization strategies from the deep learning literature: pruning and quantization.

### 2.2.1. Pruning by Layer Removal

We employed a structured pruning approach, which involves removing entire components of the network. Specifically, we removed whole AFNO blocks from the model.

**Rationale and Strategy:**

This coarse-grained strategy was chosen over more fine-grained methods (like removing individual weights) for two main reasons:

1.  **Systematic Analysis:** Our objective was to create a spectrum of models with varying complexity to systematically study the relationship between model depth, forecast accuracy, and computational cost. Removing one layer at a time provides a clear, interpretable axis of model complexity.
2.  **Hardware Efficiency:** Modern deep learning hardware (GPUs, TPUs) is optimized for dense matrix operations. Unstructured pruning creates sparse matrices that often require specialized software or hardware to yield significant speedups. By removing entire blocks, we are left with a smaller but still dense model that can be executed with maximum efficiency on standard hardware.

Our pruning process started with the fully trained 12-layer `backbone` model. We then created a series of six pruned variants by progressively removing the final AFNO blocks. This "last-in, first-out" approach is based on the hypothesis that the initial layers of a deep network learn more general features, while the final layers perform more specialized refinements. Removing the final layers, therefore, should lead to a graceful degradation in performance.

The models created for this study are:
*   `backbone`: The full, 12-layer unpruned model.
*   `pruned_1`: The backbone with its final layer removed (11 layers).
*   `pruned_2`: The backbone with its final 2 layers removed (10 layers).
*   ...
*   `pruned_6`: The backbone with its final 6 layers removed (6 layers).

This setup allows us to precisely quantify the cost in forecast accuracy for each incremental gain in computational performance.

**Fine-Tuning Protocol:** After each pruning step, the resulting model was fine-tuned on the training set (2014–2015) for a maximum of 6 epochs. Fine-tuning used the Adam optimizer with an initial learning rate of 5×10⁻⁴, decayed by a cosine annealing schedule with a minimum learning rate of 1×10⁻⁶. The batch size was set to 1 per GPU. Early stopping was applied based on validation loss computed on the 2016 held-out set, with training halted if validation loss did not improve for 2 consecutive epochs. All fine-tuning experiments were conducted on a single NVIDIA A100 GPU. Each epoch of fine-tuning took approximately 30 minutes, for a maximum total fine-tuning time of 3 hours per pruned variant.

### 2.2.2. Quantization to FP16

Quantization is the process of reducing the numerical precision of a model's parameters and, in some cases, its activations. In this work, we investigate the impact of converting the model's weights from the standard 32-bit floating-point (FP32) precision to **16-bit floating-point (FP16)**, also known as half-precision.

This technique, applied after the model has been trained, offers two compelling benefits:

1.  **Memory Reduction:** Storing a number in FP16 requires half the memory of FP32. This immediately halves the model's size on disk and in memory, which can be critical for deployment on devices with limited resources.
2.  **Computational Speedup:** Modern GPUs and other AI accelerators contain specialized hardware units (e.g., NVIDIA's Tensor Cores) that can perform FP16 mathematical operations much faster than their FP32 counterparts. This can lead to a significant reduction in the time it takes to run a forecast (latency).

We applied FP16 quantization as a post-training step to the `backbone` model and to all six of the pruned model variants. This created a second, parallel set of models for evaluation (e.g., `backbone_fp16`, `pruned_1_fp16`, etc.), allowing us to study the effects of quantization both in isolation and in combination with pruning.

## 2.3. Dataset and Evaluation Variables

All experiments were conducted using the **ERA5 reanalysis dataset**, a product of the European Centre for Medium-Range Weather Forecasts (ECMWF). ERA5 is a high-quality, globally complete, and consistent historical record of the Earth's climate system, making it the de facto standard for training data-driven weather models.

*   **Evaluation Period:** The entire year of **2018** was held out as the test set. The model never saw this data during its training phase, ensuring an unbiased evaluation of its generalization performance.
*   **Temporal Resolution:** The data is provided at a **6-hour** temporal resolution (00:00, 06:00, 12:00, 18:00 UTC).
*   **Spatial Resolution:** The data is provided on a **0.25-degree latitude-longitude grid**, which corresponds to a resolution of 721 grid points for latitude and 1440 for longitude.

For evaluating forecast skill, we focused on five key atmospheric variables that are standard in meteorological literature:
*   `z500`: Geopotential height at the 500 hPa pressure level. This variable is crucial for tracking the movement of large-scale weather systems (Rossby waves).
*   `t850`: Temperature at the 850 hPa pressure level. This is often used to identify air masses and frontal systems.
*   `t2m`: Temperature at 2 meters above the surface. This is the variable most directly experienced by humans.
*   `u10`: The zonal (west-east) component of the wind at 10 meters above the surface.
*   `v10`: The meridional (north-south) component of the wind at 10 meters above the surface.

### 2.3.1. Data Split

The ERA5 dataset was divided into three non-overlapping splits to ensure a rigorous and unbiased evaluation:

- **Training Set (2014–2015):** Used exclusively for fine-tuning the pruned model variants after layer removal. All normalization statistics (global mean and standard deviation per channel) were computed from this split only, and were never recomputed using validation or test data.
- **Validation Set (2016):** Used for early stopping during fine-tuning and for all hyperparameter selection decisions, including the number of fine-tuning epochs and the learning rate schedule. Note that 2016 is a leap year and therefore contains 1,464 timesteps at 6-hourly resolution.
- **Test Set (2018):** Held out entirely until the final evaluation. No model selection, hyperparameter tuning, or architectural decision was made based on 2018 data. All forecast skill metrics reported in Chapter 3 are computed exclusively on this split.

This split is consistent with the WeatherBench2 evaluation protocol, which designates 2018 as the standard out-of-sample test year for ERA5-based weather forecasting models.

## 2.4. Evaluation Metrics

To quantify model performance, we used two of the most common metrics in the field of meteorology. Both metrics are weighted by latitude to account for the convergence of grid cells near the poles.

1.  **Root Mean Square Error (RMSE):** This metric measures the average magnitude of the error between the forecast ($F$) and the ground truth from the ERA5 dataset ($T$). It is a direct measure of forecast error and is sensitive to large, infrequent errors. It is calculated as:
    $$ \text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (F_i - T_i)^2} $$
    A lower RMSE indicates a more accurate forecast. The units of RMSE are the same as the units of the variable being measured.

2.  **Anomaly Correlation Coefficient (ACC):** This metric measures the spatial correlation between the forecast anomaly and the true anomaly. The "anomaly" is the deviation of a field from its long-term climatological mean ($C$). ACC is a measure of pattern similarity and is less sensitive to a model having a systematic bias (e.g., being consistently too warm or too cold). It is a crucial metric for judging whether the model has correctly predicted the location and shape of weather systems. The formula is:
    $$ \text{ACC} = \frac{\sum (F' \cdot T')}{\sqrt{\sum (F')^2 \cdot \sum (T')^2}} $$
    where $F' = F - C$ and $T' = T - C$. An ACC of 1 represents a perfect forecast pattern, while an ACC of 0 indicates no skill.

## 2.5. Meteorological Baselines

A skillful forecast model must do better than simple, naive assumptions. To provide a firm context for the performance of our AI models, we compared them against two universally accepted meteorological baselines:

1.  **Persistence Forecast:** This baseline makes the simple assumption that the weather will not change over the forecast lead time. The forecast for a future time is simply the state of the atmosphere at the initial time of the forecast. While simple, it is a surprisingly difficult baseline to beat for very short lead times (e.g., 6 hours).
2.  **Climatology Forecast:** This baseline predicts that the weather at any given time will be the long-term historical average for that specific day and time of year. This forecast has no knowledge of the current state of the atmosphere and thus represents a forecast with zero predictive skill regarding short-term weather dynamics. Any useful model must have a higher ACC than climatology.

## 2.6. Computational Performance Measurement

The second major pillar of our evaluation is computational performance. To quantify the efficiency gains from our optimization strategies, we measured the following for each model variant:

*   **Parameters:** The total number of trainable parameters in the model. This is a direct measure of the model's size in memory and on disk.
*   **Latency:** The wall-clock time required to execute a single forecast step for one input sample, measured in milliseconds (ms). This was measured on the target hardware (NVIDIA A100 GPU) to get a realistic measure of inference speed.
*   **Speedup:** A normalized metric that makes it easy to compare models. It is calculated as the ratio of the `backbone` model's latency to the optimized model's latency. For example, a speedup of 2.0x means the model is twice as fast as the original `backbone` model.

**Note on Baseline Latency:** The backbone model latency reported in this work (284 ms) differs from the 142 ms reported in earlier single-step evaluations of the same architecture. This difference arises from two factors: (1) the autoregressive multi-step rollout evaluation protocol used in this work, where latency is measured as the average per-step time across a full 28-step forecast sequence, versus a single isolated forward pass, and (2) the input grid resolution used (720×1440 vs. the earlier 721×1440 configuration). All latency comparisons within this paper are made under identical conditions and are therefore internally consistent.