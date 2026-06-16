# Knowledge Transfer: Pruning, Fine-tuning, and Evaluation Workflow

This document provides a detailed guide on the experimental workflow used for model compression, focusing on layer pruning, fine-tuning, and evaluation.

## 1. Rationale: Pruning Strategy

The core idea of our structured pruning is to remove entire layers from a pre-trained model. This is a "coarse-grained" approach that guarantees a smaller model and is hardware-friendly, leading to real-world speedups. We chose to remove layers from the *end* of the network, based on the hypothesis that initial layers capture fundamental, large-scale patterns, while later layers are responsible for finer-scale refinements. Removing the refinement layers should, in theory, lead to a more graceful degradation of performance than removing the foundational layers.

## 2. Step-by-Step Workflow

The process is a cycle: prune, fine-tune, and then evaluate.

### Step 1: How to Prune a Model

Pruning is not a separate script but is integrated into the fine-tuning process. The `fine-tune.py` script takes a pre-trained model, removes a specified number of layers from the end of the network *in memory*, and then begins the fine-tuning process on this new, smaller architecture.

**Key Concept:** You are not saving a separate "pruned" model file before training. You are creating a smaller model dynamically at the start of the fine-tuning run.

### Step 2: How to Fine-Tune Pruned Models

Fine-tuning is the most critical step. After removing layers, the model's predictive capabilities are damaged. Fine-tuning on the original training dataset helps the remaining layers adapt and recover a significant amount of the lost accuracy.

**Fine-tuning Strategy:**
*   **Load Pre-trained Weights:** Always start from the fully trained `backbone.ckpt`. This ensures you are transferring the learned knowledge from the best possible model.
*   **Specify Pruning Level:** Use the `--prune_layers` argument to define how many layers to remove. For `pruned_1`, you set `--prune_layers 1`. For `pruned_6`, you set `--prune_layers 6`.
*   **Continue Training:** The script then continues the training process for a set number of epochs. This is typically a shorter duration than the original training from scratch, as the model only needs to adapt, not learn from a random initialization.

**Command Example (for `pruned_3`):**
This command loads `backbone.ckpt`, removes 3 layers, and then fine-tunes the resulting 9-layer model, saving the final checkpoint as `weights/pruned_3.ckpt`.

```bash
# Inside the Docker container
cd /FourCastNet

mpirun -np <num_gpus> python fine-tune.py \
    --config AFNO \
    --config_file config/AFNO.yaml \
    --prune_layers 3 \
    --finetune_path weights/pruned_3.ckpt
```
*You would repeat this command for each pruning level (1 through 6) to generate the full spectrum of models.*

### Step 3: How to Evaluate Pruned Models

Once a pruned and fine-tuned model is saved, you must evaluate its performance to quantify the impact of the compression.

**Evaluation Strategy:**
*   **Held-out Test Set:** All evaluations are performed on the 2018 dataset, which was not used during training or fine-tuning, to ensure an unbiased assessment of the model's generalization capabilities.
*   **Standard Metrics:** We use latitude-weighted Root Mean Square Error (RMSE) and Anomaly Correlation Coefficient (ACC) as the primary metrics for forecast skill.
*   **Multi-Lead-Time Evaluation:** It is crucial to evaluate performance not just at the start of the forecast but across the entire forecast horizon (e.g., from 6 hours to 168 hours/7 days). This reveals how error accumulates over time and can expose issues like the `t2m` performance cliff we observed.
*   **FP16 vs FP32:** Evaluate the model in both full precision (`--pqt_config_file config/pqt_fp32.yaml`) and half precision (`--pqt_config_file config/pqt_fp16.yaml`) to measure the impact of quantization separately from pruning.

**Command Example (for `pruned_3`):**
This command evaluates the `pruned_3.ckpt` checkpoint.

```bash
# Inside the Docker container
cd /FourCastNet

# Evaluate in FP32
python evaluate.py \
    --config AFNO \
    --config_file config/AFNO.yaml \
    --weights weights/pruned_3.ckpt \
    --pruned

# Evaluate in FP16 (if using a specific script for it)
python evaluate_pqt_fp16_32.py \
    --config AFNO \
    --config_file config/AFNO.yaml \
    --weights weights/pruned_3.ckpt \
    --pruned
```
*   **Crucially**, you must include the `--pruned` flag. This tells the evaluation script that the checkpoint being loaded does not match the full 12-layer architecture in the `AFNO.yaml` config, so it should dynamically build the correct smaller model before loading the weights.

## 3. Further Research & Advanced Concepts

This workflow provides a solid foundation. Further work could explore:

*   **Alternative Pruning Strategies:** Instead of removing layers from the end, one could investigate removing layers from the middle of the network.
*   **Layer Similarity Analysis:** Before pruning, one could analyze the similarity between layers to make a more informed choice about which layers to remove. Techniques like Centered Kernel Alignment (CKA) can be used to measure the similarity of representations between layers. The `find_redundant.py` script in the codebase was an initial exploration of this idea.
*   **Quantization-Aware Training (QAT):** For more aggressive quantization (e.g., to INT8), post-training quantization is often insufficient. QAT involves simulating the effects of quantization during the fine-tuning process itself to help the model adapt to the lower precision, which can significantly improve the final accuracy of the quantized model.
