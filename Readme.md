# Knowledge Transfer: Pruning, Fine-tuning, and Evaluation Workflow

This document provides a detailed guide on the experimental workflow used for model compression, focusing on layer similarity analysis, pruning, fine-tuning, and evaluation.

## 1. Preliminary Analysis: Layer Similarity

Before pruning, you can analyze the similarity between the layers of a trained model. This can help inform which layers are most redundant and are therefore good candidates for pruning. The `find_redundant.py` script uses Centered Kernel Alignment (CKA) to compare the feature representations of each layer in the `backbone` model.

**Command:**
This command loads the pre-trained `backbone.ckpt` and computes the CKA similarity matrix between all its layers. The output is a plot of the similarity matrix saved as a `.png` file.

```bash
cd /FourCastNet

python find_redundant.py \
    --config AFNO \
    --config_file config/AFNO.yaml \
    --weights weights/backbone.ckpt
```
This analysis provides a quantitative basis for the hypothesis that adjacent layers, especially near the end of the network, are often the most similar and thus the most safely prunable.

## 2. Step-by-Step Workflow: Pruning, Fine-tuning, Evaluation

The core experimental process is a cycle: prune, fine-tune, and then evaluate.

### Step 1: How to Prune & Fine-Tune

Pruning and fine-tuning are performed in a single step by the `fine-tune.py` script. It loads a pre-trained model, removes a specified number of layers from the end of the network *in memory*, and then begins the fine-tuning process on this new, smaller architecture.

**Fine-tuning Strategy:**
*   **Load Pre-trained Weights:** Always start from the fully trained `backbone.ckpt`.
*   **Specify Pruning Level:** Use the `--prune_layers` argument to define how many layers to remove. For `pruned_1`, set `--prune_layers 1`. For `pruned_6`, set `--prune_layers 6`.
*   **Continue Training:** The script continues training for a set number of epochs to allow the smaller model to adapt.

**Command Example (for `pruned_3`):**
This command loads `backbone.ckpt`, removes 3 layers, and then fine-tunes the resulting 9-layer model, saving the final checkpoint as `weights/pruned_3.ckpt`.

```bash
cd /FourCastNet

python fine-tune.py \
    --config AFNO \
    --config_file config/AFNO.yaml \
    --prune_layers 3 \
    --finetune_path weights/pruned_3.ckpt
```
*You would repeat this command for each pruning level (1 through 6) to generate the full spectrum of models.*

### Step 2: How to Evaluate Pruned Models

Once a pruned and fine-tuned model is saved, you must evaluate its performance to quantify the impact of the compression.

**Evaluation Strategy:**
*   **Held-out Test Set:** All evaluations are performed on the 2018 dataset.
*   **Standard Metrics:** Use latitude-weighted Root Mean Square Error (RMSE) and Anomaly Correlation Coefficient (ACC).
*   **Multi-Lead-Time Evaluation:** Evaluate across the entire forecast horizon (e.g., up to 7 days) to see how error accumulates.

**Command Example (for `pruned_3`):**
This command evaluates the `pruned_3.ckpt` checkpoint.

```bash
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
*   **Crucially**, you must include the `--pruned` flag. This tells the evaluation script that the checkpoint being loaded does not match the full 12-layer architecture in the `AFNO.yaml` config.

## 3. Further Research & Advanced Concepts

This workflow provides a solid foundation. Further work could explore:

*   **Alternative Pruning Strategies:** Instead of removing layers from the end, one could investigate removing layers from the middle of the network based on the CKA similarity results.
*   **Quantization-Aware Training (QAT):** For more aggressive quantization (e.g., to INT8), post-training quantization is often insufficient. QAT involves simulating the effects of quantization during the fine-tuning process itself.
