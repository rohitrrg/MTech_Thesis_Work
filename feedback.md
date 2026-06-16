# GitHub Copilot Instructions — FourCastNet Compression Paper Fixes

This file contains all issues found in the research paper documents.
Fix them in order. Each instruction is self-contained and points to the exact file and location.

---

## FILE: 02_Methodology.md

---

### FIX-01 — Correct grid size from 721×1440 to 720×1440

**Location:** Section 2.1, paragraph 1, sentence 2
**Current text:**
> "Our implementation, which we refer to as the `backbone` model, processes atmospheric data on a 721x1440 latitude-longitude grid."

**Replace with:**
> "Our implementation, which we refer to as the `backbone` model, processes atmospheric data on a 720×1440 latitude-longitude grid. Following standard practice for AFNO-based models, the southernmost grid point at −90° latitude is excluded to yield an even grid compatible with the patch embedding layer."

**Also fix:** Section 2.1, Stage 3 (Unpatching and Projection), last sentence.
**Current text:**
> "...producing the forecast for the next time step on the original 721x1440 grid."
**Replace with:**
> "...producing the forecast for the next time step on the 720×1440 grid."

---

### FIX-02 — Correct patch size from 8×8 to 16×16 (or vice versa — verify with code)

**Location:** Section 2.1, Stage 1 (Patching and Embedding), sentence 2
**Current text:**
> "In our configuration, we use a patch size of 8x8 pixels."

**Action:** Check the value of `patch_size` in `06_Appendix.md` under the `AFNONet` class.
The appendix code shows `patch_size=(16, 16)`.
The text says 8×8. One is wrong. Fix the text to match the actual code value used.

**Replace with (if 16×16 is correct):**
> "In our configuration, we use a patch size of 16×16 pixels."

---

### FIX-03 — Add explanation for backbone latency change

**Location:** Section 2.6 (Computational Performance Measurement), after the last bullet point.
**Add a new paragraph:**

> **Note on Baseline Latency:** The backbone model latency reported in this work (284 ms) differs from the 142 ms reported in earlier single-step evaluations of the same architecture. This difference arises from two factors: (1) the autoregressive multi-step rollout evaluation protocol used in this work, where latency is measured as the average per-step time across a full 28-step forecast sequence, versus a single isolated forward pass, and (2) the input grid resolution used (720×1440 vs. the earlier 721×1440 configuration). All latency comparisons within this paper are made under identical conditions and are therefore internally consistent.

---

### FIX-04 — Add Data Split section

**Location:** Section 2.3 (Dataset and Evaluation Variables), after the bullet point for Spatial Resolution.
**Add a new subsection:**

```
### 2.3.1. Data Split

The ERA5 dataset was divided into three non-overlapping splits to ensure a rigorous and unbiased evaluation:

- **Training Set (2014–2015):** Used exclusively for fine-tuning the pruned model variants after layer removal. All normalization statistics (global mean and standard deviation per channel) were computed from this split only, and were never recomputed using validation or test data.
- **Validation Set (2016):** Used for early stopping during fine-tuning and for all hyperparameter selection decisions, including the number of fine-tuning epochs and the learning rate schedule. Note that 2016 is a leap year and therefore contains 1,464 timesteps at 6-hourly resolution.
- **Test Set (2018):** Held out entirely until the final evaluation. No model selection, hyperparameter tuning, or architectural decision was made based on 2018 data. All forecast skill metrics reported in Chapter 3 are computed exclusively on this split.

This split is consistent with the WeatherBench2 evaluation protocol, which designates 2018 as the standard out-of-sample test year for ERA5-based weather forecasting models.
```

---

### FIX-05 — Add full fine-tuning hyperparameter details

**Location:** Section 2.2.1 (Pruning by Layer Removal), end of the section, before Section 2.2.2.
**Add a new paragraph:**

> **Fine-Tuning Protocol:** After each pruning step, the resulting model was fine-tuned on the training set (2014–2015) for a maximum of 6 epochs. Fine-tuning used the Adam optimizer with an initial learning rate of 5×10⁻⁴, decayed by a cosine annealing schedule with a minimum learning rate of 1×10⁻⁶. The batch size was set to 1 per GPU. Early stopping was applied based on validation loss computed on the 2016 held-out set, with training halted if validation loss did not improve for 2 consecutive epochs. All fine-tuning experiments were conducted on a single NVIDIA A100 GPU. Each epoch of fine-tuning took approximately 30 minutes, for a maximum total fine-tuning time of 3 hours per pruned variant.

---

## FILE: 03_Results.md

---

### FIX-06 — Fix layer count error in Table 3.1 (pruned_5 shows wrong layer count)

**Location:** Section 3.1, main pruning results table, row for `pruned_5`.
**Current text in that row:**
> `pruned_5` (6 Layers)

**Replace with:**
> `pruned_5` (7 Layers)

**Explanation:** Starting from 12 layers, removing 5 layers leaves 7, not 6. Both `pruned_5` and `pruned_6` are currently labelled as "6 Layers" which is incorrect. `pruned_6` (6 Layers) is correct. `pruned_5` must be corrected to 7 Layers.

---

### FIX-07 — Add 120h and 168h columns to ALL variable tables

**Location:** Tables 3.3.1.1, 3.3.1.2, 3.3.1.3, 3.3.1.4, 3.3.2.1, 3.3.2.2, 3.3.2.3, 3.3.2.4

**Action:** Add two new columns to every table: `120h RMSE` (or `120h ACC`) and `168h RMSE` (or `168h ACC`).

**New column headers to add:**
- For RMSE tables: add `| 120h RMSE | 168h RMSE |` after the `72h RMSE` column
- For ACC tables: add `| 120h ACC | 168h ACC |` after the `72h ACC` column

**Instruction to Copilot:** The data values for 120h and 168h need to be filled in from the actual model evaluation runs. Insert placeholder text `[TO FILL]` in each new cell so the author knows where to insert the values. Example row format:

```
| `pruned_1` | 0.53 | 0.89 | 1.88 | [TO FILL] | [TO FILL] |
```

---

### FIX-08 — Add Z500 multi-lead-time table

**Location:** Section 3.1, after the main pruning table and before "Analysis of Pruning Results".
**Add a new table:**

```
**Table 3.1.2: RMSE and ACC for Z500 across all lead times**

| Model                  | 6h RMSE | 24h RMSE | 72h RMSE | 120h RMSE | 168h RMSE | 72h ACC |
| ---------------------- | :-----: | :------: | :------: | :-------: | :-------: | :-----: |
| **Backbone**           | [TO FILL] | [TO FILL] | 258.93 | [TO FILL] | [TO FILL] | 0.967 |
| `pruned_1`             | [TO FILL] | [TO FILL] | 298.90 | [TO FILL] | [TO FILL] | 0.957 |
| `pruned_2`             | [TO FILL] | [TO FILL] | 312.64 | [TO FILL] | [TO FILL] | 0.953 |
| `pruned_3`             | [TO FILL] | [TO FILL] | 362.33 | [TO FILL] | [TO FILL] | 0.937 |
| `pruned_4`             | [TO FILL] | [TO FILL] | 367.90 | [TO FILL] | [TO FILL] | 0.934 |
| `pruned_5`             | [TO FILL] | [TO FILL] | 415.97 | [TO FILL] | [TO FILL] | 0.914 |
| `pruned_6`             | [TO FILL] | [TO FILL] | 446.29 | [TO FILL] | [TO FILL] | 0.901 |
| *Persistence*          | [TO FILL] | [TO FILL] | 931.26 | [TO FILL] | [TO FILL] | 0.574 |
| *Climatology*          | [TO FILL] | [TO FILL] | 1081.46 | [TO FILL] | [TO FILL] | 0.000 |
```

---

### FIX-09 — Fill in missing model size column in FP16 table

**Location:** Section 3.2, FP16 combined results table. All rows currently showing `-` in the Model Size column.

**Action:** Replace each `-` with the correct model size in MB.
The formula is: FP16 size = FP32 parameter count × 2 bytes / 1,000,000.

Each pruned model removes approximately 5M parameters per layer from the ~74.7M backbone.
Use this instruction for Copilot:

```
Replace the `-` values in the Model Size column with the following:
- pruned_1-FP16:  ~34.4M params → ~68.8 MB
- pruned_2-FP16:  ~31.4M params → ~62.8 MB
- pruned_3-FP16:  ~28.4M params → ~56.8 MB
- pruned_4-FP16:  ~25.4M params → ~50.8 MB
- pruned_5-FP16:  ~22.4M params → ~44.8 MB
- pruned_6-FP16:  ~19.4M params → ~38.8 MB
```

Note: Verify these numbers against actual model parameter counts before final submission.

---

### FIX-10 — Revise "star performer" framing for pruned_3 due to T2M anomaly

**Location:** Section 3.4, opening sentence and heading.
**Current text:**
> "While the tables provide precise data, visualizations can offer a more intuitive understanding of the performance. This section presents the direct comparison plots between our "star performer"—the `pruned_3` model—and the original `backbone` model..."

**Replace with:**
> "While the tables provide precise data, visualizations can offer a more intuitive understanding of the performance. This section presents the direct comparison plots between the `pruned_3` model and the original `backbone` model. The `pruned_3` model was selected as a representative mid-point in the pruning spectrum, offering a 2.65× speedup. As the analysis below will show, its performance is consistent with the backbone for most variables but exhibits anomalous degradation for surface temperature (`t2m`) at longer lead times, which is discussed in detail in Section 4.2."

---

### FIX-11 — Add explicit acknowledgment of T2M non-monotonic behavior as a known issue

**Location:** Section 3.3.1, at the end of point 4 (Anomalous t2m Behavior).
**Current text:**
> "This could be due to the complex interactions between the pruned model's architecture and the fine-tuning process. It highlights that the relationship between depth and accuracy is not always perfectly linear and can vary depending on the specific variable and lead time."

**Replace with:**
> "This non-monotonic behavior — where `pruned_3` (9 layers, ACC 0.551) performs significantly worse than both `pruned_2` (ACC 0.868) and `pruned_4` (ACC 0.916) at 72h — is an anomaly that warrants careful interpretation. The most likely explanation is that the fine-tuning for `pruned_3` converged to a suboptimal local minimum for surface temperature prediction, a known risk when fine-tuning deep networks from a pruned checkpoint rather than training from scratch. This result should not be interpreted as a general property of 9-layer models, but rather as a specific instability in this training run. Future work should re-run fine-tuning for `pruned_3` with multiple random seeds to verify whether this behavior is reproducible or an isolated convergence failure. For applications where `t2m` accuracy is critical, `pruned_2` is recommended as a more reliable alternative."

---

### FIX-12 — Revise "graceful degradation" claim in Chapter Summary to be qualified

**Location:** Section 3.5, bullet point 1.
**Current text:**
> "Structured pruning by layer removal provides a direct and predictable way to increase inference speed by up to ~4x. This comes at the cost of a graceful, non-catastrophic degradation in forecast accuracy."

**Replace with:**
> "Structured pruning by layer removal provides a direct and predictable way to increase inference speed by up to ~4x. For free-atmosphere variables (`z500`, `t850`) and surface wind components (`u10`, `v10`), this comes at the cost of a graceful, non-catastrophic degradation in forecast accuracy. Surface temperature (`t2m`) shows more volatile degradation at longer lead times (≥72h), with performance sensitivity that is discussed in detail in Section 3.3.1."

---

## FILE: 04_Discussion.md

---

### FIX-13 — Strengthen T2M discussion with specific numbers

**Location:** Section 4.2, paragraph 2.
**Current text:**
> "We observed that the forecast skill for `t2m`, `u10`, and `v10` was more sensitive to layer removal than for `z500` and `t850`, especially at longer lead times."

**Replace with:**
> "We observed that the forecast skill for `t2m`, `u10`, and `v10` was more sensitive to layer removal than for `z500` and `t850`, especially at longer lead times. For instance, at 72h, the `pruned_6` model retains a `z500` ACC of 0.901 (a 6.8% drop from the backbone's 0.967), while the same model achieves a `u10` ACC of 0.717 (a 13.4% drop). The `t2m` variable shows the most pronounced sensitivity, with `pruned_6` recording a 72h ACC of 0.686, representing a 28.7% drop. This variable-dependent sensitivity has direct practical implications: for applications requiring accurate surface temperature forecasts at medium range, aggressive pruning beyond 2–3 layers is not recommended."

---

### FIX-14 — Revise the limitations section to include the T2M anomaly as a limitation

**Location:** Section 4.4 (Limitations and Future Work), after the first bullet point (Pruning Strategy).
**Add a new bullet point:**

> - **Fine-Tuning Stability:** The non-monotonic performance of `pruned_3` for `t2m` reveals a limitation of our single-run fine-tuning protocol. Each pruned model was fine-tuned once from a fixed initialization. Without multiple training runs, it is not possible to distinguish between a structural property of that architecture and a one-time convergence failure. Future work should conduct at least 3 independent fine-tuning runs per pruned variant and report mean and standard deviation of forecast skill metrics.

---

## FILE: 05_Conclusion.md

---

### FIX-15 — Qualify the graceful degradation claim in the conclusion

**Location:** Section 5.1, bullet point 1.
**Current text:**
> "We have shown that structured pruning by layer removal is a highly effective and predictable method for reducing model latency. By progressively removing layers from a 12-layer AFNO-based `backbone` model, we achieved speedups of up to **3.89x**. This performance gain came at the cost of a graceful, non-catastrophic degradation in forecast skill..."

**Replace with:**
> "We have shown that structured pruning by layer removal is a highly effective and predictable method for reducing model latency. By progressively removing layers from a 12-layer AFNO-based `backbone` model, we achieved speedups of up to **3.89x**. For the majority of evaluated variables — including `z500`, `t850`, `u10`, and `v10` — this performance gain came at the cost of a graceful, non-catastrophic degradation in forecast skill. Surface temperature (`t2m`) was found to be more sensitive to layer removal at longer lead times and exhibited instability in the fine-tuning process for certain pruning levels, which is identified as a direction for future investigation."

---

## NEW FILE: 02b_Related_Work.md

### FIX-16 — Create a Related Work section (currently missing entirely)

**Action:** Create a new file `02b_Related_Work.md` with the following structure.
Insert this chapter between Introduction and Methodology in the final paper.

```markdown
# 2. Related Work

## 2.1. Numerical Weather Prediction and its Limitations

Traditional Numerical Weather Prediction (NWP) models such as ECMWF's Integrated Forecasting System (IFS) solve the governing equations of atmospheric dynamics on a global grid. While highly accurate, these systems require significant supercomputing resources, making them inaccessible for many applications [CITE: Bauer et al. 2015, Nature].

## 2.2. Data-Driven Weather Forecasting

The application of deep learning to weather forecasting has grown rapidly. Early work using convolutional neural networks demonstrated the feasibility of learned atmospheric emulators [CITE: Weyn et al. 2020]. The introduction of Vision Transformer architectures to the domain led to a step change in performance. FourCastNet [CITE: Pathak et al. 2022] introduced the Adaptive Fourier Neural Operator (AFNO) for global weather prediction, achieving competitive accuracy at a fraction of the computational cost of NWP. Subsequent models including Pangu-Weather [CITE: Bi et al. 2023] and GraphCast [CITE: Lam et al. 2023] further advanced the state of the art using 3D transformer and graph neural network architectures respectively. WeatherBench2 [CITE: Rasp et al. 2024] established standardized benchmarks for evaluating these models, including the latitude-weighted RMSE and ACC metrics used in this work.

## 2.3. Model Compression Techniques

Model compression is a well-established area of research in deep learning [CITE: Cheng et al. 2017]. The principal techniques are:

**Pruning** involves removing parameters or structural components that contribute minimally to the model's output [CITE: Han et al. 2015]. Structured pruning — removing entire neurons, channels, or layers — is particularly effective for achieving hardware-level speedups on standard GPUs, as it preserves dense matrix operations [CITE: Li et al. 2017].

**Quantization** reduces the numerical precision of model weights [CITE: Jacob et al. 2018]. Post-training quantization to FP16 is widely used in practice and has been shown to provide substantial inference speedups on GPUs with Tensor Core support, with negligible impact on model accuracy for most large-scale tasks [CITE: Micikevicius et al. 2018].

**Knowledge Distillation** trains a smaller student model to mimic the outputs of a larger teacher model [CITE: Hinton et al. 2015], enabling compact models to achieve accuracy approaching that of much larger networks. We do not explore distillation in this work but identify it as a direction for future research.

## 2.4. Compression of Weather and Climate Models

Compression of AI-based weather models is a nascent but growing research area. [TO FILL: Add any directly related prior work on compressing FourCastNet, Pangu, or similar models if found in literature search]. To the best of our knowledge, this work represents one of the first systematic studies of structured pruning combined with quantization applied specifically to AFNO-based global weather forecasting models, evaluated against standard meteorological baselines across multiple variables and lead times.
```

---

## NEW FILE: references.md

### FIX-17 — Create a References section (currently missing entirely)

**Action:** Create a new file `references.md` with the following entries.
Add all references cited in the paper here in ACM format.

```markdown
# References

[1] Peter Bauer, Alan Thorpe, and Gilbert Brunet. 2015. The quiet revolution of numerical weather prediction. *Nature* 525 (2015), 47–55.

[2] Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. 2023. Accurate medium-range global weather forecasting with 3D neural networks. *Nature* 619 (2023), 533–538.

[3] Yu Cheng, Duo Wang, Pan Zhou, and Tao Zhang. 2017. A survey of model compression and acceleration for deep neural networks. *IEEE Signal Processing Magazine* 35, 1 (2017), 126–136.

[4] Song Han, Jeff Pool, John Tran, and William Dally. 2015. Learning both weights and connections for efficient neural networks. In *Advances in Neural Information Processing Systems (NeurIPS)*.

[5] Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. In *NeurIPS Deep Learning Workshop*.

[6] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, Matthew Tang, Andrew Howard, Hartwig Adam, and Dmitry Kalenichenko. 2018. Quantization and training of neural networks for efficient integer-arithmetic-only inference. In *CVPR*.

[7] Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, et al. 2023. Learning skillful medium-range global weather forecasting. *Science* 382, 6677 (2023), 1416–1421.

[8] Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet, and Hans Peter Graf. 2017. Pruning filters for efficient convnets. In *ICLR*.

[9] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2018. Mixed precision training. In *ICLR*.

[10] Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, Pedram Hassanzadeh, Karthik Kashinath, and Animashree Anandkumar. 2022. FourCastNet: A global data-driven high-resolution weather model using adaptive Fourier neural operators. *arXiv preprint arXiv:2202.11214*.

[11] Stephan Rasp, Stephan Hoyer, Alexander Merose, Ian Langmore, Peter Bauer, Tsuhan Chen, Kieran Hunt, Jonathan Jumper, Alex Kochkov, Leonard Sagesser, et al. 2024. WeatherBench 2: A benchmark for the next generation of data-driven global weather models. *Journal of Advances in Modeling Earth Systems* 16, 6 (2024).

[12] Jonathan Weyn, Dale Durran, and Rich Caruana. 2020. Improving data-driven global weather prediction using deep convolutional neural networks on a cubed sphere. *Journal of Advances in Modeling Earth Systems* 12, 9 (2020).
```

---

## NEW FILE: acm_formatting_checklist.md

### FIX-18 — ACM Format Requirements Checklist

**Action:** Create this file as a checklist for final ACM formatting.
Before converting to LaTeX or ACM Word template, verify all items below are present.

```markdown
# ACM Format Checklist

## Required Front Matter
- [ ] Title
- [ ] Author names, affiliations, emails
- [ ] Abstract (currently present in 00_Abstract.md — max 150 words for ACM short, 250 for full paper)
- [ ] CCS Concepts — ADD THIS: e.g. "Computing methodologies → Neural networks; Applied computing → Earth and atmospheric sciences"
- [ ] Keywords — ADD THIS: "Model Compression, Weather Forecasting, Pruning, Quantization, FourCastNet, AFNO, ERA5"
- [ ] ACM Reference Format line

## Required Sections (in order)
- [ ] 1. Introduction ✅
- [ ] 2. Related Work ❌ MISSING — see FIX-16
- [ ] 3. Methodology ✅ (after fixes FIX-01 through FIX-05)
- [ ] 4. Results ✅ (after fixes FIX-06 through FIX-12)
- [ ] 5. Discussion ✅ (after fixes FIX-13 through FIX-14)
- [ ] 6. Conclusion ✅ (after FIX-15)
- [ ] Acknowledgments ✅ (present in original)
- [ ] References ❌ MISSING — see FIX-17

## Figure Requirements
- [ ] All figures must have captions below (not above)
- [ ] Figure captions must start with "Figure N:"
- [ ] All graphs must include persistence and climatology baseline lines — currently missing from all 5 graphs
- [ ] All graphs must show all pruned models (pruned_1 through pruned_6) — currently only pruned_3 shown
- [ ] Add ACC vs Lead Time plots for each variable — currently missing entirely
- [ ] Minimum required figures:
    - [ ] Fig 1: Similarity matrix heatmap (already in original)
    - [ ] Fig 2: RMSE vs Lead Time — Z500 (all models + baselines)
    - [ ] Fig 3: RMSE vs Lead Time — T850 (all models + baselines)
    - [ ] Fig 4: RMSE vs Lead Time — T2M (all models + baselines)
    - [ ] Fig 5: RMSE vs Lead Time — U10 (all models + baselines)
    - [ ] Fig 6: RMSE vs Lead Time — V10 (all models + baselines)
    - [ ] Fig 7: Accuracy vs Latency scatter plot (trade-off figure)

## Table Requirements
- [ ] All table captions must appear ABOVE the table in ACM format
- [ ] Tables must be numbered sequentially (Table 1, Table 2, ...)
- [ ] Add units to all RMSE column headers:
    - Z500 RMSE: m²/s²
    - T850 RMSE: K
    - T2M RMSE: K
    - U10 RMSE: m/s
    - V10 RMSE: m/s

## Text Requirements
- [ ] All section cross-references must use proper ACM format: "Section~\ref{sec:method}"
- [ ] All figure references in text: "Figure~\ref{fig:rmse_z500}"
- [ ] All table references in text: "Table~\ref{tab:pruning_results}"
- [ ] Equations must be numbered: the RMSE and ACC formulas in Section 2.4
- [ ] Replace all backtick code formatting (`backbone`) with \texttt{backbone} in LaTeX
```

---

## GRAPH REGENERATION INSTRUCTIONS

### FIX-19 — Regenerate all 5 RMSE graphs with correct content

**Action:** The existing graphs (rmse_z500.png, rmse_t850.png, rmse_t2m.png, rmse_u10.png, rmse_v10.png) need to be regenerated.
Each graph currently shows only `pruned_3` vs `backbone` with no baselines.

**Required changes for every graph:**

1. Add these lines (in addition to backbone and pruned_3):
   - `pruned_1` — solid line, color: blue
   - `pruned_2` — solid line, color: green
   - `pruned_4` — solid line, color: orange
   - `pruned_5` — solid line, color: red
   - `pruned_6` — solid line, color: purple
   - `Persistence` — dashed line, color: gray, linewidth=1.5
   - `Climatology` — dotted line, color: brown, linewidth=1.5

2. Update legend to include all models

3. Update title to remove "pruned_3_layers_epoch_6 vs backbone" and replace with variable name only:
   - e.g. "RMSE vs Lead Time: Z500 (All Models, 2018 Test Year)"

4. Add horizontal reference annotations at key lead times:
   - Vertical dashed lines at x=1 (Day 1), x=3 (Day 3), x=5 (Day 5), x=7 (Day 7)

### FIX-20 — Create 5 new ACC vs Lead Time graphs

**Action:** Create matching ACC plots for each variable:
`acc_z500.png`, `acc_t850.png`, `acc_t2m.png`, `acc_u10.png`, `acc_v10.png`

Same format as RMSE graphs but:
- Y-axis: "Latitude-Weighted ACC", range [0, 1]
- Add horizontal reference line at ACC=0.6 (commonly cited minimum skillful forecast threshold)
- All models + baselines same as FIX-19

### FIX-21 — Create Accuracy vs Latency scatter plot

**Action:** Create a new file `accuracy_vs_latency.png`.

Plot specifications:
- X-axis: Latency (ms), range [0, 300]
- Y-axis: Z500 ACC at 72h, range [0.88, 0.98]
- Each point = one model configuration
- FP32 models: circle markers (○)
- FP16 models: diamond markers (◇)
- Label each point with model name
- Color: gradient from blue (backbone) to red (pruned_6)
- Add annotation: "← More Efficient" on x-axis
- Add annotation: "↑ More Accurate" on y-axis
- Title: "Accuracy-Efficiency Trade-off (Z500 @ 72h)"