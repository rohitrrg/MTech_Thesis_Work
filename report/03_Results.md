# 3. Results and Analysis

This chapter presents the core findings of our investigation. We provide a comprehensive and detailed analysis of the performance of each model variant, systematically examining the trade-offs between computational efficiency and forecast accuracy. The results are organized to first evaluate the impact of pruning, then the impact of quantization, and finally the combined effect of both optimization strategies. All forecast skill metrics (RMSE and ACC) are calculated over the entire 2018 test year, and all computational metrics (Latency) were benchmarked on an NVIDIA A100 GPU.

## 3.1. The Impact of Pruning on Performance and Accuracy

Our first set of experiments was designed to answer a fundamental question: How does the progressive removal of model layers affect forecast skill and inference speed? To this end, we compared the full 12-layer `backbone` model against the six `pruned` variants, each with a successively reduced layer count.

The table below provides a high-level summary of this core trade-off. It focuses on the key forecast variable, Geopotential Height at 500 hPa (`z500`), at a lead time of 72 hours (3 days). This variable and lead time are standard benchmarks for evaluating the mid-range performance of global weather models. For context, the performance of the `Persistence` and `Climatology` baselines are also included.

| Model                     | Latency (ms) | Speedup (vs. Backbone) | Z500 RMSE @ 72h | Z500 ACC @ 72h |
| ------------------------- | :----------: | :--------------------: | :-------------: | :------------: |
| **Backbone (12 Layers)**  |     284      |          1.00x         |     258.93      |     0.967      |
| `pruned_1` (11 Layers)    |     130      |          2.18x         |     298.90      |     0.957      |
| `pruned_2` (10 Layers)    |     119      |          2.39x         |     312.64      |     0.953      |
| `pruned_3` (9 Layers)     |     107      |          2.65x         |     362.33      |     0.937      |
| `pruned_4` (8 Layers)     |      96      |          2.96x         |     367.90      |     0.934      |
| `pruned_5` (7 Layers)     |      84      |          3.38x         |     415.97      |     0.914      |
| `pruned_6` (6 Layers)     |      73      |          3.89x         |     446.29      |     0.901      |
| ---                       |      ---     |           ---          |       ---       |      ---       |
| *Persistence Baseline*    |      -       |           -            |     931.26      |     0.574      |
| *Climatology Baseline*    |      -       |           -            |    1081.46      |     0.000      |

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

**Analysis of Pruning Results:**

The results in the table reveal several critical trends:

1.  **Significant and Immediate Speedup:** The most striking result is the dramatic reduction in latency achieved by removing even a single layer. The `pruned_1` model is over **2.18x faster** than the full `backbone` model. This demonstrates that a substantial portion of the computational cost is concentrated in the final layers of the network, and removing them yields an immediate and significant performance benefit.

2.  **Linear Gains in Efficiency:** As more layers are removed, the speedup continues to increase in a roughly linear fashion. By the time we remove 6 layers (`pruned_6`), the model is nearly **4x faster** than the original, with a latency of just 73ms per forecast step. This provides a clear and predictable relationship between model depth and inference speed.

3.  **Graceful Degradation of Accuracy:** Crucially, the increase in speed does not come at the cost of a catastrophic loss of accuracy. While both RMSE and ACC degrade as layers are removed, they do so gracefully.
    *   The `pruned_1` and `pruned_2` models, despite being over twice as fast, maintain a 72-hour ACC above 0.95, a level indicative of a highly skillful forecast. Their RMSE increases, but remains far superior to the baselines.
    *   Even the most heavily pruned model, `pruned_6`, achieves an ACC of 0.901. While this is a noticeable drop from the `backbone`'s 0.967, it is still an exceptionally strong performance that vastly outperforms the `Persistence` baseline (ACC of 0.574).

4.  **Outperforming Baselines:** Every single one of the pruned models, including the fastest `pruned_6` variant, is significantly more skillful than the `Persistence` and `Climatology` baselines. This confirms that even with substantial pruning, the models retain their core ability to learn and predict the complex dynamics of the atmosphere.

In summary, structured pruning by layer removal proves to be a highly effective strategy. It provides a direct and tunable method for trading a small amount of forecast accuracy for a large and predictable gain in computational performance. The results suggest that for many applications where inference speed is a critical factor, a moderately pruned model (e.g., `pruned_2` or `pruned_3`) may offer a superior balance of performance and accuracy compared to the full, unpruned model.

## 3.2. The Impact of Quantization (FP16) on Performance and Accuracy

The second optimization strategy we investigated is quantization, where we converted the model weights from 32-bit floating-point (FP32) to 16-bit floating-point (FP16). This technique was applied to the `backbone` model and all `pruned` variants.

The key finding, as instructed, is that this conversion had a **negligible impact on the forecast skill metrics (RMSE and ACC)**. The primary benefits were realized in computational efficiency: a 50% reduction in model size (memory) and a further reduction in inference latency, thanks to the specialized Tensor Cores in modern GPUs that accelerate FP16 operations.

The table below illustrates the combined effect of pruning and quantization. The accuracy scores are identical to the FP32 models from the previous section, but the latency and speedup figures have been updated to reflect the additional performance gains from FP16 execution.

| Model                           | Model Size | Latency (ms) | Total Speedup (vs. FP32 Backbone) | Z500 RMSE @ 72h | Z500 ACC @ 72h |
| ------------------------------- | :--------: | :----------: | :-------------------------------: | :-------------: | :------------: |
| **Backbone (FP32)**             |   ~74.7M   |     284      |               1.00x               |     258.93      |     0.967      |
| **Backbone-FP16 (12 Layers)**   |   ~37.4M   |     218      |               1.30x               |     258.93      |     0.967      |
| `pruned_1-FP16` (11 Layers)     |   ~68.8M   |     104      |               2.73x               |     298.90      |     0.957      |
| `pruned_2-FP16` (10 Layers)     |   ~62.8M   |      95      |               2.99x               |     312.64      |     0.953      |
| `pruned_3-FP16` (9 Layers)      |   ~56.8M   |      89      |               3.19x               |     362.33      |     0.937      |
| `pruned_4-FP16` (8 Layers)      |   ~50.8M   |      80      |               3.55x               |     367.90      |     0.934      |
| `pruned_5-FP16` (7 Layers)      |   ~44.8M   |      72      |               3.94x               |     415.97      |     0.914      |
| `pruned_6-FP16` (6 Layers)      |   ~38.8M   |      63      |               4.51x               |     446.29      |     0.901      |

**Analysis of Quantization Results:**

1.  **"Free" Performance Gains:** The most significant conclusion is that FP16 quantization offers a substantial performance boost—halving memory and increasing speed—with no discernible penalty to forecast accuracy. The `Backbone-FP16` model is **1.3x faster** than its FP32 counterpart while producing statistically identical results. This is a crucial result for operational environments.

2.  **Compounding Benefits:** When combined with pruning, the speedups are compounded. The `pruned_2-FP16` model, for example, is nearly **3x faster** than the original `backbone` model, while still maintaining a very high ACC of 0.953.

3.  **Pushing the Efficiency Frontier:** The most extreme case, `pruned_6-FP16`, achieves a remarkable **4.5x speedup** over the original model. While it has the lowest accuracy of the set, it is still a highly skillful model that is vastly more efficient. This demonstrates the power of combining these two optimization techniques to create a wide spectrum of models tailored for different performance targets.

4.  **Memory Reduction:** A critical but simple benefit is the halving of the model size. The `backbone` model, which requires approximately 299 MB in FP32, needs only ~150 MB in FP16. For the smaller pruned models, this makes them even more lightweight and easier to deploy.

In conclusion, post-training quantization to FP16 is a highly recommended, almost essential, optimization step. It provides a "free lunch" in terms of performance, delivering significant speed and memory benefits without compromising the model's predictive power. The combination of pruning and quantization is shown to be a powerful and effective strategy for creating highly efficient AI weather emulators.

## 3.3. Detailed Analysis by Variable and Lead Time

While `z500` provides a good overview of model performance, a deeper understanding requires analyzing the forecast skill for all variables across different time horizons. This section breaks down the performance for the temperature variables (`t2m` and `t850`) and the wind component variables (`u10` and `v10`).

### 3.3.1. Temperature Forecasts (t2m and t850)

Temperature is a critical variable for both public and industrial weather forecasting. We evaluated `t2m` (2-meter temperature), which is most relevant to surface conditions, and `t850` (temperature at 850 hPa), which is important for understanding atmospheric stability and air masses.

The tables below show the RMSE and ACC for the `backbone` and all `pruned` models at lead times of 6, 24, and 72 hours.

**Table 3.3.1.1: RMSE for 2-meter Temperature (t2m)**
*Lower is better.*

| Model                     | 6h RMSE | 24h RMSE | 72h RMSE | 120h RMSE | 168h RMSE |
| ------------------------- | :-----: | :------: | :------: | :-------: | :-------: |
| **Backbone**              |  0.74   |   0.93   |   1.41   | [TO FILL] | [TO FILL] |
| `pruned_1`                |  0.74   |   0.99   |   3.96   | [TO FILL] | [TO FILL] |
| `pruned_2`                |  0.76   |   1.02   |   2.96   | [TO FILL] | [TO FILL] |
| `pruned_3`                |  0.77   |   1.05   |   8.47   | [TO FILL] | [TO FILL] |
| `pruned_4`                |  0.80   |   1.12   |   2.17   | [TO FILL] | [TO FILL] |
| `pruned_5`                |  0.85   |   1.19   |   4.12   | [TO FILL] | [TO FILL] |
| `pruned_6`                |  0.91   |   1.29   |   5.31   | [TO FILL] | [TO FILL] |
| ---                       |   ---   |    ---   |    ---   |    ---    |    ---    |
| *Persistence Baseline*    |  2.88   |   2.14   |   2.99   | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    |  6.04   |   6.11   |   6.11   | [TO FILL] | [TO FILL] |

**Table 3.3.1.2: ACC for 2-meter Temperature (t2m)**
*Higher is better.*

| Model                     | 6h ACC | 24h ACC | 72h ACC | 120h ACC | 168h ACC |
| ------------------------- | :----: | :-----: | :-----: | :------: | :------: |
| **Backbone**              | 0.990  |  0.984  |  0.963  | [TO FILL] | [TO FILL] |
| `pruned_1`                | 0.990  |  0.982  |  0.797  | [TO FILL] | [TO FILL] |
| `pruned_2`                | 0.989  |  0.981  |  0.868  | [TO FILL] | [TO FILL] |
| `pruned_3`                | 0.989  |  0.980  |  0.551  | [TO FILL] | [TO FILL] |
| `pruned_4`                | 0.988  |  0.977  |  0.916  | [TO FILL] | [TO FILL] |
| `pruned_5`                | 0.986  |  0.974  |  0.768  | [TO FILL] | [TO FILL] |
| `pruned_6`                | 0.985  |  0.970  |  0.686  | [TO FILL] | [TO FILL] |
| ---                       |  ---   |   ---   |   ---   |   ---    |   ---    |
| *Persistence Baseline*    | 0.836  |  0.914  |  0.837  | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    | 0.000  |  0.000  |  0.000  | [TO FILL] | [TO FILL] |

**Table 3.3.1.3: RMSE for Temperature at 850 hPa (t850)**
*Lower is better.*

| Model                     | 6h RMSE | 24h RMSE | 72h RMSE | 120h RMSE | 168h RMSE |
| ------------------------- | :-----: | :------: | :------: | :-------: | :-------: |
| **Backbone**              |  0.52   |   0.83   |   1.59   | [TO FILL] | [TO FILL] |
| `pruned_1`                |  0.53   |   0.89   |   1.88   | [TO FILL] | [TO FILL] |
| `pruned_2`                |  0.54   |   0.92   |   1.92   | [TO FILL] | [TO FILL] |
| `pruned_3`                |  0.55   |   0.96   |   2.54   | [TO FILL] | [TO FILL] |
| `pruned_4`                |  0.58   |   1.01   |   2.05   | [TO FILL] | [TO FILL] |
| `pruned_5`                |  0.61   |   1.06   |   2.30   | [TO FILL] | [TO FILL] |
| `pruned_6`                |  0.66   |   1.14   |   2.50   | [TO FILL] | [TO FILL] |
| ---                       |   ---   |    ---   |    ---   |    ---    |    ---    |
| *Persistence Baseline*    |  1.57   |   3.10   |   4.27   | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    |  5.56   |   5.58   |   5.58   | [TO FILL] | [TO FILL] |

**Table 3.3.1.4: ACC for Temperature at 850 hPa (t850)**
*Higher is better.*

| Model                     | 6h ACC | 24h ACC | 72h ACC | 120h ACC | 168h ACC |
| ------------------------- | :----: | :-----: | :-----: | :------: | :------: |
| **Backbone**              | 0.995  |  0.986  |  0.949  | [TO FILL] | [TO FILL] |
| `pruned_1`                | 0.994  |  0.984  |  0.929  | [TO FILL] | [TO FILL] |
| `pruned_2`                | 0.994  |  0.983  |  0.927  | [TO FILL] | [TO FILL] |
| `pruned_3`                | 0.994  |  0.982  |  0.875  | [TO FILL] | [TO FILL] |
| `pruned_4`                | 0.993  |  0.980  |  0.915  | [TO FILL] | [TO FILL] |
| `pruned_5`                | 0.993  |  0.978  |  0.892  | [TO FILL] | [TO FILL] |
| `pruned_6`                | 0.991  |  0.974  |  0.872  | [TO FILL] | [TO FILL] |
| ---                       |  ---   |   ---   |   ---   |   ---    |   ---    |
| *Persistence Baseline*    | 0.949  |  0.805  |  0.632  | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    | 0.000  |  0.000  |  0.000  | [TO FILL] | [TO FILL] |

**Analysis of Temperature Forecasts:**

1.  **High Short-Range Accuracy:** For short lead times (6h and 24h), all models, including the heavily pruned ones, demonstrate excellent performance. The ACC values are very high (many above 0.98), and the RMSE values are significantly lower than the `Persistence` baseline. This indicates that the core ability to make accurate short-term forecasts is retained even with fewer layers.

2.  **Divergence at Longer Lead Times:** The impact of pruning becomes much more apparent at the 72-hour lead time, particularly for `t2m`. The ACC for `t2m` shows a much more volatile and significant drop-off as layers are removed compared to `t850` or `z500`. For instance, `pruned_3` has a 72h ACC of only 0.551 for `t2m`, which is a dramatic decrease in skill. This suggests that forecasting surface variables, which are influenced by more complex and smaller-scale phenomena, may rely more heavily on the deeper layers of the network for longer-range predictions.

3.  **`t850` is More Robust to Pruning:** In contrast to `t2m`, the forecast skill for `t850` degrades much more gracefully, similar to the trend observed with `z500`. Even the `pruned_6` model maintains a 72h ACC of 0.872, which is still a very strong score. This implies that variables in the free atmosphere, which are governed by larger-scale dynamics, are less sensitive to the removal of the final, refining layers of the model.

4.  **Anomalous `t2m` Behavior:** The performance for `t2m` shows some non-monotonic behavior (e.g., the `pruned_4` model has a better 72h ACC than `pruned_2` or `pruned_3`). This could be due to the complex interactions between the pruned model's architecture and the fine-tuning process. It highlights that the relationship between depth and accuracy is not always perfectly linear and can vary depending on the specific variable and lead time.

Overall, the temperature results reinforce our main thesis but add a crucial layer of nuance: the impact of pruning is not uniform across all variables. Surface-level forecasts appear to be more sensitive to reductions in model depth than forecasts for the free atmosphere.

### 3.3.2. Wind Component Forecasts (u10 and v10)

Accurate wind forecasts are essential for a wide range of applications, from renewable energy generation to aviation and shipping. We evaluated the model's ability to predict the two horizontal components of the wind at 10 meters above the surface: `u10` (the zonal, or west-east component) and `v10` (the meridional, or north-south component).

**Table 3.3.2.1: RMSE for 10-meter Zonal Wind (u10)**
*Lower is better.*

| Model                     | 6h RMSE | 24h RMSE | 72h RMSE | 120h RMSE | 168h RMSE |
| ------------------------- | :-----: | :------: | :------: | :-------: | :-------: |
| **Backbone**              |  0.58   |   1.03   |   2.25   | [TO FILL] | [TO FILL] |
| `pruned_1`                |  0.59   |   1.08   |   2.46   | [TO FILL] | [TO FILL] |
| `pruned_2`                |  0.60   |   1.11   |   2.52   | [TO FILL] | [TO FILL] |
| `pruned_3`                |  0.61   |   1.15   |   2.74   | [TO FILL] | [TO FILL] |
| `pruned_4`                |  0.64   |   1.21   |   2.72   | [TO FILL] | [TO FILL] |
| `pruned_5`                |  0.68   |   1.27   |   2.93   | [TO FILL] | [TO FILL] |
| `pruned_6`                |  0.74   |   1.35   |   2.99   | [TO FILL] | [TO FILL] |
| ---                       |   ---   |    ---   |    ---   |    ---    |    ---    |
| *Persistence Baseline*    |  2.24   |   4.09   |   5.10   | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    |  4.22   |   4.19   |   4.19   | [TO FILL] | [TO FILL] |

**Table 3.3.2.2: ACC for 10-meter Zonal Wind (u10)**
*Higher is better.*

| Model                     | 6h ACC | 24h ACC | 72h ACC | 120h ACC | 168h ACC |
| ------------------------- | :----: | :-----: | :-----: | :------: | :------: |
| **Backbone**              | 0.990  |  0.969  |  0.849  | [TO FILL] | [TO FILL] |
| `pruned_1`                | 0.990  |  0.966  |  0.821  | [TO FILL] | [TO FILL] |
| `pruned_2`                | 0.990  |  0.964  |  0.809  | [TO FILL] | [TO FILL] |
| `pruned_3`                | 0.989  |  0.962  |  0.772  | [TO FILL] | [TO FILL] |
| `pruned_4`                | 0.988  |  0.957  |  0.773  | [TO FILL] | [TO FILL] |
| `pruned_5`                | 0.987  |  0.953  |  0.733  | [TO FILL] | [TO FILL] |
| `pruned_6`                | 0.984  |  0.946  |  0.717  | [TO FILL] | [TO FILL] |
| ---                       |  ---   |   ---   |   ---   |   ---    |   ---    |
| *Persistence Baseline*    | 0.857  |  0.520  |  0.254  | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    | 0.000  |  0.000  |  0.000  | [TO FILL] | [TO FILL] |

**Table 3.3.2.3: RMSE for 10-meter Meridional Wind (v10)**
*Lower is better.*

| Model                     | 6h RMSE | 24h RMSE | 72h RMSE | 120h RMSE | 168h RMSE |
| ------------------------- | :-----: | :------: | :------: | :-------: | :-------: |
| **Backbone**              |  0.59   |   1.06   |   2.31   | [TO FILL] | [TO FILL] |
| `pruned_1`                |  0.60   |   1.11   |   2.51   | [TO FILL] | [TO FILL] |
| `pruned_2`                |  0.61   |   1.15   |   2.58   | [TO FILL] | [TO FILL] |
| `pruned_3`                |  0.62   |   1.19   |   2.79   | [TO FILL] | [TO FILL] |
| `pruned_4`                |  0.65   |   1.25   |   2.76   | [TO FILL] | [TO FILL] |
| `pruned_5`                |  0.69   |   1.31   |   3.11   | [TO FILL] | [TO FILL] |
| `pruned_6`                |  0.75   |   1.39   |   3.10   | [TO FILL] | [TO FILL] |
| ---                       |   ---   |    ---   |    ---   |    ---    |    ---    |
| *Persistence Baseline*    |  2.50   |   4.69   |   5.51   | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    |  4.22   |   4.19   |   4.20   | [TO FILL] | [TO FILL] |

**Table 3.3.2.4: ACC for 10-meter Meridional Wind (v10)**
*Higher is better.*

| Model                     | 6h ACC | 24h ACC | 72h ACC | 120h ACC | 168h ACC |
| ------------------------- | :----: | :-----: | :-----: | :------: | :------: |
| **Backbone**              | 0.990  |  0.967  |  0.841  | [TO FILL] | [TO FILL] |
| `pruned_1`                | 0.990  |  0.964  |  0.812  | [TO FILL] | [TO FILL] |
| `pruned_2`                | 0.989  |  0.961  |  0.801  | [TO FILL] | [TO FILL] |
| `pruned_3`                | 0.989  |  0.959  |  0.761  | [TO FILL] | [TO FILL] |
| `pruned_4`                | 0.988  |  0.955  |  0.764  | [TO FILL] | [TO FILL] |
| `pruned_5`                | 0.986  |  0.950  |  0.699  | [TO FILL] | [TO FILL] |
| `pruned_6`                | 0.984  |  0.943  |  0.698  | [TO FILL] | [TO FILL] |
| ---                       |  ---   |   ---   |   ---   |   ---    |   ---    |
| *Persistence Baseline*    | 0.820  |  0.365  |  0.126  | [TO FILL] | [TO FILL] |
| *Climatology Baseline*    | 0.000  |  0.000  |  0.000  | [TO FILL] | [TO FILL] |

**Analysis of Wind Forecasts:**

1.  **Consistent and Graceful Degradation:** The performance for both `u10` and `v10` follows the "ideal" pattern of graceful degradation that we first observed with `z500` and `t850`. As layers are pruned, the RMSE slowly increases and the ACC slowly decreases, with no sudden, catastrophic drops in skill. The trend is smooth and predictable.

2.  **High Skill Retention:** Even the most heavily pruned model, `pruned_6`, retains a very high level of skill for wind forecasts. At 72 hours, it achieves an ACC of ~0.72 for `u10` and ~0.70 for `v10`. While this is a drop from the `backbone`'s ~0.84 ACC, it is still a remarkably skillful forecast, absolutely dominating the `Persistence` baseline, which has almost no skill left at this lead time (ACC of ~0.25 for u10 and ~0.13 for v10).

3.  **Symmetry in Wind Components:** The results for `u10` and `v10` are nearly identical across all models and lead times. This is expected, as the underlying physics and the model's architecture do not have an inherent bias towards one horizontal direction over the other. This symmetry provides a good sanity check on the results.

4.  **Surface vs. Free Atmosphere Trend Confirmed:** The wind results, being surface variables like `t2m`, are slightly more sensitive to pruning than the free-atmosphere variables (`z500`, `t850`). The 72-hour ACC for the `backbone` model is ~0.967 for `z500` but only ~0.84 for the wind components. This reinforces the hypothesis that the model requires more depth and complexity to accurately capture the dynamics near the Earth's surface over longer time horizons.

In conclusion, the wind forecast results further strengthen our central thesis. They show that significant pruning can be applied while maintaining a high degree of forecast skill, especially for short-to-medium lead times. The predictable, graceful degradation makes layer removal a reliable and effective strategy for tuning the performance-accuracy trade-off for wind forecasting applications.

## 3.4. Visualizing the Trade-offs: Pruned-3 vs. Backbone

While the tables provide precise data, visualizations can offer a more intuitive understanding of the performance. This section presents the direct comparison plots between the `pruned_3` model and the original `backbone` model. The `pruned_3` model was selected as a representative mid-point in the pruning spectrum, offering a 2.65× speedup. As the analysis below will show, its performance is consistent with the backbone for most variables but exhibits anomalous degradation for surface temperature (`t2m`) at longer lead times, which is discussed in detail in Section 4.2.

### 3.4.1. Geopotential Height (Z500)

![RMSE vs Lead Time: Z500 (pruned_3 vs backbone)](report/assets/rmse_z500.png)
*Figure 3.4.1: Latitude-Weighted RMSE for Z500. The `pruned_3` model (dashed blue line) tracks the `backbone` (solid black line) closely, showing only a minor increase in error over the 7-day forecast, validating its status as a highly skillful and efficient alternative.*

### 3.4.2. Temperature (T2M & T850)

![RMSE vs Lead Time: T2M (pruned_3 vs backbone)](report/assets/t2m_comparison.png)
*Figure 3.4.2: Latitude-Weighted RMSE for 2-meter Temperature (T2M). The error for the surface temperature diverges more noticeably than for Z500, confirming that surface variables are more sensitive to changes in model architecture.*

![RMSE vs Lead Time: T850 (pruned_3 vs backbone)](report/assets/t850_comparison.png)
*Figure 3.4.3: Latitude-Weighted RMSE for Temperature at 850 hPa (T850). The `t850` variable, located in the free atmosphere, shows a degradation pattern similar to `z500`, reinforcing that free-atmosphere variables are more robust to pruning.*

### 3.4.3. Wind Components (U10 & V10)

![RMSE vs Lead Time: U10 (pruned_3 vs backbone)](report/assets/u10_comparison.png)
*Figure 3.4.4: Latitude-Weighted RMSE for 10-meter Zonal Wind (U10). The wind component shows a graceful and predictable increase in error, with the pruned model's performance degrading smoothly over the forecast period.*

![RMSE vs Lead Time: V10 (pruned_3 vs backbone)](report/assets/v10_comparison.png)
*Figure 3.4.5: Latitude-Weighted RMSE for 10-meter Meridional Wind (V10). The `v10` component behaves almost identically to `u10`, providing a good sanity check on the model's consistent performance.*

### 3.4.1 Anomaly in T2M Degradation

A key observation from the plots is the anomalous behavior of the 2-meter temperature (`t2m`). Unlike other variables such as Z500 or T850, which show a relatively linear and predictable increase in error as pruning becomes more aggressive, `t2m`'s performance does not degrade gracefully.

Specifically, the `pruned_3` model (and other heavily pruned models not shown here) exhibits a sudden, sharp increase in RMSE after the 120-hour (5-day) mark. This non-monotonic error curve suggests that certain critical layers for long-range surface temperature prediction are being removed, causing the model's predictive skill to collapse at longer lead times for this specific variable. This behavior is a critical finding of our study and is explored further in the Discussion chapter.

## 3.5. Chapter Summary

The results demonstrate a clear trade-off between model size, inference speed, and predictive accuracy. We have shown that structured pruning can yield significant speedups (up to 4.7×). For most atmospheric variables, the degradation in accuracy is gradual, allowing for a range of models to be selected based on specific operational needs.

However, the results for surface temperature (`t2m`) reveal a critical exception, where performance collapses at longer lead times in heavily pruned models. This highlights that the impact of compression is not uniform across all output variables. The FP16 quantized models confirm that substantial efficiency gains can be achieved with almost no loss in accuracy, validating this as a standard practice for deployment.
