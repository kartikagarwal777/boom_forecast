# BOOM Benchmark Predictors

Date: 2025-09-28
Author: Kartik Agarwal

## 1 Executive summary

The project implements and compares multiple forecasting approaches on the BOOM benchmark. The implemented predictors are:
- Persistence (last-value baseline)
- Seasonal Naive
- LSTM
- Temporal Fusion Transformer
- AutoARIMA (wrapper present, disabled by default due to runtime)

Preliminary results show that LSTM performs consistently well at medium frequencies, Persistence dominates at very high frequencies, and SeasonalNaive remains strong at daily/weekly resolutions, with TFT adding interpretability and has performance between seasonalnaive and LSTM most of the times.

# 2 DataSet Selection

Why Choose the BOOM Dataset

The BOOM dataset (Datadog) is an ideal choice for this task because it directly aligns with the project’s goal of forecasting cloud and Kubernetes infrastructure metrics. It provides a realistic, high-fidelity collection of operational data such as CPU utilization, memory consumption, latency, and request throughput. These are the same categories of metrics that real-world cloud platforms monitor for performance, scalability, and reliability, ensuring the prototype addresses practical challenges rather than artificial benchmarks

## Realism and Relevance
The dataset originates from real cloud/Kubernetes environments, which makes it a better reflection of real operational scenarios compared to synthetic datasets. This realism ensures that any forecasting model built will have direct applicability in production-like settings such as capacity planning, anomaly detection, and latency management.

## Coverage and Diversity of Metrics
BOOM captures multiple dimensions of infrastructure performance (CPU, memory, latency, etc.) over time. This diversity is crucial for building models that can handle multi-metric forecasting and learn the interplay between different resources—important for tasks like predicting bottlenecks or cascading failures in distributed systems.

## Benchmark Recognition
Since BOOM is a curated dataset shared on Hugging Face, it is well-documented, standardized, and widely recognized in the research community. This makes it easier to reproduce results, compare against other methods, and ensures transparency in methodology.

## Suitability for Time-Series Forecasting
The dataset has the temporal granularity and scale required for short- and long-term predictions. Its structure allows for meaningful train/validation/test splits and supports feature engineering (e.g., lag features, moving averages) without risk of future leakage.

## Practical Utility in Cloud Operations
By working with BOOM, the prototype demonstrates clear potential applications in cloud workload forecasting, autoscaling policies, and proactive system reliability management—all of which are critical in modern DevOps and SRE practices.

Among the whole BOOM dataset, we choose a subset of 18 datasets to be analyzed instead of the whole universe because of compute limitations. 3 datasets emcompassing all different frequencies are choosen. 

# 3 Design and implementation

High level
- The top-level entry is `main.py`, which calls `models.compare.run_compare`. The compare runner iterates datasets/terms, prepares training series, evaluates models via GluonTS evaluation utilities, and writes results and plots.

Data handling
- Datasets are loaded using `gift_eval.data.Dataset` (BOOM benchmark wrapper).
- `run_compare` robustly extracts the training series from dataset entries (handles dict, list/tuple, and array-like variations used in the BOOM data loader).

## Persistence
Implemented in models/persistence_predictor.py, this baseline simply projects the last observed value forward across all forecast horizons. While intentionally simplistic, it serves as a critical benchmark: any more sophisticated model must demonstrate performance gains over Persistence to justify its added complexity. Its parsimony ensures a clear lower bound on predictive capability.

## SeasonalNaive
Designed as a lightweight yet effective seasonal baseline, this model wraps statsforecast-style logic to produce GluonTS-compatible forecast objects. Seasonality is automatically inferred from Dataset.freq, enabling it to capture recurring patterns in the data with minimal overhead. This provides a robust point of comparison for evaluating seasonality-aware deep learning models.

## Long Short-Term Memory (LSTM)
Implemented in models/lstm_predictor.py as a standard PyTorch-based architecture, the LSTM model is tailored for sequential learning and temporal dependencies.

- Normalization: Each series undergoes per-series z-score normalization during window generation, enhancing training stability and comparability across diverse time series.

- Strengths: Effective at modeling medium- to long-term dependencies, LSTMs provide a solid deep learning baseline for sequence forecasting.

## Temporal Fusion Transformer (TFT)
Leveraging the pytorch-forecasting library’s implementation of the Temporal Fusion Transformer, trained with lightning.pytorch.Trainer, this model represents a state-of-the-art deep learning approach to time-series forecasting.

- Interpretability: Through attention mechanisms and variable selection, TFT provides insights into feature importance and how seasonality and trends are captured.

- Multi-horizon Forecasting: Its architecture is explicitly designed for robust multi-period predictions, handling both short- and long-term horizons with high flexibility.

- Advantage: TFT balances predictive power with interpretability, making it particularly well-suited for production-oriented forecasting tasks in operational environments.

## AutoARIMA
A classical statistical approach implemented as a wrapper in a pmdarima/statsforecast style. Although the evaluation call is commented out in models/compare.py—due to the computational intensity of running AutoARIMA across large collections of series—the implementation is preserved for smaller-scale benchmarking and for highlighting contrasts between traditional and deep learning methods.

# 4 Evaluation protocol and metrics

Evaluation method
- Models are evaluated using `gluonts.model.evaluate_model`, which returns standard GluonTS metric frames. Evaluation uses test sets provided by the BOOM `Dataset` object.

Metrics recorded
- MSE (mean, 0.5), MAE, MASE, MAPE, sMAPE, MSIS, RMSE, NRMSE, ND, MeanWeightedSumQuantileLoss (quantiles 0.1..0.9)

Output format
- Results are serialized incrementally to `output/compare/all_results.csv` with one row per dataset/term/model containing the selected metrics and dataset metadata.

Visualization
- For each dataset and model, plots are generated and stored under `output/plots/<dataset>/<term>/<model>/`. These plots include forecast vs. actual visualizations and summary metric plots. The plotting utilities can be extended to export per-series CSVs of predictions and timestamps.

# 5 Insights

## High-frequency data

- Persistence is surprisingly strong here, with much lower RMSE and MAE than deep models.

- LSTM and TFT perform worse on raw error metrics This suggests high-frequency data may be dominated by noise and short-term volatility, making simple baselines competitive.

Insight: For ultra-granular telemetry, simple methods may outperform complex architectures due to overfitting and noise sensitivity.

## Medium-frequency data

- LSTM generally shows stable improvements over baselines, with lower MAE.

- TFT sometimes struggles compared to LSTM, likely due to its higher complexity needing more data per frequency.

Insight: At medium granularities, deep learning starts to capture meaningful temporal dependencies, offering gains over seasonal baselines.

## Hourly and daily data

- SeasonalNaive often becomes very competitive (or even dominant), since clear daily/weekly patterns emerge.

- TFT shows advantages in multi-horizon tasks where interpretability of seasonality matters.

Insight: For coarser-grained metrics, seasonality-aware baselines are very effective, but transformers provide interpretability and robustness when multi-metric forecasting is needed.

# 6 Final Suggestions

Persistence: Best suited for volatile high-frequency series (10s-level).

SeasonalNaive: Strong on structured periodic data (daily/hourly).

LSTM: Balanced performer across medium frequencies, handling temporal dependencies effectively.

TFT: High potential for interpretability and multi-horizon prediction, but requires richer data and benefits more from longer horizons than short noisy series. But still performs decent at H/D frequency

Overall, this layered evaluation across frequencies highlights that no single model is universally best; instead, model choice should align with the temporal resolution and operational requirements of the forecasting task.