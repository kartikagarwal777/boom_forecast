import os
import csv
import json
from typing import List

import torch
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
import numpy as np

from data_hf import download_boom_benchmark
from gift_eval.data import Dataset
from statsforecast_predictor import SeasonalNaivePredictor, AutoARIMAPredictor
from models.persistence_predictor import PersistencePredictor
from data_utils import load_boom_dataset
from models.plotting import plot_forecasts_for_dataset, plot_metrics_summary
from lstm_predictor import LSTMPredictor
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from tft_predictor import TFTPredictor

from gluonts.ev.metrics import (
    MAE,
    MAPE,
    MASE,
    MSE,
    MSIS,
    ND,
    NRMSE,
    RMSE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)

metrics = [
    MSE(forecast_type="mean"),
    MSE(forecast_type=0.5),
    MAE(),
    MASE(),
    MAPE(),
    SMAPE(),
    MSIS(),
    RMSE(),
    NRMSE(),
    ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]),
]

COMMON_QUANTILES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


def run_compare(boom_path: str, output_dir: str, datasets_to_run: List[str] = None):
    """
    Run evaluations for a selected list of BOOM datasets and save results and plots.

    Behavior:
    - If datasets_to_run is None, all datasets from boom_properties will be used.
    - Saves per-series forecast plots and per-model metric summaries under output/plots/<dataset>/<model>/
    """
    download_boom_benchmark(boom_path)
    props = json.load(open(os.path.join(boom_path, "boom_benchmark/boom_properties.json")))

    # default datasets list
    if datasets_to_run is None:
        all_datasets = list(props.keys())
    else:
        all_datasets = datasets_to_run

    # ensure output dirs
    os.makedirs(output_dir, exist_ok=True)
    plots_root = os.path.join(os.path.dirname(output_dir), "plots")
    os.makedirs(plots_root, exist_ok=True)

    csv_file_path = os.path.join(output_dir, "all_results.csv")
    with open(csv_file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "dataset","model","eval_metrics/MSE[mean]","eval_metrics/MSE[0.5]","eval_metrics/MAE[0.5]",
            "eval_metrics/MASE[0.5]","eval_metrics/MAPE[0.5]","eval_metrics/sMAPE[0.5]","eval_metrics/MSIS",
            "eval_metrics/RMSE[mean]","eval_metrics/NRMSE[mean]","eval_metrics/ND[0.5]","eval_metrics/mean_weighted_sum_quantile_loss",
            "domain","num_variates"
        ])


    for ds_name in all_datasets:
        _run_single_dataset(ds_name, props, boom_path, output_dir, plots_root, csv_file_path)


def _run_single_dataset(ds_name: str, props: dict, boom_path: str, output_dir: str, plots_root: str, csv_file_path: str):
    dataset_term = props[ds_name]["term"]
    terms = ["short","medium","long"]
    for term in terms:
        if (term in ["medium","long"]) and dataset_term == "short":
            continue
        ds_freq = props[ds_name]["frequency"]
        ds_config = f"{ds_name}/{ds_freq}/{term}"
        print(f"Processing {ds_config}")

        to_univariate = False if Dataset(name=ds_name, term=term,to_univariate=False,storage_env_var="BOOM").target_dim == 1 else True
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate, storage_env_var="BOOM")
        season_length = get_seasonality(dataset.freq)

        # make per-dataset plots directory
        ds_plots_dir = os.path.join(plots_root, ds_name.replace('/', '_'), term)
        os.makedirs(ds_plots_dir, exist_ok=True)

        metrics_by_model = {}


        # Persistence baseline (last-value persistence)
        pers = PersistencePredictor(prediction_length=dataset.prediction_length, freq=dataset.freq)
        res_pers = evaluate_model(pers, test_data=dataset.test_data, metrics=metrics, batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=season_length)
        write_row(csv_file_path, ds_config, "persistence", res_pers, props, ds_name)
        plot_forecasts_for_dataset(ds_plots_dir, ds_name, "persistence", dataset, pers, num_series=5)
        plot_metrics_summary(ds_plots_dir, "persistence", res_pers)
        metrics_by_model['persistence'] = res_pers

        # Seasonal Naive
        sn = SeasonalNaivePredictor(dataset.prediction_length, season_length=season_length, freq=dataset.freq, quantile_levels=COMMON_QUANTILES, batch_size=512, max_length=2048)
        res_sn = evaluate_model(sn, test_data=dataset.test_data, metrics=metrics, batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=season_length)
        write_row(csv_file_path, ds_config, "seasonalnaive", res_sn, props, ds_name)
        plot_forecasts_for_dataset(ds_plots_dir, ds_name, "seasonalnaive", dataset, sn, num_series=5)
        plot_metrics_summary(ds_plots_dir, "seasonalnaive", res_sn)
        metrics_by_model['seasonalnaive'] = res_sn


        # AutoARIMA -- taking too long to run on many datasets so disabled due to lack of time
        # ar = AutoARIMAPredictor(dataset.prediction_length, season_length=season_length, freq=dataset.freq, quantile_levels=COMMON_QUANTILES, batch_size=512, max_length=2048, stepwise=True)
        # res_ar = evaluate_model(ar, test_data=dataset.test_data, metrics=metrics, batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=season_length)
        # write_row(csv_file_path, ds_config, "autoarima", res_ar, props, ds_name)
        # plot_forecasts_for_dataset(ds_plots_dir, ds_name, "autoarima", dataset, ar, num_series=5)
        # plot_metrics_summary(ds_plots_dir, "autoarima", res_ar)
        # metrics_by_model['autoarima'] = res_ar

        # LSTM - build a list of training series robustly
        train_series = []
        if hasattr(dataset, "train_data"):
            entries = dataset.train_data
        elif hasattr(dataset, "train"):
            try:
                entries = list(dataset.train)
            except Exception:
                entries = dataset.test_data
        else:
            entries = dataset.test_data

        for entry in entries:
            # robust target extraction: entries may be dicts with 'target', or tuples/lists
            t = None
            try:
                if isinstance(entry, dict):
                    t = entry.get("target", None)
                elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    # some dataset representations use (start, target)
                    t = entry[1]
                elif isinstance(entry, (list, tuple)) and len(entry) == 1:
                    t = entry[0]
            except Exception:
                t = None

            if t is None:
                continue

            # If t is a dict (nested), try to find the first array-like value inside
            if isinstance(t, dict):
                found = False
                for v in t.values():
                    if isinstance(v, (list, tuple, np.ndarray)):
                        t = v
                        found = True
                        break
                if not found:
                    # cannot extract numeric series
                    continue

            try:
                arr = np.asarray(t, dtype=float)
            except Exception:
                # skip entries that cannot be coerced to numeric arrays
                continue
            # ensure 1-D array
            arr = np.atleast_1d(arr)
            if arr.size == 0:
                continue
            # if the series includes the forecast horizon at the end (test-data style), use only historical part
            if arr.size > dataset.prediction_length:
                train_series.append(arr[:-dataset.prediction_length])
            else:
                train_series.append(arr)
                
        lstm = LSTMPredictor(prediction_length=dataset.prediction_length, freq=dataset.freq, device="cpu")
        lstm.train(train_series, epochs=50, early_stopping_patience=2)

        res_lstm = evaluate_model(lstm, test_data=dataset.test_data, metrics=metrics, batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=season_length)
        write_row(csv_file_path, ds_config, "lstm", res_lstm, props, ds_name)
        plot_forecasts_for_dataset(ds_plots_dir, ds_name, "lstm", dataset, lstm, num_series=5)
        plot_metrics_summary(ds_plots_dir, "lstm", res_lstm)
        metrics_by_model['lstm'] = res_lstm

        # TFT predictor
        tft = TFTPredictor(prediction_length=dataset.prediction_length, freq=dataset.freq)
        tft.train(train_series)

        res_tft = evaluate_model(tft, test_data=dataset.test_data, metrics=metrics, batch_size=512, axis=None, mask_invalid_label=True, allow_nan_forecast=False, seasonality=season_length)
        write_row(csv_file_path, ds_config, "tft", res_tft, props, ds_name)
        plot_forecasts_for_dataset(ds_plots_dir, ds_name, "tft", dataset, tft, num_series=5)
        plot_metrics_summary(ds_plots_dir, "tft", res_tft)
        metrics_by_model['tft'] = res_tft

        # after all models for this dataset/term are evaluated, save a comparison plot
        from models.plotting import plot_compare_metrics
        plot_compare_metrics(ds_plots_dir, metrics_by_model)



def write_row(csv_file_path, ds_config, model_name, res, props, ds_name):
    """
     Write a single row to the CSV results file.
     As it will take a long time to run all datasets, we write results incrementally instead of making a pandas dataframe and writing at the end.
     """
    with open(csv_file_path, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            ds_config,
            model_name,
            res.get("MSE[mean]").iloc[0] if res.get("MSE[mean]") is not None else "",
            res.get("MSE[0.5]").iloc[0] if res.get("MSE[0.5]") is not None else "",
            res.get("MAE[0.5]").iloc[0] if res.get("MAE[0.5]") is not None else "",
            res.get("MASE[0.5]").iloc[0] if res.get("MASE[0.5]") is not None else "",
            res.get("MAPE[0.5]").iloc[0] if res.get("MAPE[0.5]") is not None else "",
            res.get("sMAPE[0.5]").iloc[0] if res.get("sMAPE[0.5]") is not None else "",
            res.get("MSIS").iloc[0] if res.get("MSIS") is not None else "",
            res.get("RMSE[mean]").iloc[0] if res.get("RMSE[mean]") is not None else "",
            res.get("NRMSE[mean]").iloc[0] if res.get("NRMSE[mean]") is not None else "",
            res.get("ND[0.5]").iloc[0] if res.get("ND[0.5]") is not None else "",
            res.get("mean_weighted_sum_quantile_loss").iloc[0] if res.get("mean_weighted_sum_quantile_loss") is not None else "",
            props[ds_name]["domain"],
            props[ds_name]["num_variates"],
        ])



