import inspect
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Type
import logging

import numpy as np
import pandas as pd
from gluonts.core.component import validated
from gluonts.dataset import Dataset
from gluonts.dataset.util import forecast_start
from gluonts.model import Forecast
from gluonts.model.forecast import QuantileForecast
from gluonts.model.predictor import RepresentablePredictor
from gluonts.transform.feature import LastValueImputation, MissingValueImputation
from statsforecast import StatsForecast
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    AutoARIMA,
)


@dataclass
class ModelConfig:
    quantile_levels: Optional[List[float]] = None
    forecast_keys: List[str] = field(init=False)
    statsforecast_keys: List[str] = field(init=False)
    intervals: Optional[List[int]] = field(init=False)

    def __post_init__(self):
        self.forecast_keys = ["mean"]
        self.statsforecast_keys = ["mean"]
        if self.quantile_levels is None:
            self.intervals = None
            return

        intervals = set()

        for quantile_level in self.quantile_levels:
            interval = round(200 * (max(quantile_level, 1 - quantile_level) - 0.5))
            intervals.add(interval)
            side = "hi" if quantile_level > 0.5 else "lo"
            self.forecast_keys.append(str(quantile_level))
            self.statsforecast_keys.append(f"{side}-{interval}")

        self.intervals = sorted(intervals)


class StatsForecastPredictor(RepresentablePredictor):
    """
    A predictor type that wraps models from the `statsforecast` package.

    Subclass this and set the ModelType class attribute to any statsforecast model
    (for example AutoARIMA, SeasonalNaive, etc).
    """

    ModelType: Type

    @validated()
    def __init__(
        self,
        prediction_length: int,
        season_length: int,
        freq: str,
        quantile_levels: Optional[List[float]] = None,
        imputation_method: MissingValueImputation = LastValueImputation(),
        max_length: Optional[int] = None,
        batch_size: int = 1,
        parallel: bool = False,
        **model_params,
    ) -> None:
        super().__init__(prediction_length=prediction_length)

        if "season_length" in inspect.signature(self.ModelType.__init__).parameters:
            model_params["season_length"] = season_length

        self.freq = freq
        self.model = StatsForecast(
            models=[self.ModelType(**model_params)],
            freq=freq,
            n_jobs=-1 if parallel else 1,
        )
        # Do not create a fallback model; if the model returns NaNs we will raise an error
        self.config = ModelConfig(quantile_levels=quantile_levels)
        self.imputation_method = imputation_method
        self.batch_size = batch_size
        self.max_length = max_length

        # Set up the logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def predict(self, dataset: Dataset, **kwargs) -> Iterator[Forecast]:
        batch = {}
        batch_mapping = {}
        total_series = len(dataset)
        self.logger.info(f"Starting prediction on {total_series} series.")

        for idx, entry in enumerate(dataset):
            # normalize entry to a dict-like that contains 'target' and 'start'
            entry_dict = None
            if isinstance(entry, dict):
                entry_dict = entry
            elif isinstance(entry, (list, tuple)):
                # prefer the second element when it's a dict (common (start, target) or (train, test) formats)
                if len(entry) >= 2 and isinstance(entry[1], dict):
                    entry_dict = entry[1]
                elif len(entry) >= 1 and isinstance(entry[0], dict):
                    entry_dict = entry[0]
                else:
                    # fallback: if tuple is (start, target) convert to dict
                    try:
                        start_val = entry[0]
                        target_val = entry[1] if len(entry) >= 2 else entry[0]
                        entry_dict = {"start": start_val, "target": target_val}
                    except Exception:
                        raise RuntimeError("Unsupported entry format for statsforecast predictor")
            else:
                raise RuntimeError("Unsupported entry type for statsforecast predictor")

            target = np.asarray(entry_dict["target"], np.float32)
            start = entry_dict["start"]
            item_id = entry_dict.get("item_id", None)
            # Use a unique internal mapping key to avoid collisions when multiple entries share the same item_id
            mapping_key = f"{item_id if item_id is not None else idx}_{idx}"

            # trim to max_length if requested
            if self.max_length is not None and target.size > self.max_length:
                # adjust start forward by the amount trimmed
                trim = target.size - self.max_length
                try:
                    entry["start"] += trim
                except Exception:
                    pass
                target = target[-self.max_length :]

            # impute NaNs if present
            if np.isnan(target).any():
                target = target.copy()
                target = self.imputation_method(target)

            # Handle univariate and multivariate targets
            unique_ids = []
            if target.ndim == 1:
                uid = f"{mapping_key}_{str(forecast_start(entry_dict))}_{len(batch)}"
                df = pd.DataFrame(
                    {
                        "unique_id": uid,
                        "ds": pd.date_range(start=start.to_timestamp(), periods=len(target), freq=start.freq).to_numpy(),
                        "y": target,
                    }
                )
                batch[uid] = df
                unique_ids.append(uid)
            elif target.ndim == 2:
                # treat axis 0 as variates
                for v in range(target.shape[0]):
                    series = target[v, :].astype(np.float32)
                    uid = f"{mapping_key}_v{v}_{str(forecast_start(entry_dict))}_{len(batch)}"
                    df = pd.DataFrame(
                        {
                            "unique_id": uid,
                            "ds": pd.date_range(start=start.to_timestamp(), periods=len(series), freq=start.freq).to_numpy(),
                            "y": series,
                        }
                    )
                    batch[uid] = df
                    unique_ids.append(uid)
            else:
                raise RuntimeError("Unsupported target ndim for statsforecast predictor")

            batch_mapping[mapping_key] = {"unique_ids": unique_ids, "start": start, "orig_id": item_id if item_id is not None else str(idx), "mapping_key": mapping_key}

            # process when batch is full (counts unique series)
            if len(batch) >= self.batch_size:
                self.logger.info(f"Processing batch {idx // self.batch_size + 1}.")
                results = self.sf_predict(pd.concat(batch.values()))
                yield from self.yield_forecast(batch_mapping, results)
                batch = {}
                batch_mapping = {}

        if len(batch) > 0:
            self.logger.info(f"Processing final batch.")
            results = self.sf_predict(pd.concat(batch.values()))
            yield from self.yield_forecast(batch_mapping, results)

        self.logger.info("Prediction completed.")

    def sf_predict(self, Y_df: pd.DataFrame) -> pd.DataFrame:
        kwargs = {}
        if self.config.intervals is not None:
            kwargs["level"] = self.config.intervals
        results = self.model.forecast(
            df=Y_df,
            h=self.prediction_length,
            **kwargs,
        )
        # If any NaNs present in the forecasts, raise immediately (no fallback) so the caller sees the error
        if results.isnull().values.any():
            row_nan = results.isnull().any(axis=1)
            nan_ids = results[row_nan].index.tolist()
            raise RuntimeError(f"Forecast contains NaN values for ids: {nan_ids}")

        return results

    def yield_forecast(self, mapping, results: pd.DataFrame) -> Iterator[QuantileForecast]:
        """
        mapping: dict mapping original_item_id -> dict with keys:
            - 'unique_ids': list of unique_id strings (one per variate)
            - 'start': original start object
        """

        results.set_index("unique_id", inplace=True)
        model_name = self.ModelType.__name__

        for _, meta in mapping.items():
            unique_ids = meta["unique_ids"]
            start = meta.get("start")
            orig_id = meta.get("orig_id")
            mapping_key = meta.get("mapping_key")
            # collect per-key per-var arrays
            per_key_arrays = []
            for key in self.config.statsforecast_keys:
                # for each variate, extract the column for this key
                var_arrays = []
                for uid in unique_ids:
                    if key == "mean":
                        col_name = model_name
                    else:
                        col_name = f"{model_name}-{key}"
                    # prediction for this uid is a DataFrame with h rows
                    if uid not in results.index:
                        raise RuntimeError(f"Missing forecast results for unique_id {uid}")
                    pred_df = results.loc[uid]
                    var_arrays.append(pred_df.loc[:, col_name].to_numpy())
                # stack variates into shape (num_vars, h)
                per_key_arrays.append(np.stack(var_arrays, axis=0))

            # per_key_arrays is length num_keys, each is (num_vars, h) -> stack into (num_keys, num_vars, h)
            forecast_arrays = np.stack(per_key_arrays, axis=0)
            # For univariate series, remove the middle axis to match GluonTS expectations: (num_keys, h)
            if per_key_arrays and per_key_arrays[0].ndim == 2 and per_key_arrays[0].shape[0] == 1:
                forecast_arrays = np.squeeze(forecast_arrays, axis=1)

            # determine start_date
            try:
                start_date = start.to_period(freq=self.freq)
            except Exception:
                # fallback to deriving start from the first uid's prediction ds
                first_uid = unique_ids[0]
                pred_df = results.loc[first_uid]
                start_date = pred_df.ds.iloc[0].to_period(freq=self.freq)

            # Use the internal mapping_key as the Forecast.item_id to guarantee uniqueness for the evaluator.
            # Preserve the original id in the info field so downstream code can still see it.
            # GluonTS version in this environment expects `.forecast_array` and `_forecast_dict` attributes
            # Build the per-quantile dict used by newer GluonTS: map forecast_keys -> array shaped (num_vars, h) or (h,) for univariate
            forecast_dict = {}
            for i, key in enumerate(self.config.forecast_keys):
                forecast_dict[key] = forecast_arrays[i]

            qf = QuantileForecast(
                forecast_arrays=forecast_arrays,
                forecast_keys=self.config.forecast_keys,
                start_date=start_date,
                item_id=mapping_key,
                info={"orig_id": orig_id},
            )

            # set compatibility attributes depending on installed GluonTS implementation
            # prefer `forecast_array` and `_forecast_dict` as observed in this environment
            try:
                setattr(qf, 'forecast_array', forecast_arrays)
                setattr(qf, '_forecast_dict', {k: np.asarray(v) for k, v in forecast_dict.items()})
            except Exception:
                # fallback: set forecast_arrays attribute if available
                try:
                    setattr(qf, 'forecast_arrays', forecast_arrays)
                except Exception:
                    pass

            yield qf


class SeasonalNaivePredictor(StatsForecastPredictor):
    ModelType = SeasonalNaive


class AutoARIMAPredictor(StatsForecastPredictor):
    ModelType = AutoARIMA
