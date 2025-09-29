import numpy as np
from typing import Iterator
import pandas as pd
from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import QuantileForecast
from gluonts.dataset.util import forecast_start


class PersistencePredictor(RepresentablePredictor):
    """Simple persistence predictor: repeats the last observed value across the horizon.

    Produces a QuantileForecast with duplicated mean and simple quantile keys so it is
    compatible with evaluate_model.
    """

    COMMON_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, prediction_length: int, freq: str):
        super().__init__(prediction_length=prediction_length)
        self.prediction_length = prediction_length
        self.freq = freq

    def train(self, *args, **kwargs):
        # no training needed
        return

    def predict(self, dataset, **kwargs) -> Iterator[QuantileForecast]:
        for entry in dataset:
            # extract target similar to other predictors
            target = None
            start = None
            item_id = None
            if isinstance(entry, dict):
                target = entry.get('target')
                start = entry.get('start')
                item_id = entry.get('item_id', None)
            elif isinstance(entry, (list, tuple)):
                inner = None
                for el in reversed(entry):
                    if isinstance(el, dict) and ('target' in el or 'start' in el):
                        inner = el
                        break
                if inner is not None:
                    target = inner.get('target')
                    start = inner.get('start')
                    item_id = inner.get('item_id', None)
                else:
                    if len(entry) >= 2:
                        start, target = entry[0], entry[1]
                    else:
                        target = entry[0]

            if target is None:
                raise RuntimeError('Cannot extract target for PersistencePredictor')

            arr = np.asarray(target, dtype=float)

            # determine last value per variate
            if arr.ndim == 1:
                last = arr[-1] if arr.size > 0 else 0.0
                mean_row = np.full((1, self.prediction_length), fill_value=last, dtype=float)
                quantile_rows = [mean_row.copy() for _ in self.COMMON_QUANTILES]
                forecast_arrays = np.vstack([mean_row] + quantile_rows)
            elif arr.ndim == 2:
                last_vals = arr[:, -1]
                num_vars = arr.shape[0]
                mean_row = np.tile(last_vals.reshape(num_vars, 1), (1, self.prediction_length))
                quantile_rows = [mean_row.copy() for _ in self.COMMON_QUANTILES]
                forecast_arrays = np.stack([mean_row] + quantile_rows, axis=0)
            else:
                raise RuntimeError('Unsupported target ndim for PersistencePredictor')

            forecast_keys = ['mean'] + [str(q) for q in self.COMMON_QUANTILES]

            # determine start_date for the forecast
            try:
                if isinstance(entry, dict):
                    fs = forecast_start(entry)
                else:
                    fs = forecast_start({'start': start, 'target': target})
                if hasattr(fs, 'to_period'):
                    start_date = fs.to_period(freq=self.freq)
                else:
                    start_date = pd.Period(fs, freq=self.freq)
            except Exception:
                # fallback: try to coerce start
                try:
                    start_date = pd.Period(start, freq=self.freq)
                except Exception:
                    start_date = pd.Period(pd.Timestamp.now(), freq=self.freq)

            yield QuantileForecast(forecast_arrays=forecast_arrays, forecast_keys=forecast_keys, start_date=start_date, item_id=item_id)
