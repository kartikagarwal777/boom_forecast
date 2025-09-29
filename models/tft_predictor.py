import logging
from typing import Iterator, List
import numpy as np
import pandas as pd

from gluonts.model.predictor import RepresentablePredictor
from gluonts.model.forecast import QuantileForecast
from gluonts.dataset.util import forecast_start

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TimeSeriesDataSet, Baseline
from pytorch_forecasting.models import TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer

logger = logging.getLogger(__name__)


class TFTPredictor(RepresentablePredictor):
    """Temporal Fusion Transformer predictor using pytorch-forecasting.

    train(train_series: List[np.ndarray]) is expected to receive a list of 1-D
    numpy arrays (one per series). The predictor constructs a TimeSeriesDataSet,
    trains a TemporalFusionTransformer, and then predict yields GluonTS
    QuantileForecast objects for each test entry.
    """

    def __init__(self, prediction_length: int, freq: str, max_encoder_length: int = None, epochs: int = 5, batch_size: int = 64):
        super().__init__(prediction_length=prediction_length)
        self.prediction_length = prediction_length
        self.freq = freq
        self.max_encoder_length = max_encoder_length or prediction_length * 4
        self.epochs = epochs
        self.batch_size = batch_size

        self.tft = None
        self.train_dataset = None
        self.trainer = None

    def train(self, train_series: List[np.ndarray], **kwargs):
        # build training DataFrame
        rows = []
        for sid, arr in enumerate(train_series):
            arr = np.atleast_1d(np.asarray(arr, dtype=float))
            # pad short series so we have at least one encoder step + prediction
            min_len = self.prediction_length + 1
            if arr.size < min_len:
                pad_len = int(min_len - arr.size)
                pad_value = float(arr[-1]) if arr.size > 0 else 0.0
                pad = np.full((pad_len,), pad_value, dtype=float)
                arr = np.concatenate([pad, arr])
            for t, val in enumerate(arr):
                rows.append({"series": str(sid), "time_idx": int(t), "target": float(val)})

        df = pd.DataFrame(rows)

        if df.empty:
            raise RuntimeError("No training data provided for TFT")

        # compute encoder length and log series sizes for debugging
        series_sizes = df.groupby('series').size()
        logger.info(f"TFT training series sizes: {series_sizes.to_dict()}")
        max_encoder_length = int(min(self.max_encoder_length, int(series_sizes.max())))

        # construct TimeSeriesDataSet with a small min_encoder_length to avoid filtering
        self.train_dataset = TimeSeriesDataSet(
            df,
            time_idx='time_idx',
            target='target',
            group_ids=['series'],
            max_encoder_length=max_encoder_length,
            min_encoder_length=1,
            max_prediction_length=self.prediction_length,
            time_varying_unknown_reals=['target'],
            target_normalizer=GroupNormalizer(groups=['series']),
        )

        # dataloaders
        train_dataloader = self.train_dataset.to_dataloader(train=True, batch_size=self.batch_size, num_workers=8)
        val_dataloader = self.train_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=8)

        # build model
        self.tft = TemporalFusionTransformer.from_dataset(
            self.train_dataset,
            learning_rate=1e-3,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            hidden_continuous_size=8,
            output_size=7,  # number of quantiles to predict by default in PTF
            loss=None,
            log_interval=0,
            reduce_on_plateau_patience=4,
        )

        # trainer
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=3, mode="min")
        self.trainer = Trainer(max_epochs=self.epochs, accelerator="cpu", devices=1, callbacks=[early_stop_callback], enable_checkpointing=False)
        # fit
        try:
            import lightning.pytorch as pl
            model_to_fit = self.tft
            if not isinstance(model_to_fit, pl.LightningModule):
                # try common conversion attributes
                for attr in ("to_lightning", "to_lightning_module", "lightning_module", "_lightning_module", "to_pl"):
                    fn = getattr(self.tft, attr, None)
                    if callable(fn):
                        candidate = fn()
                        if isinstance(candidate, pl.LightningModule):
                            model_to_fit = candidate
                            break
                    elif fn is not None and isinstance(fn, pl.LightningModule):
                        model_to_fit = fn
                        break
            if not isinstance(model_to_fit, pl.LightningModule):
                raise TypeError(f"TemporalFusionTransformer instance is not a LightningModule (type={type(self.tft)}); cannot fit. Available attrs: {dir(self.tft)}")
            self.trainer.fit(model_to_fit, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        except Exception:
            logger.exception("TFT training failed")
            raise

    def predict(self, dataset, **kwargs) -> Iterator[QuantileForecast]:
        # Collect all entries to predict and build an inference dataframe
        infer_rows = []
        meta = []  # store (item_id, start, history_length)
        sid = 0
        for entry in dataset:
            # robust extraction
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
                        item_id = None
                    else:
                        target = entry[0]; start = None; item_id = None
            else:
                continue

            arr = np.atleast_1d(np.asarray(target, dtype=float))
            h = self.prediction_length
            # if arr contains history+future, strip future off for history
            if arr.size > h:
                history = arr[:-h]
            elif arr.size == h:
                history = np.array([])
            else:
                history = arr
            # ensure history is long enough for the model encoder by padding at the left
            encoder_needed = int(getattr(self.train_dataset, 'max_encoder_length', self.max_encoder_length))
            if history.size < encoder_needed:
                pad_len = int(encoder_needed - history.size)
                pad_value = float(history[-1]) if history.size > 0 else 0.0
                history = np.concatenate([np.full((pad_len,), pad_value, dtype=float), history])

            # build rows for history
            for t, val in enumerate(history):
                infer_rows.append({"series": str(sid), "time_idx": int(t), "target": float(val)})
            # add future rows with NaN target so TimeSeriesDataSet knows to predict these
            start_idx = len(history)
            for t in range(start_idx, start_idx + h):
                infer_rows.append({"series": str(sid), "time_idx": int(t), "target": np.nan})

            meta.append((item_id, start, len(history)))
            sid += 1

        if len(infer_rows) == 0:
            return

        infer_df = pd.DataFrame(infer_rows)

        # fill NaN targets for the dataset construction; model won't use these during prediction
        if infer_df['target'].isna().any():
            infer_df['target'] = infer_df['target'].fillna(0.0)

        # build dataset for inference using same parameters as training
        predict_dataset = TimeSeriesDataSet.from_dataset(
            self.train_dataset,
            infer_df,
            min_prediction_length=self.prediction_length,
            min_encoder_length=self.train_dataset.max_encoder_length,
            allow_missing_timesteps=True,
            constant_fill_strategy={"target": 0.0},
            predict_mode=True,
        )

        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=self.batch_size, num_workers=0)

        # get predictions (return pandas DataFrame with quantiles)
        preds = self.tft.predict(predict_dataloader, return_x=False)

        # Normalize preds into a numpy array of shape (n_series, prediction_length) for the mean
        try:
            if isinstance(preds, np.ndarray):
                mean_arr = preds
                quantile_arrays = None
            elif isinstance(preds, pd.DataFrame):
                # Many PTF versions return a DataFrame with a 'prediction' column containing arrays
                if 'prediction' in preds.columns:
                    mean_arr = np.stack([np.asarray(x) for x in preds['prediction'].values])
                    quantile_arrays = None
                else:
                    # try to detect numeric column names that represent quantiles
                    qcols = []
                    for c in preds.columns:
                        try:
                            fq = float(c)
                            qcols.append((fq, c))
                        except Exception:
                            continue
                    if len(qcols) > 0:
                        qcols_sorted = [c for _, c in sorted(qcols)]
                        quantile_arrays = [np.stack([np.asarray(x) for x in preds[c].values]) for c in qcols_sorted]
                        # try to get mean/prediction column if present
                        if 'mean' in preds.columns:
                            mean_arr = np.stack([np.asarray(x) for x in preds['mean'].values])
                        elif 'prediction' in preds.columns:
                            mean_arr = np.stack([np.asarray(x) for x in preds['prediction'].values])
                        else:
                            mean_arr = np.mean(np.stack(quantile_arrays), axis=0)
                    else:
                        # fallback: take the first column as arrays
                        col0 = preds.columns[0]
                        mean_arr = np.stack([np.asarray(x) for x in preds[col0].values])
                        quantile_arrays = None
            else:
                # unknown type; attempt to coerce to numpy
                mean_arr = np.asarray(preds)
                quantile_arrays = None

            # ensure 2D
            if mean_arr.ndim == 1:
                mean_arr = np.atleast_2d(mean_arr)

            n_series = mean_arr.shape[0]
            for i in range(n_series):
                # build forecast_arrays: mean first, then quantiles if available, else duplicate mean
                mean_row = np.atleast_1d(mean_arr[i])
                if quantile_arrays is not None:
                    qrows = [q[i] for q in quantile_arrays]
                else:
                    qrows = [mean_row.copy() for _ in range(9)]
                forecast_arrays = np.vstack([mean_row] + qrows)
                item_id, start, hist_len = meta[i]
                # compute start_date from start or forecast_start
                try:
                    fs = None
                    if start is not None:
                        fs = forecast_start({"start": start, "target": np.zeros(hist_len + self.prediction_length)})
                    else:
                        fs = predict_dataset.index.get_level_values('time_idx')[0]
                    if hasattr(fs, 'to_period'):
                        start_date = fs.to_period(freq=self.freq)
                    else:
                        start_date = pd.Period(fs, freq=self.freq)
                except Exception:
                    start_date = pd.Period(pd.Timestamp.now(), freq=self.freq)

                # keys: mean + quantiles
                if quantile_arrays is not None:
                    # build keys from detected quantile column names if possible
                    qkeys = [str(q) for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]
                else:
                    qkeys = [str(q) for q in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]]

                yield QuantileForecast(forecast_arrays=forecast_arrays, forecast_keys=['mean'] + qkeys, start_date=start_date, item_id=item_id)
        except Exception:
            logger.exception("TFT predict/formatting failed; falling back to persistence per-series forecasts")
            # As a conservative fallback, yield persistence-like forecasts
            COMMON_QUANTILES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
            sid = 0
            for item_id, start, hist_len in meta:
                # try to pull last history value from infer_df
                hist = infer_df[(infer_df.series == str(sid)) & (~infer_df.target.isna())]['target'].values
                last = hist[-1] if len(hist) > 0 else 0.0
                mean_row = np.full((1, self.prediction_length), fill_value=last, dtype=float)
                qrows = [mean_row.copy() for _ in COMMON_QUANTILES]
                forecast_arrays = np.vstack([mean_row] + qrows)
                try:
                    fs = forecast_start({"start": start, "target": np.zeros(hist_len + self.prediction_length)})
                    if hasattr(fs, 'to_period'):
                        start_date = fs.to_period(freq=self.freq)
                    else:
                        start_date = pd.Period(fs, freq=self.freq)
                except Exception:
                    start_date = pd.Period(pd.Timestamp.now(), freq=self.freq)
                yield QuantileForecast(forecast_arrays=forecast_arrays, forecast_keys=['mean'] + [str(q) for q in COMMON_QUANTILES], start_date=start_date, item_id=item_id)
                sid += 1
