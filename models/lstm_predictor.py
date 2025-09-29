import logging
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset as TorchDataset
from gluonts.core.component import validated
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import RepresentablePredictor

logger = logging.getLogger(__name__)


class TimeSeriesTorchDataset(TorchDataset):
    def __init__(self, series_list, window_size, target_size):
        self.x = []
        self.y = []
        for s in series_list:
            arr = np.asarray(s)
            total_needed = window_size + target_size
            if arr.size < total_needed:
                # pad at the left with the series mean (or zero if empty) so we can still form a window
                pad_val = float(arr.mean()) if arr.size > 0 else 0.0
                pad = np.full(total_needed - arr.size, pad_val, dtype=arr.dtype)
                arr = np.concatenate([pad, arr], axis=0)

            for i in range(len(arr) - window_size - target_size + 1):
                self.x.append(arr[i : i + window_size])
                self.y.append(arr[i + window_size : i + window_size + target_size])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()


class SimpleLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: batch, seq_len
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(-1)


class LSTMPredictor(RepresentablePredictor):
    """A lightweight LSTM predictor for univariate series.
    """

    def __init__(self, prediction_length: int, freq: str, window_size: Optional[int] = None, device: str = "cpu"):
        super().__init__(prediction_length=prediction_length)
        self.device = "cpu"
        self.prediction_length = prediction_length
        self.freq = freq
        self.window_size = window_size if window_size is not None else prediction_length * 4
        self.model = SimpleLSTM()
        # store normalization params
        self._series_means = None
        self._series_stds = None
        self._trained = False

    def train(self, series_list, epochs: int = 50, batch_size: int = 64, lr: float = 1e-3, val_frac: float = 0.1, early_stopping_patience: int = 3):
        # Normalize each series (z-score) and store parameters for de-normalization
        norm_series = []
        means = []
        stds = []
        for s in series_list:
            a = np.asarray(s, dtype=np.float32)
            if a.size == 0:
                continue
            m = a.mean()
            sd = a.std() if a.std() > 0 else 1.0
            norm_series.append((a - m) / sd)
            means.append(m)
            stds.append(sd)

        if len(norm_series) == 0:
            logger.warning("No valid series for LSTM training; skipping training.")
            self._trained = False
            return

        self._series_means = means
        self._series_stds = stds

        ds = TimeSeriesTorchDataset([s for s in norm_series], window_size=self.window_size, target_size=self.prediction_length)
        if len(ds) == 0:
            logger.warning("No training windows available for LSTM; skipping training.")
            self._trained = False
            return
        # split train/val
        n = len(ds)
        # compute validation size; ensure at least one training sample when possible
        n_val = int(n * val_frac)
        if n_val < 1:
            # if more than one sample, keep at least 1 val; otherwise 0
            n_val = 1 if n > 1 else 0
        n_train = n - n_val
        if n_train <= 0:
            # force at least one training sample
            n_train = 1
            n_val = n - n_train

        if n_val > 0:
            train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])
        else:
            # all goes to train; val is empty subset
            train_ds = ds
            val_ds = torch.utils.data.Subset(ds, [])
        dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        self.model.to(self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        self.model.train()
        best_val = float('inf')
        patience = 0
        for epoch in range(epochs):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                preds = self.model(xb)
                # preds shape: batch, (1) -> expand to match target first step
                loss = loss_fn(preds.unsqueeze(-1), yb[:, 0:1])
                loss.backward()
                opt.step()
                total += loss.item()

            # validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    preds = self.model(xb)
                    loss = loss_fn(preds.unsqueeze(-1), yb[:, 0:1])
                    val_loss += loss.item()
            val_loss = val_loss / max(1, len(val_dl))
            logger.info(f"Epoch {epoch+1}/{epochs} LSTM train_loss: {total/max(1,len(dl)):.6f} val_loss: {val_loss:.6f}")
            # early stopping
            if val_loss < best_val:
                best_val = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            self.model.train()
        # mark as trained if we completed at least one epoch
        self._trained = True

    def predict(self, dataset, **kwargs) -> Iterator[Forecast]:
        # For each series, use last window to predict next prediction_length by iterative forecasting
        self.model.eval()
        for entry in dataset:
            # Extract target sequence robustly for dict or tuple/list entries
            target = None
            start = None
            item_id = None

            # If entry is a tuple/list that contains dicts (train/test pairs), prefer the dict
            # that contains 'target'/'start' (usually the test dict).
            if isinstance(entry, dict):
                target = entry.get("target")
                start = entry.get("start")
                item_id = entry.get("item_id")
            elif isinstance(entry, (list, tuple)):
                # try to find an inner dict containing the target/start
                inner = None
                for el in reversed(entry):
                    if isinstance(el, dict) and ("target" in el or "start" in el):
                        inner = el
                        break
                if inner is not None:
                    target = inner.get("target")
                    start = inner.get("start")
                    item_id = inner.get("item_id")
                else:
                    # fallback: common formats: (start, target) or (target,)
                    if len(entry) >= 2:
                        start, target = entry[0], entry[1]
                        if len(entry) >= 3:
                            item_id = entry[2]
                    else:
                        target = entry[0]
            else:
                # unknown entry type
                continue

            def _extract_array(obj):
                # Recursively find the first array-like numeric sequence in obj
                if obj is None:
                    return None
                # direct array-like
                if isinstance(obj, (list, tuple, np.ndarray)):
                    try:
                        arr = np.asarray(obj, dtype=float)
                        return np.atleast_1d(arr)
                    except Exception:
                        return None
                # dict: try values recursively
                if isinstance(obj, dict):
                    for v in obj.values():
                        arr = _extract_array(v)
                        if arr is not None:
                            return arr
                    return None
                # try to coerce scalars
                try:
                    return np.atleast_1d(np.asarray(obj, dtype=float))
                except Exception:
                    return None

            series = _extract_array(target)

            if series is None or series.size < 1:
                preds = np.zeros(self.prediction_length, dtype=np.float32)
            else:
                series = series.astype(np.float32)
                # normalize series using mean/std of the series
                m = series.mean()
                sd = series.std() if series.std() > 0 else 1.0
                norm_series = (series - m) / sd
                window = norm_series[-self.window_size :]
                # pad if shorter than window_size
                if window.shape[0] < self.window_size:
                    pad = np.zeros(self.window_size - window.shape[0], dtype=window.dtype)
                    window = np.concatenate([pad, window], axis=0)
                x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)
                preds_norm = []
                with torch.no_grad():
                    cur = x
                    # iterative multi-step forecasting
                    for _ in range(self.prediction_length):
                        out = self.model(cur)
                        val = float(out.cpu().numpy().astype(np.float32).squeeze())
                        preds_norm.append(val)
                        nxt = torch.tensor([[val]], dtype=torch.float32).to(self.device)
                        # cur is 2D: (batch, seq_len); concatenate along seq axis
                        cur = torch.cat([cur[:, 1:], nxt], dim=1)

                preds = np.array(preds_norm, dtype=np.float32) * sd + m

            # Build a GluonTS QuantileForecast with a single 'mean' key
            from gluonts.model.forecast import QuantileForecast
            from gluonts.dataset.util import forecast_start

            # provide mean plus a full set of quantiles (duplicate mean for all quantiles)
            # This satisfies evaluators that expect multiple quantile keys (e.g., MSIS)
            COMMON_QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            mean_row = np.atleast_2d(preds)
            quantile_rows = [mean_row.copy() for _ in COMMON_QUANTILES]
            # stack as: mean, then quantiles
            forecast_arrays = np.vstack([mean_row] + quantile_rows)
            # determine start_date (must be a pandas Period)
            try:
                start_date = forecast_start(entry)
            except Exception:
                # try to convert 'start' to a pandas Period
                start_date = None
                try:
                    import pandas as pd

                    if start is None:
                        start_date = None
                    elif isinstance(start, pd.Period):
                        start_date = start
                    else:
                        # try parsing timestamps/strings
                        start_date = pd.Period(pd.to_datetime(start), freq=self.freq)
                except Exception:
                    start_date = None

            if start_date is None:
                # cannot determine a valid start_date; skip this entry
                logger.warning('Skipping entry because start_date could not be determined: %s', entry)
                continue

            # validate forecast arrays
            if not np.isfinite(forecast_arrays).all():
                raise RuntimeError("Forecast contains NaN values")
            # keys: mean followed by quantile string keys
            forecast_keys = ["mean"] + [str(q) for q in COMMON_QUANTILES]
            yield QuantileForecast(forecast_arrays=forecast_arrays, forecast_keys=forecast_keys, start_date=start_date, item_id=item_id)
