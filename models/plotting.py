import os
from typing import Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gluonts.dataset.util import forecast_start


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_single_series(ax, history, forecast_mean, forecast_qs: Optional[dict], hist_idx, fc_idx, title=None):
    # history: 1-D numpy array
    # forecast_mean: 1-D numpy array of length h
    # forecast_qs: dict quantile_str -> array of length h (optional)
    # hist_idx and fc_idx are pandas DatetimeIndex
    ax.plot(hist_idx, history, label='actual', color='black')
    ax.plot(fc_idx, forecast_mean, label='predicted_mean', color='C1')
    if forecast_qs:
        # shade between 10-90 if available
        if '0.1' in forecast_qs and '0.9' in forecast_qs:
            ax.fill_between(fc_idx, forecast_qs['0.1'], forecast_qs['0.9'], color='C1', alpha=0.2, label='10-90')
    if title:
        ax.set_title(title)
    ax.legend()


def _extract_forecast_arrays(forecast):
    # forecast may be GluonTS QuantileForecast - try to extract mean and quantiles
    # Try common attributes
    try:
        # newer interface
        f_dict = getattr(forecast, '_forecast_dict', None)
        if f_dict is None:
            f_dict = getattr(forecast, 'forecast_arrays', None)
            if f_dict is not None:
                # forecast_arrays is ndarray; first axis keys unknown
                arr = f_dict
                mean = arr[0]
                qs = None
                return mean, qs
        else:
            # f_dict: map key->array
            mean = np.asarray(f_dict.get('mean'))
            qs = {str(k): np.asarray(v) for k, v in f_dict.items() if k != 'mean'}
            return mean, qs
    except Exception:
        pass

    # try attribute 'forecast_array' (single array)
    try:
        arr = getattr(forecast, 'forecast_array', None)
        if arr is not None:
            # arr shape (k, h) or (k, num_vars, h)
            mean = arr[0]
            return mean, None
    except Exception:
        pass

    # last-resort: try to use .mean or .samples
    try:
        arr = np.asarray(forecast.mean)
        return arr, None
    except Exception:
        pass

    raise RuntimeError('Unsupported forecast object for plotting')


def plot_forecasts_for_dataset(output_dir: str, ds_name: str, model_name: str, dataset, predictor, num_series: int = 5):
    """Run predictor on a few series from dataset.test_data and save plots."""
    _ensure_dir(output_dir)
    it = iter(dataset.test_data)
    plotted = 0
    for entry in it:
        if plotted >= num_series:
            break
        # extract history and start
        if isinstance(entry, dict):
            target = entry.get('target')
            start = entry.get('start')
        elif isinstance(entry, (list, tuple)):
            # prefer dict inside
            inner = None
            for el in reversed(entry):
                if isinstance(el, dict) and ('target' in el or 'start' in el):
                    inner = el
                    break
            if inner is not None:
                target = inner.get('target')
                start = inner.get('start')
            else:
                if len(entry) >= 2:
                    start, target = entry[0], entry[1]
                else:
                    target = entry[0]
                    start = None
        else:
            continue

        arr_target = np.asarray(target)

        # get forecast for this single entry first to learn the horizon h
        single_ds = [entry]
        preds = list(predictor.predict(single_ds))
        if len(preds) == 0:
            continue
        fc = preds[0]
        try:
            mean, qs = _extract_forecast_arrays(fc)
        except Exception:
            continue

        h = len(mean)

        # Split arr_target into history and true future if possible.
        # Cases:
        # - len(arr_target) > h : arr_target contains history + future (common where full series provided)
        # - len(arr_target) == h: arr_target contains only the future (test-only) -> no history available
        # - len(arr_target) < h : unexpected; treat all as history
        if len(arr_target) > h:
            hist_len = len(arr_target) - h
            history = arr_target[:hist_len]
            future_actual = arr_target[hist_len:]
        elif len(arr_target) == h:
            # only future available in this entry
            history = np.asarray([])
            future_actual = arr_target
        else:
            history = arr_target
            future_actual = None

        # determine start points explicitly
        def safe_forecast_start(e):
            # handle dict-like entries or tuples/lists
            if isinstance(e, dict):
                return forecast_start(e)
            if isinstance(e, (list, tuple)):
                # try to find an inner dict with start/target
                for el in reversed(e):
                    if isinstance(el, dict) and ("start" in el or "target" in el):
                        return forecast_start(el)
                # fallback: assume (start, target)
                try:
                    st = e[0]
                    tg = e[1]
                    import numpy as _np

                    fake = {"start": st, "target": _np.asarray(tg)}
                    return forecast_start(fake)
                except Exception:
                    raise RuntimeError("Cannot determine forecast start for entry")
            raise RuntimeError("Unsupported entry type for forecast_start")

        fs = safe_forecast_start(entry)
        # forecast_start should be the first forecast timestamp
        if isinstance(fs, pd.Period):
            forecast_start_ts = fs.to_timestamp()
        else:
            forecast_start_ts = pd.to_datetime(fs)

        # Prefer to construct history index from the entry's 'start' if available
        entry_start = None
        try:
            if isinstance(entry, dict):
                entry_start = entry.get('start', None)
            elif isinstance(entry, (list, tuple)):
                # find a dict with start inside
                inner = None
                for el in reversed(entry):
                    if isinstance(el, dict) and 'start' in el:
                        inner = el
                        break
                if inner is not None:
                    entry_start = inner.get('start')
        except Exception:
            entry_start = None

        if entry_start is not None:
            try:
                if isinstance(entry_start, pd.Period):
                    hist_start_ts = entry_start.to_timestamp()
                else:
                    hist_start_ts = pd.to_datetime(entry_start)
                hist_idx = pd.date_range(start=hist_start_ts, periods=len(history), freq=dataset.freq)
            except Exception:
                # fallback to compute hist by ending one step before forecast start
                try:
                    from pandas.tseries.frequencies import to_offset
                    offset = to_offset(dataset.freq)
                except Exception:
                    offset = pd.tseries.frequencies.to_offset('D')
                hist_idx = pd.date_range(end=forecast_start_ts - offset, periods=len(history), freq=dataset.freq)
        else:
            # fallback: compute history index so it ends right before forecast_start
            try:
                from pandas.tseries.frequencies import to_offset
                offset = to_offset(dataset.freq)
            except Exception:
                offset = pd.tseries.frequencies.to_offset('D')
            hist_idx = pd.date_range(end=forecast_start_ts - offset, periods=len(history), freq=dataset.freq)

        fc_idx = pd.date_range(start=forecast_start_ts, periods=len(mean), freq=dataset.freq)

        fig, ax = plt.subplots(figsize=(8, 3))
        plot_single_series(ax, history, mean, qs, hist_idx, fc_idx, title=f"{ds_name} - {model_name}")
        outpath = os.path.join(output_dir, f"{model_name}_series_{plotted}.png")
        fig.tight_layout()
        fig.savefig(outpath)
        plt.close(fig)

        # Save predicted vs actual CSV, including true future actuals when present
        try:
            # build series for history, future (if any), and predictions
            if len(history) > 0:
                hist_series = pd.Series(data=history, index=hist_idx)
            else:
                hist_series = pd.Series(dtype=float)

            pred_series = pd.Series(data=mean, index=fc_idx)

            if future_actual is not None and len(future_actual) == len(mean):
                fut_series = pd.Series(data=future_actual, index=fc_idx)
            else:
                fut_series = pd.Series(dtype=float)

            # unified index covering history and forecast timestamps
            idx = hist_series.index.union(pred_series.index)

            # DataFrame with unified index
            df_out = pd.DataFrame(index=idx)

            # assign actuals: history first, then future overwrite where applicable
            if not hist_series.empty:
                df_out.loc[hist_series.index, 'actual'] = hist_series.values
            if not fut_series.empty:
                df_out.loc[fut_series.index, 'actual'] = fut_series.values

            # assign predictions
            df_out.loc[pred_series.index, 'predicted'] = pred_series.values

            df_out = df_out.reset_index().rename(columns={'index': 'timestamp'})
            df_out = df_out.sort_values('timestamp')
            csv_out = os.path.join(output_dir, f"{model_name}_series_{plotted}.csv")
            df_out.to_csv(csv_out, index=False)
        except Exception:
            # non-fatal: plotting should not crash the full run
            pass
        plotted += 1


def plot_metrics_summary(output_dir: str, model_name: str, metrics_res):
    _ensure_dir(output_dir)
    # metrics_res is a pandas Series-like mapping: index->value per metric
    try:
        import pandas as pd
    except Exception:
        return
    # make a simple bar plot for a subset of metrics
    keys = [k for k in ["MSE[mean]", "MAE[0.5]", "MAPE[0.5]", "sMAPE[0.5]"] if metrics_res.get(k) is not None]
    if not keys:
        return
    values = [metrics_res.get(k).iloc[0] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(keys, values)
    ax.set_title(f"{model_name} metrics")
    ax.set_xticklabels(keys, rotation=30, ha='right')
    fig.tight_layout()
    outpath = os.path.join(output_dir, f"{model_name}_metrics.png")
    fig.savefig(outpath)
    plt.close(fig)


def plot_compare_metrics(output_dir: str, metrics_by_model: dict):
    """Plot a side-by-side comparison of selected metrics for multiple models.

    metrics_by_model: mapping model_name -> metrics_res (pandas Series-like)
    """
    _ensure_dir(output_dir)
    keys = ["MSE[mean]", "MAE[0.5]", "MAPE[0.5]", "sMAPE[0.5]", "MSIS"]
    models = list(metrics_by_model.keys())
    vals = {k: [] for k in keys}
    for m in models:
        res = metrics_by_model.get(m, {})
        for k in keys:
            try:
                v = res.get(k).iloc[0] if res.get(k) is not None else float('nan')
            except Exception:
                v = float('nan')
            vals[k].append(v)

    # create grouped bar chart
    x = np.arange(len(models))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), 3))
    for i, k in enumerate(keys):
        ax.bar(x + i * width, vals[k], width, label=k)
    ax.set_xticks(x + width * (len(keys) - 1) / 2)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.legend(fontsize='small')
    fig.tight_layout()
    outpath = os.path.join(output_dir, 'compare_metrics.png')
    fig.savefig(outpath)
    plt.close(fig)
