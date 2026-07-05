from flask import Flask, request, jsonify
import os
import pickle
import base64
import numpy as np
import math
import threading
import time
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

training_jobs: dict = {}        # job_id -> live status dict
prediction_jobs: dict = {}      # job_id -> live status dict
historical_load_jobs: dict = {} # job_id -> live status dict

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')
INTERNAL_SECRET = os.environ.get('XGB_INTERNAL_SECRET', '')

MODEL_FEATURE_NAMES = {
    'tendencia':      ['vs20','vs50','vs200','adx_norm','rsi_norm','macd_norm'],
    'momentum':       ['rsi_norm','macd_norm','roc5','roc10','roc20'],
    'volatilidad':    ['bb_pos','bb_squeeze','atr_norm','hv_norm'],
    'volumen':        ['obv_dir','roc5','candle','vs20'],
    'estructura':     ['vs50','vs200','bb_pos','adx_dir'],
    'elliott':        ['vs20','roc10','roc20','bb_pos'],
    'velas':          ['candle','vs20','rsi_norm','bb_pos'],
    'macro':          ['roc20','vs200','rsi_norm','bb_pos'],
    'fundamental':    ['roc20','vs200','rsi_norm','vs50'],
    'sentimiento':    ['rsi_norm','bb_pos','roc5','candle'],
    'regresion':      ['roc5','roc10','roc20','vs20'],
    'reversion':      ['neg_vs20','neg_bb','neg_rsi','neg_vs50'],
    'divergencias':   ['rsi_norm','macd_norm','roc5','obv_dir'],
    'estacionalidad': ['sin_month','cos_month','rsi_norm','roc20','vs200'],
    'beta_mercado':   ['roc5','roc10','roc20','vs200','rsi_norm'],
    'fuerza_relativa':['roc5','roc10','roc20','vs50'],
}

MIN_MOVE_PCT  = {7: 0.3, 14: 0.5, 30: 0.8, 60: 1.2, 90: 1.5}
HORIZON_BUCKETS = [7, 14, 30, 60, 90]


def cl(v, scale):
    return max(-1.0, min(1.0, float(v or 0) / scale)) if scale else 0.0


def cl3(v, lo=-3.0, hi=3.0):
    return max(lo, min(hi, float(v or 0)))


# ── Vectorized feature computation (pandas, per asset) ───────────────────────

def compute_all_features_for_asset(rows):
    """
    Returns {date_str: feature_dict} for all dates with enough history.
    Uses pandas rolling windows — O(n) per feature, no per-row slicing.
    """
    import pandas as pd

    if len(rows) < 21:
        return {}

    dates = [r['trade_date'] for r in rows]

    c = pd.Series(
        [float(r.get('adj_close') or r.get('close') or 0) for r in rows]
    ).replace(0, float('nan')).ffill().bfill()

    h = pd.Series([float(r.get('high')   or 0) for r in rows]).where(lambda s: s > 0, c)
    l = pd.Series([float(r.get('low')    or 0) for r in rows]).where(lambda s: s > 0, c)
    o = pd.Series([float(r.get('open')   or 0) for r in rows]).where(lambda s: s > 0, c)
    v = pd.Series([float(r.get('volume') or 0) for r in rows])

    # ── SMA ratios ────────────────────────────────────────────────────────────
    sma20  = c.rolling(20,  min_periods=5).mean()
    sma50  = c.rolling(50,  min_periods=20).mean()
    sma200 = c.rolling(200, min_periods=50).mean()

    vs20  = ((c - sma20)  / sma20  * 100).clip(-20, 20) / 20
    vs50  = ((c - sma50)  / sma50  * 100).clip(-20, 20) / 20
    vs200 = ((c - sma200) / sma200 * 100).clip(-30, 30) / 30

    # ── RSI ───────────────────────────────────────────────────────────────────
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14, min_periods=5).mean()
    loss  = (-delta.clip(upper=0)).rolling(14, min_periods=5).mean()
    rsi   = (100 - 100 / (1 + gain / loss.replace(0, 1e-10))).fillna(50)
    rsi_norm = (rsi - 50) / 25.0

    # ── ROC ───────────────────────────────────────────────────────────────────
    roc5  = (c.pct_change(5)  * 100).clip(-20, 20) / 20
    roc10 = (c.pct_change(10) * 100).clip(-30, 30) / 30
    roc20 = (c.pct_change(20) * 100).clip(-40, 40) / 40

    # ── MACD (EMA12 − EMA26) ─────────────────────────────────────────────────
    ema12     = c.ewm(span=12, adjust=False).mean()
    ema26     = c.ewm(span=26, adjust=False).mean()
    macd_h    = ema12 - ema26
    macd_norm = (macd_h / (macd_h.abs() + 0.01)).clip(-3, 3) * 0.33

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    std20    = c.rolling(20, min_periods=5).std()
    bb_range = (2 * std20 * 2).replace(0, float('nan'))
    bb_pct_b = ((c - (sma20 - 2 * std20)) / bb_range).clip(0, 1).fillna(0.5)
    bb_pos   = bb_pct_b * 2 - 1
    avg_std  = std20.rolling(20, min_periods=10).mean()
    bb_squeeze = ((std20 < 0.75 * avg_std) & avg_std.notna()).astype(float)

    # ── ATR ───────────────────────────────────────────────────────────────────
    prev_c = c.shift(1)
    tr     = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr    = tr.rolling(14, min_periods=5).mean()
    atr_norm = (atr / c * 100).clip(0, 5) / 5

    # ── Historical Volatility ─────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1))
    hv      = log_ret.rolling(20, min_periods=10).std() * math.sqrt(252) * 100
    hv_norm = hv.clip(0, 100) / 100

    # ── OBV direction (5-day) ─────────────────────────────────────────────────
    obv_dir = np.sign((v * np.sign(delta).fillna(0)).rolling(5, min_periods=1).sum()).fillna(0)

    # ── Candle ───────────────────────────────────────────────────────────────
    candle = np.sign(c - o).replace(0, -1).fillna(-1)

    # ── ADX (simplified 14-day) ───────────────────────────────────────────────
    pdm   = (h - h.shift(1)).clip(lower=0)
    mdm   = (l.shift(1) - l).clip(lower=0)
    atr14 = tr.rolling(14, min_periods=5).mean().replace(0, float('nan'))
    pdi   = (100 * pdm.rolling(14, min_periods=5).mean() / atr14).fillna(0)
    mdi   = (100 * mdm.rolling(14, min_periods=5).mean() / atr14).fillna(0)
    dx    = (100 * (pdi - mdi).abs() / (pdi + mdi + 1e-10)).clip(0, 100)
    adx_norm = (dx.clip(0, 50) / 50 - 0.4).fillna(-0.4)

    # ── Seasonality ───────────────────────────────────────────────────────────
    months    = [datetime.strptime(d, '%Y-%m-%d').month if d else 6 for d in dates]
    sin_month = pd.Series([math.sin(2 * math.pi * m / 12) for m in months])
    cos_month = pd.Series([math.cos(2 * math.pi * m / 12) for m in months])

    def _f(series, i, default=0.0):
        v = series.iloc[i]
        return float(v) if v == v else default  # NaN check

    result = {}
    for i in range(21, len(rows)):
        close_val = c.iloc[i]
        if close_val != close_val or close_val <= 0:
            continue
        vs20_v  = _f(vs20, i)
        vs50_v  = _f(vs50, i)
        bb_pos_v = _f(bb_pos, i)
        rsi_v   = _f(rsi_norm, i)
        feat = {
            'vs20':       vs20_v,
            'vs50':       vs50_v,
            'vs200':      _f(vs200, i),
            'rsi_norm':   rsi_v,
            'macd_norm':  _f(macd_norm, i),
            'bb_pos':     bb_pos_v,
            'bb_squeeze': _f(bb_squeeze, i),
            'atr_norm':   _f(atr_norm, i, 0.2),
            'hv_norm':    _f(hv_norm, i, 0.2),
            'obv_dir':    _f(obv_dir, i),
            'roc5':       _f(roc5, i),
            'roc10':      _f(roc10, i),
            'roc20':      _f(roc20, i),
            'candle':     _f(candle, i, -1.0),
            'adx_norm':   _f(adx_norm, i, -0.4),
            'adx_dir':    _f(adx_norm, i, -0.4),
            'neg_vs20':   -vs20_v,
            'neg_bb':     -bb_pos_v,
            'neg_rsi':    -rsi_v,
            'neg_vs50':   -vs50_v,
            'sin_month':  float(sin_month.iloc[i]),
            'cos_month':  float(cos_month.iloc[i]),
        }
        result[dates[i]] = feat

    return result


# ── Extract features from indicators row (daily prediction) ──────────────────

def extract_features(ind):
    vs20  = cl(ind.get('price_vs_sma20', 0), 20)
    vs50  = cl(ind.get('price_vs_sma50', 0), 20)
    vs200 = cl(ind.get('price_vs_sma200', 0), 30)
    rsi   = float(ind.get('rsi_14', 50) or 50)
    rsi_norm = (rsi - 50) / 25.0

    macd_h = ind.get('macd_histogram')
    if macd_h is None:
        macd_h = 0.1 if ind.get('macd_signal') == 'bullish_cross' else -0.1
    macd_h = float(macd_h)
    macd_norm = cl3(macd_h / (abs(macd_h) + 0.01)) * 0.33

    bb_b      = float(ind.get('bb_pct_b', 0.5) or 0.5)
    bb_pos    = bb_b * 2 - 1
    bb_squeeze = 1.0 if ind.get('bb_squeeze') else 0.0
    atr_norm  = cl(ind.get('atr_pct', 1), 5)
    hv_norm   = cl(ind.get('hist_vol_20', 20), 100)

    obv_t   = ind.get('obv_trend', '')
    obv_dir = 1.0 if obv_t == 'rising' else (-1.0 if obv_t == 'falling' else 0.0)

    roc5  = cl(ind.get('roc_5', 0), 20)
    roc10 = cl(ind.get('roc_10', 0), 30)
    roc20 = cl(ind.get('roc_20', 0), 40)

    candle   = 1.0 if ind.get('candle_signal') == 'bullish' else -1.0
    adx_norm = cl(ind.get('adx_14', 20), 50) - 0.4

    dt_str = ind.get('computed_date', '')
    try:
        month = datetime.strptime(dt_str, '%Y-%m-%d').month if dt_str else 6
    except Exception:
        month = 6
    sin_month = math.sin(2 * math.pi * month / 12)
    cos_month = math.cos(2 * math.pi * month / 12)

    return {
        'vs20': vs20, 'vs50': vs50, 'vs200': vs200,
        'rsi_norm': rsi_norm, 'macd_norm': macd_norm,
        'bb_pos': bb_pos, 'bb_squeeze': bb_squeeze,
        'atr_norm': atr_norm, 'hv_norm': hv_norm,
        'obv_dir': obv_dir,
        'roc5': roc5, 'roc10': roc10, 'roc20': roc20,
        'candle': candle,
        'adx_norm': adx_norm, 'adx_dir': adx_norm,
        'neg_vs20': -vs20, 'neg_bb': -bb_pos,
        'neg_rsi': -rsi_norm, 'neg_vs50': -vs50,
        'sin_month': sin_month, 'cos_month': cos_month,
    }


# ── Training ─────────────────────────────────────────────────────────────────

def _fetch_asset_rows(sb) -> dict:
    """Fetch all price_history rows and return {asset_id: [rows]} sorted by date.
    Called once and reused when training multiple models."""
    all_rows: list = []
    offset = 0
    PAGE = 1000
    while True:
        print(f'[train] fetching price_history offset={offset}', flush=True)
        try:
            resp = sb.table('price_history').select(
                'asset_id,trade_date,open,high,low,close,volume,adj_close'
            ).order('trade_date').range(offset, offset + PAGE - 1).execute()
            rows = resp.data or []
            print(f'[train] got {len(rows)} rows at offset={offset}', flush=True)
        except Exception as e:
            print(f'[train] FETCH ERROR at offset={offset}: {e}', flush=True)
            break
        all_rows.extend(rows)
        if len(rows) < PAGE:
            break
        offset += PAGE

    print(f'[train] total rows fetched: {len(all_rows)}', flush=True)

    if not all_rows:
        raise ValueError('No price history data found')

    asset_rows: dict = defaultdict(list)
    for row in all_rows:
        if not isinstance(row['trade_date'], str):
            row['trade_date'] = str(row['trade_date'])
        asset_rows[row['asset_id']].append(row)
    for aid in asset_rows:
        asset_rows[aid].sort(key=lambda r: r['trade_date'])

    print(f'[train] assets grouped: {len(asset_rows)}', flush=True)
    return dict(asset_rows)


def _run_training_background(job_id: str, asset_rows: dict):
    """Called in a daemon thread — updates training_jobs[job_id] live."""
    import gc
    import traceback
    # Pre-load xgboost + sklearn NOW while RAM is plentiful.
    # If we wait until mid-loop, OOM can silently set SKLEARN_INSTALLED=False
    # inside xgboost.sklearn, breaking XGBClassifier for the rest of the run.
    import xgboost as xgb
    import sklearn.base  # force sklearn fully into memory before training starts
    gc.collect()

    job = training_jobs[job_id]
    model_names = list(MODEL_FEATURE_NAMES.keys())
    model_times: list = []

    for i, mn in enumerate(model_names):
        model_start = time.time()
        job['current_model'] = mn
        job['models_done'] = i
        try:
            r = train_model(mn, asset_rows=asset_rows)
            job['results'][mn] = r.get('buckets', {})
        except Exception as e:
            tb = traceback.format_exc()
            error_msg = f'Error en modelo "{mn}": {type(e).__name__}: {e}\n\n{tb}'
            print(f'[train_all] FATAL — stopping at model={mn}:\n{tb}', flush=True)
            job['status'] = 'error'
            job['error'] = error_msg
            job['failed_model'] = mn
            job['models_done'] = i
            job['current_model'] = None
            return  # abort entire run
        model_times.append(time.time() - model_start)
        job['models_done'] = i + 1
        avg = sum(model_times) / len(model_times)
        job['estimated_remaining'] = int(avg * (len(model_names) - (i + 1)))
        gc.collect()  # free numpy arrays + pandas frames from previous model

    job['status'] = 'done'
    job['current_model'] = None
    job['estimated_remaining'] = 0
    print(f'[train_all] DONE all {len(model_names)} models', flush=True)


def train_model(model_name: str, asset_rows: dict = None) -> dict:
    from supabase import create_client
    import xgboost as xgb

    print(f'[train] START model={model_name}', flush=True)

    if model_name not in MODEL_FEATURE_NAMES:
        raise ValueError(f'Unknown model: {model_name}')

    feature_names = MODEL_FEATURE_NAMES[model_name]
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch data if not pre-supplied (single-model path)
    if asset_rows is None:
        print(f'[train] fetching data (single-model path)', flush=True)
        asset_rows = _fetch_asset_rows(sb)

    # Single pass over assets: compute features + build bucket training data directly
    # Avoids caching 37k feature dicts in memory simultaneously
    bucket_X: dict = {b: [] for b in HORIZON_BUCKETS}
    bucket_y: dict = {b: [] for b in HORIZON_BUCKETS}

    for aid, rows in asset_rows.items():
        feats_by_date = compute_all_features_for_asset(rows)
        if not feats_by_date:
            continue

        d2c: dict = {}
        for r in rows:
            close_p = float(r.get('adj_close') or r.get('close') or 0)
            if close_p > 0:
                d2c[r['trade_date']] = close_p

        for date_str, feats in feats_by_date.items():
            close_p = d2c.get(date_str, 0)
            if close_p <= 0:
                continue
            try:
                dt = datetime.strptime(date_str, '%Y-%m-%d')
            except Exception:
                continue

            for bucket in HORIZON_BUCKETS:
                target = dt + timedelta(days=int(bucket * 1.45))
                future = None
                for delta in range(-3, 8):
                    check = (target + timedelta(days=delta)).strftime('%Y-%m-%d')
                    fp = d2c.get(check, 0)
                    if fp > 0:
                        future = fp
                        break
                if future is None:
                    continue
                pct = (future - close_p) / close_p * 100
                bucket_X[bucket].append([feats.get(f, 0.0) for f in feature_names])
                bucket_y[bucket].append(1 if pct >= MIN_MOVE_PCT[bucket] else 0)

    for b in HORIZON_BUCKETS:
        print(f'[train] bucket={b} samples={len(bucket_X[b])}', flush=True)

    bucket_results = {}
    for bucket in HORIZON_BUCKETS:
        X_rows = bucket_X[bucket]
        y_rows = bucket_y[bucket]

        if len(X_rows) < 50:
            bucket_results[bucket] = {'skipped': True, 'samples': len(X_rows)}
            continue

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)

        pos_rate = float(y.mean())
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0.01 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
        )
        model.fit(X, y)
        preds      = model.predict(X)
        train_acc  = float((preds == y).mean())
        del preds, X, y  # free large arrays before Supabase upsert
        model_b64  = base64.b64encode(pickle.dumps(model)).decode()
        del model  # free trained model from RAM (already serialised to b64)

        # Fetch previous accuracy — only overwrite model if new one is strictly better
        try:
            old_row = sb.table('xgb_models').select(
                'train_accuracy,train_samples'
            ).eq('model_name', model_name).eq('horizon_bucket', bucket).maybe_single().execute()
            old_acc  = float(old_row.data['train_accuracy']) if old_row.data and old_row.data.get('train_accuracy') is not None else None
            old_samp = int(old_row.data['train_samples'])    if old_row.data and old_row.data.get('train_samples')   is not None else None
        except Exception:
            old_acc, old_samp = None, None

        improved = old_acc is None or train_acc > old_acc

        if improved:
            sb.table('xgb_models').upsert({
                'model_name':     model_name,
                'horizon_bucket': bucket,
                'model_data':     model_b64,
                'feature_names':  feature_names,
                'train_accuracy': train_acc,
                'train_samples':  len(X_rows),
            }, on_conflict='model_name,horizon_bucket').execute()
            _old_s = f'{old_acc:.4f}' if old_acc else 'new'
            print(f'[train] {model_name}/{bucket}d UPDATED {_old_s} → {train_acc:.4f}', flush=True)
        else:
            print(f'[train] {model_name}/{bucket}d KEPT old={old_acc:.4f} >= new={train_acc:.4f}', flush=True)

        del model_b64  # no longer needed

        # Always record the attempt in history
        try:
            sb.table('xgb_training_history').insert({
                'model_name':     model_name,
                'horizon_bucket': bucket,
                'old_accuracy':   old_acc,
                'new_accuracy':   round(train_acc, 6),
                'old_samples':    old_samp,
                'new_samples':    len(X_rows),
            }).execute()
        except Exception:
            pass  # history is non-critical

        bucket_results[bucket] = {
            'samples':       len(X_rows),
            'accuracy':      round(train_acc, 4),
            'pos_rate':      round(pos_rate, 4),
            'old_accuracy':  round(old_acc, 4) if old_acc is not None else None,
            'delta':         round(train_acc - old_acc, 4) if old_acc is not None else None,
            'improved':      improved,
        }

    return {'model_name': model_name, 'buckets': bucket_results}


# ── Prediction ────────────────────────────────────────────────────────────────

def run_predictions(progress_cb=None) -> dict:
    """Generate XGBoost predictions for all assets.

    Processes one model_name at a time (5 horizons) to cap peak RAM usage.
    Loading all 80 models at once (~100 MB) caused OOM on Render free tier.
    """
    import gc
    from supabase import create_client

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # ── 1. Fetch indicators (small, no model blobs) ───────────────────────────
    cols = (
        'asset_id,computed_date,price_close,'
        'price_vs_sma20,price_vs_sma50,price_vs_sma200,'
        'rsi_14,macd_histogram,macd_signal,bb_pct_b,bb_squeeze,'
        'atr_pct,hist_vol_20,obv_trend,roc_5,roc_10,roc_20,'
        'candle_signal,adx_14,assets(ticker,is_active)'
    )
    today      = datetime.now().strftime('%Y-%m-%d')
    indicators = sb.table('indicators').select(cols).eq('computed_date', today).execute().data or []
    if not indicators:
        yesterday  = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        indicators = sb.table('indicators').select(cols).eq('computed_date', yesterday).execute().data or []
    if not indicators:
        return {'predictions': 0, 'reason': 'No indicator data for today'}

    pred_date  = indicators[0].get('computed_date', today)
    model_names = list(MODEL_FEATURE_NAMES.keys())
    total_preds = 0

    # ── 2. One model_name at a time (5 horizons) → predict → free → repeat ───
    for idx, model_name in enumerate(model_names):
        if progress_cb:
            progress_cb(model_name, idx, len(model_names))

        model_rows = sb.table('xgb_models').select(
            'horizon_bucket,model_data,feature_names'
        ).eq('model_name', model_name).execute().data or []

        if not model_rows:
            print(f'[predict] no trained model for {model_name}, skipping', flush=True)
            continue

        # Load 5 horizon models for this model_name
        loaded: dict = {}
        for row in model_rows:
            bucket = int(row['horizon_bucket'])
            loaded[bucket] = {
                'model':         pickle.loads(base64.b64decode(row['model_data'])),
                'feature_names': row['feature_names'],
            }
        del model_rows  # free raw b64 blobs

        batch = []
        for ind_row in indicators:
            ticker = (ind_row.get('assets') or {}).get('ticker', '')
            if not ticker:
                continue
            feats_all = extract_features(ind_row)
            for bucket, info in loaded.items():
                x = np.array([[feats_all.get(f, 0.0) for f in info['feature_names']]], dtype=np.float32)
                prob_up = float(info['model'].predict_proba(x)[0][1])
                batch.append({
                    'ticker':          ticker,
                    'model_name':      model_name,
                    'horizon_bucket':  bucket,
                    'probability_up':  round(prob_up, 6),
                    'prediction_date': pred_date,
                })

        del loaded  # free 5 XGBoost models before upsert
        gc.collect()

        CHUNK = 500
        for i in range(0, len(batch), CHUNK):
            sb.table('xgb_daily_predictions').upsert(
                batch[i:i + CHUNK],
                on_conflict='ticker,model_name,horizon_bucket,prediction_date'
            ).execute()
        total_preds += len(batch)
        del batch
        print(f'[predict] {model_name} done ({idx + 1}/{len(model_names)})', flush=True)

    if progress_cb:
        progress_cb(None, len(model_names), len(model_names))

    return {
        'predictions': total_preds,
        'date':        pred_date,
        'assets':      len(indicators),
        'models':      len(model_names),
    }


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


def _check_secret() -> bool:
    secret = request.headers.get('x-internal-secret', '')
    return not (INTERNAL_SECRET and secret != INTERNAL_SECRET)


@app.after_request
def _cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'content-type, authorization, x-internal-secret'
    return response


@app.route('/api/train_xgb', methods=['POST', 'OPTIONS'])
def train():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body       = request.get_json(silent=True) or {}
    model_name = body.get('model_name', 'tendencia')
    try:
        result = train_model(model_name)
        return jsonify({'ok': True, **result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/train_xgb_all', methods=['POST', 'OPTIONS'])
def train_all():
    """Start background training of all 16 models. Returns job_id immediately."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job_id = str(uuid.uuid4())[:12]
    training_jobs[job_id] = {
        'status': 'fetching',
        'current_model': None,
        'models_done': 0,
        'models_total': len(MODEL_FEATURE_NAMES),
        'results': {},
        'start_time': time.time(),
        'estimated_remaining': None,
        'error': None,
    }

    def run():
        from supabase import create_client
        job = training_jobs[job_id]
        try:
            sb = create_client(SUPABASE_URL, SUPABASE_KEY)
            print('[train_all] fetching price data once for all models', flush=True)
            asset_rows = _fetch_asset_rows(sb)
            job['status'] = 'training'
            _run_training_background(job_id, asset_rows)
        except Exception as e:
            job['status'] = 'error'
            job['error'] = str(e)
            print(f'[train_all] FATAL ERROR: {e}', flush=True)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/api/train_status/<job_id>', methods=['GET', 'OPTIONS'])
def train_status_endpoint(job_id):
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job = training_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'Job not found (process may have restarted)'}), 404

    elapsed = int(time.time() - job['start_time'])
    return jsonify({
        'ok': True,
        'status': job['status'],
        'current_model': job.get('current_model'),
        'models_done': job.get('models_done', 0),
        'models_total': job.get('models_total', len(MODEL_FEATURE_NAMES)),
        'elapsed': elapsed,
        'estimated_remaining': job.get('estimated_remaining'),
        'results': job.get('results') if job['status'] == 'done' else None,
        'error': job.get('error'),
        'failed_model': job.get('failed_model'),
    })


@app.route('/api/predict_xgb', methods=['POST', 'OPTIONS'])
def predict():
    """Start background XGBoost predictions. Returns job_id immediately."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job_id = str(uuid.uuid4())[:12]
    prediction_jobs[job_id] = {
        'status': 'running',
        'current_model': None,
        'models_done': 0,
        'models_total': len(MODEL_FEATURE_NAMES),
        'start_time': time.time(),
        'result': None,
        'error': None,
    }

    def run():
        import traceback
        job = prediction_jobs[job_id]
        def cb(mn, done, total):
            job['current_model'] = mn
            job['models_done']   = done
            job['models_total']  = total
        try:
            result = run_predictions(progress_cb=cb)
            job['status'] = 'done'
            job['result'] = result
        except Exception as e:
            job['status'] = 'error'
            job['error']  = f'{type(e).__name__}: {e}\n\n{traceback.format_exc()}'
            print(f'[predict] FATAL: {e}', flush=True)

    threading.Thread(target=run, daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/api/predict_status/<job_id>', methods=['GET', 'OPTIONS'])
def predict_status_endpoint(job_id):
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job = prediction_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'Job not found (process may have restarted)'}), 404

    return jsonify({
        'ok': True,
        'status': job['status'],
        'current_model': job.get('current_model'),
        'models_done': job.get('models_done', 0),
        'models_total': job.get('models_total', len(MODEL_FEATURE_NAMES)),
        'elapsed': int(time.time() - job['start_time']),
        'result': job.get('result'),
        'error': job.get('error'),
    })


# ── LR Intraday Training ──────────────────────────────────────────────────────

LR_FEATURE_NAMES = [
    # Core scores (score_divergencias dropped: avg importance 7 vs 227 for top features)
    'score_tendencia', 'score_momentum', 'score_volatilidad', 'score_volumen',
    'score_estructura', 'score_velas', 'score_regresion', 'score_reversion',
    'score_beta_mercado', 'score_vwap', 'score_apertura', 'score_horario',
    # Technical indicators
    'rsi_7', 'price_vs_vwap', 'bb_pct_b', 'volume_ratio',
    'momentum_15m', 'momentum_30m', 'momentum_60m', 'atr_pct',
    # Timing
    'minutes_since_open', 'minutes_to_close',
    # SPY context + session setup (activated: >500 live samples as of 2026-07-03)
    'spy_return_15m', 'premarket_gap', 'prev_day_return',
    # spy_return_session: COALESCE to 0 — zero-variance until ~3 weeks post calculador fix (2026-07-03)
    'spy_return_session',
    # peer_momentum_15m: avg momentum_15m of cluster peers (5-min lag) — zero-variance until calculador v10
    'peer_momentum_15m',
]

lr_training_jobs: dict = {}

TICKER_CLUSTERS: dict = {
    # high_beta: speculative/volatile growth — momentum-driven
    'NVDA': 'high_beta', 'TSLA': 'high_beta', 'AMD': 'high_beta',
    'META': 'high_beta', 'SMCI': 'high_beta', 'COIN': 'high_beta',
    'MSTR': 'high_beta', 'PLTR': 'high_beta', 'AFRM': 'high_beta',
    'HOOD': 'high_beta', 'SOFI': 'high_beta', 'SQ': 'high_beta',
    'IONQ': 'high_beta', 'RGTI': 'high_beta', 'RKLB': 'high_beta',
    'LUNR': 'high_beta', 'RXRX': 'high_beta', 'BEAM': 'high_beta',
    'SNAP': 'high_beta', 'PLUG': 'high_beta', 'CRWD': 'high_beta',
    'FSLR': 'high_beta', 'ARM': 'high_beta',
    # mega_tech: large-cap tech + semiconductor
    'AAPL': 'mega_tech', 'MSFT': 'mega_tech', 'GOOGL': 'mega_tech',
    'AMZN': 'mega_tech', 'NFLX': 'mega_tech', 'AVGO': 'mega_tech',
    'ORCL': 'mega_tech', 'MRVL': 'mega_tech', 'ADBE': 'mega_tech',
    'CRM': 'mega_tech',  'QCOM': 'mega_tech', 'INTC': 'mega_tech',
    'IBM': 'mega_tech',  'TXN': 'mega_tech',  'CSCO': 'mega_tech',
    'UBER': 'mega_tech', 'PYPL': 'mega_tech', 'LLY': 'mega_tech',
    'MELI': 'mega_tech',
    # financials: banks + payment networks
    'JPM': 'financials', 'BAC': 'financials', 'GS': 'financials',
    'MS': 'financials',  'WFC': 'financials', 'AXP': 'financials',
    'MA': 'financials',  'V': 'financials',   'BLK': 'financials',
    'GGAL': 'financials',
    # defensive: healthcare, consumer, industrial, energy
    'ABBV': 'defensive', 'AMGN': 'defensive', 'MRK': 'defensive',
    'PFE': 'defensive',  'JNJ': 'defensive',  'TMO': 'defensive',
    'UNH': 'defensive',  'KO': 'defensive',   'PEP': 'defensive',
    'WMT': 'defensive',  'COST': 'defensive', 'MCD': 'defensive',
    'SBUX': 'defensive', 'HD': 'defensive',   'PG': 'defensive',
    'NKE': 'defensive',  'BA': 'defensive',   'CAT': 'defensive',
    'GE': 'defensive',   'HON': 'defensive',  'RTX': 'defensive',
    'LMT': 'defensive',  'UPS': 'defensive',  'ETN': 'defensive',
    'XOM': 'defensive',  'CVX': 'defensive',  'COP': 'defensive',
    'OXY': 'defensive',  'BRK-B': 'defensive', 'YPF': 'defensive',
    # macro_etf: ETFs and cross-asset benchmarks
    'SPY': 'macro_etf', 'QQQ': 'macro_etf', 'IWM': 'macro_etf',
    'DIA': 'macro_etf', 'GLD': 'macro_etf', 'GDX': 'macro_etf',
    'EEM': 'macro_etf', 'TLT': 'macro_etf', 'IEF': 'macro_etf',
    'HYG': 'macro_etf', 'USO': 'macro_etf',
}


def apply_beta_adj(y_arr, spy_arr, beta):
    out = y_arr.copy()
    valid = ~np.isnan(spy_arr) & ~np.isnan(y_arr)
    out[valid] -= beta * spy_arr[valid]
    return out


def _sync_earnings_calendar():
    """Fetch earnings dates for all active intraday tickers via yfinance and upsert to DB."""
    import yfinance as yf
    from supabase import create_client

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('assets').select('ticker').eq('is_active', True).eq('intraday_active', True).execute()
    tickers = [r['ticker'] for r in (resp.data or [])]

    upserts = []
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            ed = t.earnings_dates  # DataFrame index = earnings datetime (past + upcoming)
            if ed is not None and len(ed) > 0:
                for dt in ed.index:
                    try:
                        report_date = dt.date().isoformat()
                        upserts.append({'ticker': ticker, 'report_date': report_date, 'source': 'yfinance'})
                    except Exception:
                        pass
        except Exception as e:
            print(f'[earnings_sync] {ticker}: {e}', flush=True)

    if upserts:
        sb.table('earnings_calendar').upsert(upserts, on_conflict='ticker,report_date').execute()

    print(f'[earnings_sync] upserted {len(upserts)} dates for {len(tickers)} tickers', flush=True)
    return len(upserts)


def _run_lr_training(job_id: str):
    from supabase import create_client
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import lightgbm as lgb
    from datetime import timezone

    job = lr_training_jobs[job_id]
    HALF_LIFE_DAYS = 90
    lam = math.log(2) / HALF_LIFE_DAYS

    def _parse_ts(ts):
        if not ts:
            return datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return ts if getattr(ts, 'tzinfo', None) else ts.replace(tzinfo=timezone.utc)

    def _wf_mae(X, y_signed, y_spy, beta):
        """Walk-forward MAE across 3 chronological folds using Ridge signed model."""
        n = len(X)
        if n < 25:
            return None
        min_tr = max(15, n // 4)
        fold_sz = max(3, (n - min_tr) // 3)
        maes = []
        for fold in range(3):
            t_end = min_tr + fold * fold_sz
            v_end = min(t_end + fold_sz, n)
            if t_end >= n or v_end - t_end < 3:
                break
            yt = apply_beta_adj(y_signed[:t_end], y_spy[:t_end], beta)
            yv = apply_beta_adj(y_signed[t_end:v_end], y_spy[t_end:v_end], beta)
            tr_m = ~np.isnan(yt); v_m = ~np.isnan(yv)
            if tr_m.sum() < 10 or v_m.sum() < 3:
                continue
            sc = StandardScaler()
            Xts = sc.fit_transform(X[:t_end][tr_m])
            Xvs = sc.transform(X[t_end:v_end][v_m])
            reg = Ridge(alpha=1.0)
            reg.fit(Xts, yt[tr_m])
            maes.append(float(np.mean(np.abs(yv[v_m] - reg.predict(Xvs)))))
        return float(np.mean(maes)) if maes else None

    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        job['status'] = 'fetching'
        # PostgREST caps at 1000 rows/request — paginate in 1000-row chunks.
        # SQL function now has 60-day filter + LIMIT 20000, so each query ~700ms.
        # 20 batches × 700ms = ~14s total (vs old 60 × 1.67s = 100s that timed out).
        all_rows = []
        batch_size = 1000
        for offset in range(0, 20001, batch_size):
            resp = sb.rpc('get_intraday_training_data').range(offset, offset + batch_size - 1).execute()
            batch = resp.data or []
            all_rows.extend(batch)
            if len(batch) < batch_size:
                break
        print(f'[lr_train] fetched {len(all_rows)} rows', flush=True)

        job['total_samples'] = len(all_rows)
        if not all_rows:
            job['status'] = 'done'
            job['models_trained'] = 0
            return

        now_utc = datetime.now(timezone.utc)
        all_rows.sort(key=lambda r: _parse_ts(r.get('created_at')))

        # Earnings filter — sync calendar then exclude rows ±2 days from any earnings
        try:
            _sync_earnings_calendar()
            ec_resp = sb.table('earnings_calendar').select('ticker, report_date').execute()
            from collections import defaultdict as _dd
            from datetime import date as _date
            _earn_by_ticker = _dd(list)
            for ec in (ec_resp.data or []):
                try:
                    _earn_by_ticker[ec['ticker']].append(_date.fromisoformat(ec['report_date']))
                except Exception:
                    pass

            def _near_earnings(ticker, created_at_str):
                row_date = _parse_ts(created_at_str).date()
                for earn_d in _earn_by_ticker.get(ticker, []):
                    if abs((row_date - earn_d).days) <= 2:
                        return True
                return False

            pre_filter = len(all_rows)
            all_rows = [r for r in all_rows if not _near_earnings(r.get('ticker', ''), r.get('created_at'))]
            removed = pre_filter - len(all_rows)
            print(f'[lr_train] earnings filter: {pre_filter} → {len(all_rows)} rows ({removed} removed)', flush=True)
        except Exception as e_earn:
            print(f'[lr_train] earnings filter skipped: {e_earn}', flush=True)
        holdout_cutoff = now_utc - timedelta(days=30)

        def decay_w(ts_str):
            age = (now_utc - _parse_ts(ts_str)).total_seconds() / 86400
            return math.exp(-lam * max(0.0, age))

        # Group chronologically by (model_name, horizon_minutes)
        groups: dict = {}
        for row in all_rows:
            key = (row['model_name'], int(row['horizon_minutes']))
            if key not in groups:
                groups[key] = {'X': [], 'y_dir': [], 'y_signed': [], 'y_mag': [], 'y_spy': [], 'w': [], 'ts': []}
            groups[key]['X'].append([float(row.get(fn) or 0) for fn in LR_FEATURE_NAMES])
            groups[key]['y_dir'].append(1 if row['direction_correct'] else 0)
            groups[key]['y_signed'].append(row.get('actual_signed_pct'))
            groups[key]['y_mag'].append(row.get('actual_magnitude'))
            groups[key]['y_spy'].append(row.get('spy_actual_pct'))
            groups[key]['w'].append(decay_w(row.get('created_at')))
            groups[key]['ts'].append(row.get('created_at'))

        job['status'] = 'training'
        job['models_total'] = len(groups)
        upserts = []
        results = {}

        # Cache LGBM per horizon — data is identical across model_names for same horizon,
        # so training 13 separate LGBMs wastes compute and produces identical models.
        lgbm_horizon_cache: dict = {}  # {horizon_minutes: {model_b64, val_mae, importance, beta_spy}}

        for (model_name, horizon_minutes), data in groups.items():
            n = len(data['X'])
            if n < 20:
                continue

            X_np = np.array(data['X'], dtype=float)
            y_dir_np = np.array(data['y_dir'], dtype=float)
            y_signed_np = np.array([float(v) if v is not None else float('nan') for v in data['y_signed']])
            y_mag_np = np.array([float(v) if v is not None else float('nan') for v in data['y_mag']])
            y_spy_np = np.array([float(v) if v is not None else float('nan') for v in data['y_spy']])
            w_np = np.array(data['w'], dtype=float)

            # Walk-forward: holdout = last 30 days (never used for training)
            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in data['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() >= 20:
                X_tv = X_np[tv_mask]; y_dir_tv = y_dir_np[tv_mask]
                y_signed_tv = y_signed_np[tv_mask]; y_mag_tv = y_mag_np[tv_mask]
                y_spy_tv = y_spy_np[tv_mask]; w_tv = w_np[tv_mask]
            else:
                X_tv, y_dir_tv, y_signed_tv, y_mag_tv, y_spy_tv, w_tv = (
                    X_np, y_dir_np, y_signed_np, y_mag_np, y_spy_np, w_np)

            # Compute OLS beta_spy from all tv data (y = beta*spy + idio, minimize variance)
            beta_spy = 0.0
            valid_beta = ~np.isnan(y_signed_tv) & ~np.isnan(y_spy_tv)
            if valid_beta.sum() >= 20:
                y_b = y_signed_tv[valid_beta]; spy_b = y_spy_tv[valid_beta]
                spy_c = spy_b - spy_b.mean()
                denom = float(np.dot(spy_c, spy_c))
                if denom > 1e-10:
                    beta_spy = float(np.clip(
                        np.dot(y_b - y_b.mean(), spy_c) / denom, 0.0, 3.0
                    ))

            wf_val_mae = _wf_mae(X_tv, y_signed_tv, y_spy_tv, beta_spy)

            split = max(10, int(len(X_tv) * 0.8))
            X_train, X_val = X_tv[:split], X_tv[split:]
            y_dir_train = y_dir_tv[:split]
            y_signed_train = apply_beta_adj(y_signed_tv[:split], y_spy_tv[:split], beta_spy)
            y_signed_val   = apply_beta_adj(y_signed_tv[split:], y_spy_tv[split:], beta_spy)
            y_mag_train = y_mag_tv[:split]
            w_train = w_tv[:split]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, X_train.shape[1]))

            # Ridge direction classifier — kept for backward compat with inference edge function
            clf = LogisticRegression(max_iter=100, C=1.0, solver='liblinear')
            clf.fit(X_train_s, y_dir_train, sample_weight=w_train)
            accuracy = float(clf.score(X_train_s, y_dir_train))

            # Ridge magnitude
            mag_coeff = mag_bias_val = mag_r2_val = avg_mag = median_mag = None
            y_mag_tv_valid = y_mag_tv[~np.isnan(y_mag_tv) & (y_mag_tv > 0)]
            if len(y_mag_tv_valid) >= 20:
                avg_mag = float(np.mean(y_mag_tv_valid))
                median_mag = float(np.median(y_mag_tv_valid))
                mag_train_mask = ~np.isnan(y_mag_train) & (y_mag_train > 0)
                if mag_train_mask.sum() >= 10:
                    y_mag_log = np.log(y_mag_train[mag_train_mask] + 0.01)
                    reg_m = Ridge(alpha=1.0)
                    reg_m.fit(X_train_s[mag_train_mask], y_mag_log)
                    mag_r2_val = float(r2_score(y_mag_log, reg_m.predict(X_train_s[mag_train_mask])))
                    mag_coeff = reg_m.coef_.tolist()
                    mag_bias_val = float(reg_m.intercept_)

            # Ridge signed — kept for backward compat
            signed_coeff = signed_bias_val = signed_r2_val = val_mae_ridge = None
            Xs = ys = ws = None
            train_signed_mask = ~np.isnan(y_signed_train)
            if train_signed_mask.sum() >= 20:
                Xs = X_train_s[train_signed_mask]
                ys = y_signed_train[train_signed_mask]
                ws = w_train[train_signed_mask]
                reg_s = Ridge(alpha=1.0)
                reg_s.fit(Xs, ys, sample_weight=ws)
                signed_r2_val = float(r2_score(ys, reg_s.predict(Xs)))
                signed_coeff = reg_s.coef_.tolist()
                signed_bias_val = float(reg_s.intercept_)
                if len(X_val_s) > 0:
                    val_sm = ~np.isnan(y_signed_val)
                    if val_sm.sum() > 0:
                        val_mae_ridge = float(np.mean(np.abs(
                            y_signed_val[val_sm] - reg_s.predict(X_val_s[val_sm])
                        )))

            # LightGBM — train once per horizon with Optuna tuning, reuse across model_names.
            # Target ATR-normalized so model learns in ATR units; inference denormalizes.
            lgbm_model_b64 = lgbm_val_mae = lgbm_importance = None
            if horizon_minutes in lgbm_horizon_cache:
                cached = lgbm_horizon_cache[horizon_minutes]
                lgbm_model_b64 = cached['model_b64']
                lgbm_val_mae = cached['val_mae']
                lgbm_importance = cached['importance']
                beta_spy = cached['beta_spy']
            elif Xs is not None and len(Xs) >= 30:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                atr_idx = LR_FEATURE_NAMES.index('atr_pct')
                atr_train_raw = np.clip(X_train[train_signed_mask][:, atr_idx], 0.1, 10.0)
                ys_norm = ys / atr_train_raw

                val_sm = ~np.isnan(y_signed_val) if len(X_val_s) > 0 else np.zeros(0, dtype=bool)
                has_val = val_sm.sum() >= 5
                if has_val:
                    atr_val_raw = np.clip(X_val[:, atr_idx][val_sm], 0.1, 10.0)
                    y_val_norm = y_signed_val[val_sm] / atr_val_raw

                def _lgbm_objective(trial):
                    params = dict(
                        num_leaves=trial.suggest_int('num_leaves', 15, 127),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
                        max_depth=trial.suggest_int('max_depth', 3, 8),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
                        n_estimators=300, random_state=42, verbose=-1,
                        objective='regression_l1',
                    )
                    m = lgb.LGBMRegressor(**params)
                    m.fit(Xs, ys_norm, sample_weight=ws)
                    if has_val:
                        preds_d = m.predict(X_val_s[val_sm]) * atr_val_raw
                        return float(np.mean(np.abs(y_signed_val[val_sm] - preds_d)))
                    return float(np.mean(np.abs(ys_norm - m.predict(Xs))))

                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                n_trials = 50 if len(Xs) >= 30 else 25
                study.optimize(_lgbm_objective, n_trials=n_trials, show_progress_bar=False)
                best_p = study.best_params
                print(f'[optuna] H={horizon_minutes} best={best_p} val_mae={study.best_value:.4f}', flush=True)

                # Train final model with best params + early stopping
                eval_set_norm = [(X_val_s[val_sm], y_val_norm)] if has_val else None
                callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_norm else None
                lgb_reg = lgb.LGBMRegressor(
                    **best_p, n_estimators=500, random_state=42, verbose=-1,
                    objective='regression_l1',
                )
                lgb_reg.fit(Xs, ys_norm, sample_weight=ws, eval_set=eval_set_norm, callbacks=callbacks)
                if val_sm.sum() > 0:
                    atr_val_raw = np.clip(X_val[:, atr_idx][val_sm], 0.1, 10.0)
                    preds_denorm = lgb_reg.predict(X_val_s[val_sm]) * atr_val_raw
                    lgbm_val_mae = float(np.mean(np.abs(y_signed_val[val_sm] - preds_denorm)))
                lgbm_model_b64 = base64.b64encode(pickle.dumps(lgb_reg)).decode('utf-8')
                lgbm_importance = dict(zip(LR_FEATURE_NAMES, lgb_reg.feature_importances_.tolist()))
                lgbm_horizon_cache[horizon_minutes] = {
                    'model_b64': lgbm_model_b64,
                    'val_mae': lgbm_val_mae,
                    'importance': lgbm_importance,
                    'beta_spy': beta_spy,
                }

            upserts.append({
                'model_name': model_name,
                'horizon_minutes': horizon_minutes,
                'feature_names': LR_FEATURE_NAMES,
                'coefficients': clf.coef_[0].tolist(),
                'bias': float(clf.intercept_[0]),
                'feature_means': scaler.mean_.tolist(),
                'feature_stds': scaler.scale_.tolist(),
                'train_samples': len(X_tv),
                'train_accuracy': accuracy,
                'mag_coefficients': mag_coeff,
                'mag_bias': mag_bias_val,
                'mag_r2': mag_r2_val,
                'avg_actual_mag': avg_mag,
                'median_actual_mag': median_mag,
                'signed_coefficients': signed_coeff,
                'signed_bias': signed_bias_val,
                'signed_r2': signed_r2_val,
                'lgbm_model': lgbm_model_b64,
                'lgbm_val_mae': lgbm_val_mae,
                'lgbm_feature_importance': lgbm_importance,
                'val_mae_ridge': val_mae_ridge,
                'beta_spy': beta_spy,
                'wf_val_mae': wf_val_mae,
            })
            results[f'{model_name}:{horizon_minutes}'] = {
                'samples': len(X_tv), 'accuracy': round(accuracy, 3),
                'avg_mag': round(avg_mag, 3) if avg_mag else None,
                'val_mae_ridge': round(val_mae_ridge, 3) if val_mae_ridge else None,
                'lgbm_val_mae': round(lgbm_val_mae, 3) if lgbm_val_mae else None,
                'wf_val_mae': round(wf_val_mae, 3) if wf_val_mae else None,
            }
            job['models_done'] = len(upserts)
            print(
                f'[lr_train] {model_name}:{horizon_minutes} n={len(X_tv)} acc={accuracy:.3f} '
                f'wf_val_mae={wf_val_mae} lgbm_val_mae={lgbm_val_mae}',
                flush=True,
            )

        for u in upserts:
            sb.rpc('upsert_lr_params', {'p_params': [u]}).execute()

        # ── Step 7: Session-specific LGBM models ─────────────────────────────
        # Train one LGBM per (model_name, horizon_minutes, market_session).
        # Only LightGBM — Ridge stays global for backward compat.
        # Stored in lgbm_session_models_intraday; global model is fallback at inference.
        def _session_from_mso(mso: float) -> str:
            if mso < 30:  return 'open'
            if mso < 120: return 'morning'
            if mso < 270: return 'midday'
            return 'close'

        # Aggregate by (horizon_minutes, session) — pool ALL model_names so small sessions
        # like 'morning' (~44 total rows) still have enough data to train one shared model.
        session_groups: dict = {}    # (horizon_minutes, session) -> data
        session_mnames: dict = {}    # (horizon_minutes, session) -> set of model_names seen
        for row in all_rows:
            mso = float(row.get('minutes_since_open') or 0)
            sess = _session_from_mso(mso)
            key = (int(row['horizon_minutes']), sess)
            if key not in session_groups:
                session_groups[key] = {'X': [], 'y_signed': [], 'y_spy': [], 'w': [], 'ts': [], 'seen': set()}
                session_mnames[key] = set()
            # Deduplicate by (created_at) across model_names — same event counted once
            dedup = row.get('created_at', '')
            if dedup and dedup in session_groups[key]['seen']:
                session_mnames[key].add(row['model_name'])
                continue
            if dedup:
                session_groups[key]['seen'].add(dedup)
            session_groups[key]['X'].append([float(row.get(fn) or 0) for fn in LR_FEATURE_NAMES])
            session_groups[key]['y_signed'].append(row.get('actual_signed_pct'))
            session_groups[key]['y_spy'].append(row.get('spy_actual_pct'))
            session_groups[key]['w'].append(decay_w(row.get('created_at')))
            session_groups[key]['ts'].append(row.get('created_at'))
            session_mnames[key].add(row['model_name'])

        # All known model_names (for filling upsert entries with shared model)
        all_model_names = list(MODEL_FEATURE_NAMES.keys())

        session_upserts = []
        atr_idx = LR_FEATURE_NAMES.index('atr_pct')
        for (horizon_minutes, session), sdata in session_groups.items():
            X_np = np.array(sdata['X'], dtype=float)
            y_np = np.array([float(v) if v is not None else float('nan') for v in sdata['y_signed']])
            spy_np = np.array([float(v) if v is not None else float('nan') for v in sdata['y_spy']])
            w_np = np.array(sdata['w'], dtype=float)

            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in sdata['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() < 20:
                tv_mask = np.ones(len(X_np), dtype=bool)

            X_tv = X_np[tv_mask]; y_tv = y_np[tv_mask]
            spy_tv = spy_np[tv_mask]; w_tv = w_np[tv_mask]

            # Compute per-session OLS beta
            sess_beta = 0.0
            valid_b = ~np.isnan(y_tv) & ~np.isnan(spy_tv)
            if valid_b.sum() >= 20:
                y_b2 = y_tv[valid_b]; spy_b2 = spy_tv[valid_b]
                spy_c2 = spy_b2 - spy_b2.mean()
                d2 = float(np.dot(spy_c2, spy_c2))
                if d2 > 1e-10:
                    sess_beta = float(np.clip(np.dot(y_b2 - y_b2.mean(), spy_c2) / d2, 0.0, 3.0))

            split = max(10, int(len(X_tv) * 0.8))
            X_tr, X_v = X_tv[:split], X_tv[split:]
            y_tr_raw, y_v_raw = y_tv[:split], y_tv[split:]
            spy_tr, spy_v = spy_tv[:split], spy_tv[split:]
            w_tr = w_tv[:split]

            y_tr = apply_beta_adj(y_tr_raw, spy_tr, sess_beta)
            y_v  = apply_beta_adj(y_v_raw,  spy_v,  sess_beta)

            tr_mask = ~np.isnan(y_tr)
            if tr_mask.sum() < 20:
                print(f'[lr_train:session] {horizon_minutes}:{session} only {tr_mask.sum()} train samples — skip', flush=True)
                continue

            scaler_s = StandardScaler()
            Xs_s = scaler_s.fit_transform(X_tr[tr_mask])
            ys_s = y_tr[tr_mask]
            ws_s = w_tr[tr_mask]

            atr_tr = np.clip(X_tr[tr_mask][:, atr_idx], 0.1, 10.0)
            ys_norm_s = ys_s / atr_tr

            X_v_s = scaler_s.transform(X_v) if len(X_v) > 0 else np.empty((0, X_tr.shape[1]))
            val_sm_s = ~np.isnan(y_v) if len(X_v) > 0 else np.zeros(0, dtype=bool)
            eval_set_s = None
            if val_sm_s.sum() >= 5:
                atr_v = np.clip(X_v[:, atr_idx][val_sm_s], 0.1, 10.0)
                eval_set_s = [(X_v_s[val_sm_s], y_v[val_sm_s] / atr_v)]

            cbs = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_s else None
            lgb_s = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1,
                objective='regression_l1',
            )
            lgb_s.fit(Xs_s, ys_norm_s, sample_weight=ws_s, eval_set=eval_set_s, callbacks=cbs)

            sess_val_mae = None
            sess_ep25 = sess_ep50 = sess_ep75 = sess_ep90 = None
            if val_sm_s.sum() > 0:
                atr_v2 = np.clip(X_v[:, atr_idx][val_sm_s], 0.1, 10.0)
                preds_d = lgb_s.predict(X_v_s[val_sm_s]) * atr_v2
                val_residuals_s = np.abs(y_v[val_sm_s] - preds_d)
                sess_val_mae = float(np.mean(val_residuals_s))
                sess_ep25 = float(np.percentile(val_residuals_s, 25))
                sess_ep50 = float(np.percentile(val_residuals_s, 50))
                sess_ep75 = float(np.percentile(val_residuals_s, 75))
                sess_ep90 = float(np.percentile(val_residuals_s, 90))

            lgbm_s_b64 = base64.b64encode(pickle.dumps(lgb_s)).decode('utf-8')
            lgbm_s_imp = dict(zip(LR_FEATURE_NAMES, lgb_s.feature_importances_.tolist()))
            n_train = int(tr_mask.sum())
            print(f'[lr_train:session] {horizon_minutes}min:{session} n={n_train} val_mae={sess_val_mae} beta={sess_beta:.3f}', flush=True)

            # Store one row per model_name (all share the same pooled model)
            mnames_for_key = session_mnames.get((horizon_minutes, session), set()) or set(all_model_names)
            for mn in mnames_for_key:
                session_upserts.append({
                    'model_name': mn,
                    'horizon_minutes': horizon_minutes,
                    'market_session': session,
                    'lgbm_model': lgbm_s_b64,
                    'lgbm_val_mae': sess_val_mae,
                    'lgbm_feature_importance': lgbm_s_imp,
                    'train_samples': n_train,
                    'last_updated': now_utc.isoformat(),
                    'beta_spy': sess_beta,
                    'error_p25': sess_ep25,
                    'error_p50': sess_ep50,
                    'error_p75': sess_ep75,
                    'error_p90': sess_ep90,
                })

        for su in session_upserts:
            sb.table('lgbm_session_models_intraday').upsert(
                su, on_conflict='model_name,horizon_minutes,market_session'
            ).execute()
        print(f'[lr_train] session models: {len(session_upserts)} upserts done', flush=True)

        # ── Step 8: Per-ticker LGBM models ───────────────────────────────────
        # Group unique (ticker, horizon) from all_rows — deduplicated across model_names.
        ticker_horizon_data: dict = {}
        for row in all_rows:
            t = row.get('ticker')
            if not t:
                continue
            tk = (t, int(row['horizon_minutes']))
            if tk not in ticker_horizon_data:
                ticker_horizon_data[tk] = {'X': [], 'y_signed': [], 'y_spy': [], 'w': [], 'ts': [], 'seen': set()}
            # Deduplicate by (created_at) — same timestamp appears once per model_name
            dedup_key = row.get('created_at', '')
            if dedup_key in ticker_horizon_data[tk]['seen']:
                continue
            ticker_horizon_data[tk]['seen'].add(dedup_key)
            ticker_horizon_data[tk]['X'].append([float(row.get(fn) or 0) for fn in LR_FEATURE_NAMES])
            ticker_horizon_data[tk]['y_signed'].append(row.get('actual_signed_pct'))
            ticker_horizon_data[tk]['y_spy'].append(row.get('spy_actual_pct'))
            ticker_horizon_data[tk]['w'].append(decay_w(row.get('created_at')))
            ticker_horizon_data[tk]['ts'].append(row.get('created_at'))

        ticker_upserts = []
        for (ticker, horizon_minutes), tdata in ticker_horizon_data.items():
            n = len(tdata['X'])
            if n < 50:
                continue
            X_np = np.array(tdata['X'], dtype=float)
            y_np = np.array([float(v) if v is not None else float('nan') for v in tdata['y_signed']])
            spy_np = np.array([float(v) if v is not None else float('nan') for v in tdata['y_spy']])
            w_np = np.array(tdata['w'], dtype=float)

            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in tdata['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() < 20:
                tv_mask = np.ones(n, dtype=bool)
            X_tv = X_np[tv_mask]; y_tv = y_np[tv_mask]
            spy_tv = spy_np[tv_mask]; w_tv = w_np[tv_mask]

            # OLS beta per ticker×horizon
            t_beta = 0.0
            vb = ~np.isnan(y_tv) & ~np.isnan(spy_tv)
            if vb.sum() >= 20:
                yb = y_tv[vb]; sb2 = spy_tv[vb]
                sc = sb2 - sb2.mean(); d = float(np.dot(sc, sc))
                if d > 1e-10:
                    t_beta = float(np.clip(np.dot(yb - yb.mean(), sc) / d, 0.0, 3.0))

            split = max(10, int(len(X_tv) * 0.8))
            X_tr, X_v = X_tv[:split], X_tv[split:]
            y_tr = apply_beta_adj(y_tv[:split], spy_tv[:split], t_beta)
            y_v  = apply_beta_adj(y_tv[split:], spy_tv[split:], t_beta)
            w_tr = w_tv[:split]

            tm = ~np.isnan(y_tr)
            if tm.sum() < 20:
                continue

            sc_t = StandardScaler()
            Xs_t = sc_t.fit_transform(X_tr[tm])
            ys_t = y_tr[tm]; ws_t = w_tr[tm]
            atr_t = np.clip(X_tr[tm][:, atr_idx], 0.1, 10.0)
            ys_tn = ys_t / atr_t

            X_v_t = sc_t.transform(X_v) if len(X_v) > 0 else np.empty((0, X_tr.shape[1]))
            vm_t = ~np.isnan(y_v) if len(X_v) > 0 else np.zeros(0, dtype=bool)
            eval_t = None
            if vm_t.sum() >= 5:
                atr_v = np.clip(X_v[:, atr_idx][vm_t], 0.1, 10.0)
                eval_t = [(X_v_t[vm_t], y_v[vm_t] / atr_v)]

            cbs_t = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)] if eval_t else None
            lgb_t = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.05, num_leaves=15,
                min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, objective='regression_l1',
            )
            lgb_t.fit(Xs_t, ys_tn, sample_weight=ws_t, eval_set=eval_t, callbacks=cbs_t)

            t_val_mae = None
            if vm_t.sum() > 0:
                atr_v = np.clip(X_v[:, atr_idx][vm_t], 0.1, 10.0)
                preds_t = lgb_t.predict(X_v_t[vm_t]) * atr_v
                t_val_mae = float(np.mean(np.abs(y_v[vm_t] - preds_t)))

            ticker_upserts.append({
                'ticker': ticker,
                'horizon_minutes': horizon_minutes,
                'lgbm_model': base64.b64encode(pickle.dumps(lgb_t)).decode('utf-8'),
                'lgbm_val_mae': t_val_mae,
                'beta_spy': t_beta,
                'train_samples': int(tm.sum()),
                'last_updated': now_utc.isoformat(),
            })
            print(f'[lr_train:ticker] {ticker}:{horizon_minutes} n={tm.sum()} val_mae={t_val_mae} beta={t_beta:.3f}', flush=True)

        for tu in ticker_upserts:
            sb.table('lgbm_ticker_models_intraday').upsert(
                tu, on_conflict='ticker,horizon_minutes'
            ).execute()
        print(f'[lr_train] ticker models: {len(ticker_upserts)} trained', flush=True)

        # ── Step 9: Cluster LGBM models ───────────────────────────────────────
        # Aggregate rows by (cluster_name, horizon_minutes); deduplicate by (ticker, created_at)
        # so each prediction timestamp counts once per ticker, not 13× per model_name.
        cluster_data: dict = {}
        for row in all_rows:
            t = row.get('ticker', '')
            cluster = TICKER_CLUSTERS.get(t)
            if not cluster:
                continue
            h = int(row['horizon_minutes'])
            ck = (cluster, h)
            if ck not in cluster_data:
                cluster_data[ck] = {'X': [], 'y_signed': [], 'y_spy': [], 'w': [], 'ts': [], 'seen': set()}
            dedup = f"{t}:{row.get('created_at', '')}"
            if dedup in cluster_data[ck]['seen']:
                continue
            cluster_data[ck]['seen'].add(dedup)
            cluster_data[ck]['X'].append([float(row.get(fn) or 0) for fn in LR_FEATURE_NAMES])
            cluster_data[ck]['y_signed'].append(row.get('actual_signed_pct'))
            cluster_data[ck]['y_spy'].append(row.get('spy_actual_pct'))
            cluster_data[ck]['w'].append(decay_w(row.get('created_at')))
            cluster_data[ck]['ts'].append(row.get('created_at'))

        cluster_upserts = []
        cluster_lgbm_cache: dict = {}
        for (cluster_name, horizon_minutes), cdata in cluster_data.items():
            n = len(cdata['X'])
            if n < 50:
                continue
            X_np = np.array(cdata['X'], dtype=float)
            y_np = np.array([float(v) if v is not None else float('nan') for v in cdata['y_signed']])
            spy_np = np.array([float(v) if v is not None else float('nan') for v in cdata['y_spy']])
            w_np = np.array(cdata['w'], dtype=float)

            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in cdata['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() < 20:
                tv_mask = np.ones(n, dtype=bool)
            X_tv = X_np[tv_mask]; y_tv = y_np[tv_mask]
            spy_tv = spy_np[tv_mask]; w_tv = w_np[tv_mask]

            # OLS beta per cluster
            c_beta = 0.0
            vb = ~np.isnan(y_tv) & ~np.isnan(spy_tv)
            if vb.sum() >= 20:
                yb = y_tv[vb]; sb2 = spy_tv[vb]
                sc2 = sb2 - sb2.mean(); d2 = float(np.dot(sc2, sc2))
                if d2 > 1e-10:
                    c_beta = float(np.clip(np.dot(yb - yb.mean(), sc2) / d2, 0.0, 3.0))

            split = max(10, int(len(X_tv) * 0.8))
            X_tr, X_v = X_tv[:split], X_tv[split:]
            y_tr = apply_beta_adj(y_tv[:split], spy_tv[:split], c_beta)
            y_v  = apply_beta_adj(y_tv[split:], spy_tv[split:], c_beta)
            w_tr = w_tv[:split]

            tm = ~np.isnan(y_tr)
            if tm.sum() < 20:
                continue

            sc_c = StandardScaler()
            Xs_c = sc_c.fit_transform(X_tr[tm])
            ys_c = y_tr[tm]; ws_c = w_tr[tm]
            atr_c = np.clip(X_tr[tm][:, atr_idx], 0.1, 10.0)
            ys_cn = ys_c / atr_c

            X_v_c = sc_c.transform(X_v) if len(X_v) > 0 else np.empty((0, X_tr.shape[1]))
            vm_c = ~np.isnan(y_v) if len(X_v) > 0 else np.zeros(0, dtype=bool)
            eval_c = None
            if vm_c.sum() >= 5:
                atr_vc = np.clip(X_v[:, atr_idx][vm_c], 0.1, 10.0)
                eval_c = [(X_v_c[vm_c], y_v[vm_c] / atr_vc)]

            cbs_c = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)] if eval_c else None
            lgb_c = lgb.LGBMRegressor(
                n_estimators=400, learning_rate=0.05, num_leaves=31,
                min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1, objective='regression_l1',
            )
            lgb_c.fit(Xs_c, ys_cn, sample_weight=ws_c, eval_set=eval_c, callbacks=cbs_c)

            c_val_mae = None
            if vm_c.sum() > 0:
                atr_vc2 = np.clip(X_v[:, atr_idx][vm_c], 0.1, 10.0)
                preds_c = lgb_c.predict(X_v_c[vm_c]) * atr_vc2
                c_val_mae = float(np.mean(np.abs(y_v[vm_c] - preds_c)))

            cluster_upserts.append({
                'cluster_name': cluster_name,
                'horizon_minutes': horizon_minutes,
                'lgbm_model': base64.b64encode(pickle.dumps(lgb_c)).decode('utf-8'),
                'lgbm_val_mae': c_val_mae,
                'beta_spy': c_beta,
                'train_samples': int(tm.sum()),
                'lgbm_feature_importance': dict(zip(LR_FEATURE_NAMES, lgb_c.feature_importances_.tolist())),
                'last_updated': now_utc.isoformat(),
            })
            print(f'[lr_train:cluster] {cluster_name}:{horizon_minutes} n={tm.sum()} val_mae={c_val_mae} beta={c_beta:.3f}', flush=True)

        for cu in cluster_upserts:
            sb.table('lgbm_cluster_models_intraday').upsert(
                cu, on_conflict='cluster_name,horizon_minutes'
            ).execute()
        print(f'[lr_train] cluster models: {len(cluster_upserts)} trained', flush=True)
        # ─────────────────────────────────────────────────────────────────────

        job['status'] = 'done'
        job['models_trained'] = len(upserts)
        job['results'] = results
        print(f'[lr_train] done: {len(upserts)} models trained', flush=True)

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        print(f'[lr_train] ERROR: {e}', flush=True)


@app.route('/api/train_lr_intraday', methods=['POST', 'OPTIONS'])
def train_lr_intraday():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job_id = str(uuid.uuid4())[:12]
    lr_training_jobs[job_id] = {
        'status': 'starting',
        'models_done': 0,
        'models_total': 0,
        'models_trained': 0,
        'total_samples': 0,
        'results': {},
        'start_time': time.time(),
        'error': None,
    }
    threading.Thread(target=_run_lr_training, args=(job_id,), daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/api/sync-earnings', methods=['POST', 'OPTIONS'])
def sync_earnings():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    try:
        n = _sync_earnings_calendar()
        return jsonify({'ok': True, 'upserted': n})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/lr_train_status/<job_id>', methods=['GET', 'OPTIONS'])
def lr_train_status(job_id):
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job = lr_training_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'Job not found'}), 404

    return jsonify({
        'ok': True,
        'status': job['status'],
        'models_done': job.get('models_done', 0),
        'models_total': job.get('models_total', 0),
        'models_trained': job.get('models_trained', 0),
        'total_samples': job.get('total_samples', 0),
        'elapsed': int(time.time() - job['start_time']),
        'results': job.get('results', {}),
        'error': job.get('error'),
    })


# ── Historical training samples from price_history ────────────────────────────

def _build_historical_samples(sb) -> list:
    """
    Fetch price_history for all tickers, compute technical indicators from OHLCV,
    and generate (features, target) training samples for all 6 daily horizons.

    XGB scores and earnings_days are set to 0 (neutral) since we have no historical data.
    SPY and ^VIX are fetched from price_history to compute market/macro features.
    Returns list of dicts compatible with get_daily_training_data() format.
    """
    import pandas as pd
    import numpy as np
    from datetime import date as dt_date, timedelta

    HORIZONS = [1, 7, 14, 30, 60, 90]
    MIN_LOOKBACK = 210  # need 200 for SMA200 + buffer

    print('[hist] Fetching assets ticker map...', flush=True)
    asset_map: dict = {}  # asset_id -> ticker
    a_resp = sb.from_('assets').select('id, ticker').execute()
    for a in (a_resp.data or []):
        asset_map[a['id']] = a['ticker']
    print(f'[hist] {len(asset_map)} assets in map', flush=True)

    print('[hist] Fetching price_history (paginated)...', flush=True)
    # Paginate — PostgREST caps at 1000 rows/request
    rows = []
    PAGE = 1000
    offset = 0
    while True:
        resp = sb.from_('price_history').select(
            'asset_id, trade_date, open, high, low, close, volume'
        ).order('trade_date').range(offset, offset + PAGE - 1).execute()
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < PAGE:
            break
        offset += PAGE
    print(f'[hist] fetched {len(rows)} rows from price_history', flush=True)
    if not rows:
        print('[hist] No data in price_history', flush=True)
        return []

    # Group by ticker using asset_map
    by_ticker: dict = {}
    for r in rows:
        ticker = asset_map.get(r.get('asset_id'))
        if not ticker:
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append(r)

    print(f'[hist] {len(by_ticker)} tickers loaded', flush=True)

    # Build SPY series for correlation + sp500 features
    def _make_df(ticker_rows):
        df = pd.DataFrame([{
            'date': r['trade_date'],
            'open': float(r['open'] or 0),
            'high': float(r['high'] or 0),
            'low':  float(r['low']  or 0),
            'close': float(r['close'] or 0),
            'volume': float(r['volume'] or 0),
        } for r in ticker_rows])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

    spy_df = _make_df(by_ticker.get('SPY', []))
    vix_df = _make_df(by_ticker.get('^VIX', []))

    # Date → SPY close lookup
    spy_close_map: dict = {}
    spy_ret_map: dict   = {}  # log returns for future spy
    if not spy_df.empty:
        spy_close_map = dict(zip(spy_df['date'].dt.date, spy_df['close']))
        spy_ret = spy_df['close'].pct_change() * 100
        spy_ret_map = dict(zip(spy_df['date'].dt.date, spy_ret))

    vix_close_map: dict = {}
    if not vix_df.empty:
        vix_close_map = dict(zip(vix_df['date'].dt.date, vix_df['close']))

    def _safe(s: pd.Series, i: int, default=0.0) -> float:
        v = s.iloc[i] if 0 <= i < len(s) else None
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute all needed indicators for a ticker. Returns df with indicator columns."""
        c = df['close']
        h = df['high']
        l = df['low']
        v = df['volume']

        # ── SMAs ──
        df['sma20']  = c.rolling(20).mean()
        df['sma50']  = c.rolling(50).mean()
        df['sma200'] = c.rolling(200).mean()
        df['price_vs_sma20']  = (c - df['sma20'])  / df['sma20'].replace(0, np.nan)  * 100
        df['price_vs_sma50']  = (c - df['sma50'])  / df['sma50'].replace(0, np.nan)  * 100
        df['price_vs_sma200'] = (c - df['sma200']) / df['sma200'].replace(0, np.nan) * 100

        # ── RSI 14 ──
        delta = c.diff()
        gain  = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rs    = gain / loss.replace(0, 1e-10)
        df['rsi_14'] = 100 - 100 / (1 + rs)

        # ── MACD ──
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        df['macd_histogram'] = macd - macd.ewm(span=9, adjust=False).mean()

        # ── Stochastic RSI ──
        rsi_min = df['rsi_14'].rolling(14).min()
        rsi_max = df['rsi_14'].rolling(14).max()
        df['stoch_rsi'] = (df['rsi_14'] - rsi_min) / (rsi_max - rsi_min + 1e-10) * 100

        # ── MFI 14 ──
        tp = (h + l + c) / 3
        mf = tp * v
        pos_mf = mf.where(tp > tp.shift(1), 0.0)
        neg_mf = mf.where(tp < tp.shift(1), 0.0)
        pmf14  = pos_mf.rolling(14).sum()
        nmf14  = neg_mf.rolling(14).sum()
        df['mfi_14'] = 100 - 100 / (1 + pmf14 / nmf14.replace(0, 1e-10))

        # ── CCI 20 ──
        hl_avg = (h + l + c) / 3
        df['cci_20'] = (hl_avg - hl_avg.rolling(20).mean()) / (0.015 * hl_avg.rolling(20).std().replace(0, 1e-10))

        # ── Williams %R 14 ──
        hi14 = h.rolling(14).max()
        lo14 = l.rolling(14).min()
        df['williams_r_14'] = (hi14 - c) / (hi14 - lo14 + 1e-10) * -100

        # ── Bollinger Bands ──
        bb_mid = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df['bb_pct_b']  = (c - bb_lower) / (bb_upper - bb_lower + 1e-10)
        bb_width = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)
        df['bb_squeeze'] = (bb_width < bb_width.rolling(60).min() * 1.1).astype(float)

        # ── ATR % ──
        prev_c = c.shift(1)
        tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        df['atr_pct'] = atr14 / c.replace(0, np.nan) * 100

        # ── Historical Volatility 20 / 60 ──
        log_ret = np.log(c / c.shift(1))
        df['hist_vol_20'] = log_ret.rolling(20).std() * np.sqrt(252) * 100
        df['hist_vol_60'] = log_ret.rolling(60).std() * np.sqrt(252) * 100

        # ── ADX 14 ──
        ph = h - h.shift(1)
        pl = l.shift(1) - l
        pdm = ph.where((ph > 0) & (ph > pl), 0.0).fillna(0)
        ndm = pl.where((pl > 0) & (pl > ph), 0.0).fillna(0)
        atr_s = tr.ewm(span=14, adjust=False).mean()
        pdi = 100 * pdm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, 1e-10)
        ndi = 100 * ndm.ewm(span=14, adjust=False).mean() / atr_s.replace(0, 1e-10)
        dx  = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
        df['adx_14'] = dx.ewm(span=14, adjust=False).mean()

        # ── ROC ──
        df['roc_5']  = c.pct_change(5)  * 100
        df['roc_10'] = c.pct_change(10) * 100
        df['roc_20'] = c.pct_change(20) * 100

        # ── Support / Resistance (52-week rolling) ──
        df['support_52w']    = l.rolling(252, min_periods=20).min()
        df['resistance_52w'] = h.rolling(252, min_periods=20).max()
        df['dist_to_support_pct']    = (c - df['support_52w'])    / c.replace(0, np.nan) * 100
        df['dist_to_resistance_pct'] = (df['resistance_52w'] - c) / c.replace(0, np.nan) * 100

        # ── Volume ratio (vs 20d avg) ──
        avg_vol = v.rolling(20).mean()
        df['volume_ratio'] = v / avg_vol.replace(0, 1e-10)

        # ── CMF 20 ──
        mfm = ((c - l) - (h - c)) / (h - l + 1e-10)
        df['cmf_20'] = (mfm * v).rolling(20).sum() / v.rolling(20).sum().replace(0, 1e-10)

        # ── Stochastic K ──
        lo14k = l.rolling(14).min()
        hi14k = h.rolling(14).max()
        df['stoch_k'] = (c - lo14k) / (hi14k - lo14k + 1e-10) * 100

        # ── Candle signal (simple: body vs range) ──
        body = (c - df['open']).abs()
        total_range = (h - l).replace(0, 1e-10)
        body_ratio = body / total_range
        df['candle_signal'] = np.where(
            (c > df['open']) & (body_ratio > 0.6), 'bullish',
            np.where((c < df['open']) & (body_ratio > 0.6), 'bearish', 'neutral')
        )

        # ── OBV trend ──
        obv = (v * np.sign(c.diff().fillna(0))).cumsum()
        obv_slope = obv.rolling(5).apply(lambda x: (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0, raw=True)
        df['obv_trend'] = np.where(obv_slope > 0, 'rising', np.where(obv_slope < 0, 'falling', 'flat'))

        return df

    def _compute_spy_features(spy_df: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """For each date in df, compute market_corr_60d and sp500_rsi vs SPY."""
        # Merge df dates with SPY dates to get aligned series
        merged = df[['date', 'close']].rename(columns={'close': 'ticker_c'}).merge(
            spy_df[['date', 'close', 'rsi_14']].rename(columns={'close': 'spy_c', 'rsi_14': 'spy_rsi'}),
            on='date', how='left'
        )
        ticker_ret = merged['ticker_c'].pct_change()
        spy_ret    = merged['spy_c'].pct_change()
        corr60 = ticker_ret.rolling(60).corr(spy_ret)
        merged['market_corr_60d'] = corr60
        merged['sp500_rsi'] = merged['spy_rsi']

        # SP500 trend: SMA50 vs SMA200
        spy_sma50  = merged['spy_c'].rolling(50).mean()
        spy_sma200 = merged['spy_c'].rolling(200).mean()
        merged['sp500_trend'] = np.where(
            spy_sma50 > spy_sma200, 'bullish',
            np.where(spy_sma50 < spy_sma200, 'bearish', 'neutral')
        )
        return merged[['date', 'market_corr_60d', 'sp500_rsi', 'sp500_trend']]

    # Pre-compute SPY indicators
    if not spy_df.empty:
        spy_df = _compute_indicators(spy_df)

    all_samples = []
    skip_tickers = {'SPY', '^VIX', '^GSPC', 'QQQ'}  # market proxies, skip as targets

    for ticker, ticker_rows in by_ticker.items():
        if ticker in skip_tickers:
            continue
        if len(ticker_rows) < MIN_LOOKBACK:
            continue

        try:
            df = _make_df(ticker_rows)
            df = _compute_indicators(df)

            # Merge SPY features
            if not spy_df.empty:
                spy_feats = _compute_spy_features(spy_df, df)
                df = df.merge(spy_feats, on='date', how='left')
            else:
                df['market_corr_60d'] = 0.0
                df['sp500_rsi'] = 50.0
                df['sp500_trend'] = 'neutral'

            # Date index for fast future lookup
            date_arr = df['date'].dt.date.values
            close_arr = df['close'].values
            n = len(df)

            for i in range(MIN_LOOKBACK, n):
                if pd.isna(df['sma200'].iloc[i]):
                    continue

                cur_date   = date_arr[i]
                cur_close  = close_arr[i]
                if cur_close <= 0:
                    continue

                # VIX on this date
                vix_val = vix_close_map.get(cur_date, 20.0) or 20.0

                # Build feature row dict matching _extract_daily_features format
                row = {
                    'price_vs_sma20':  _safe(df['price_vs_sma20'], i),
                    'price_vs_sma50':  _safe(df['price_vs_sma50'], i),
                    'price_vs_sma200': _safe(df['price_vs_sma200'], i),
                    'rsi_14':          _safe(df['rsi_14'], i, 50),
                    'macd_histogram':  _safe(df['macd_histogram'], i),
                    'stoch_rsi':       _safe(df['stoch_rsi'], i, 50),
                    'mfi_14':          _safe(df['mfi_14'], i, 50),
                    'cci_20':          _safe(df['cci_20'], i),
                    'williams_r_14':   _safe(df['williams_r_14'], i, -50),
                    'bb_pct_b':        _safe(df['bb_pct_b'], i, 0.5),
                    'bb_squeeze':      bool(df['bb_squeeze'].iloc[i]),
                    'atr_pct':         _safe(df['atr_pct'], i, 1),
                    'hist_vol_20':     _safe(df['hist_vol_20'], i, 20),
                    'hist_vol_60':     _safe(df['hist_vol_60'], i, 20),
                    'adx_14':          _safe(df['adx_14'], i, 20),
                    'roc_5':           _safe(df['roc_5'], i),
                    'roc_10':          _safe(df['roc_10'], i),
                    'roc_20':          _safe(df['roc_20'], i),
                    'dist_to_support_pct':    _safe(df['dist_to_support_pct'], i, 5),
                    'dist_to_resistance_pct': _safe(df['dist_to_resistance_pct'], i, 5),
                    'volume_ratio':    _safe(df['volume_ratio'], i, 1),
                    'cmf_20':          _safe(df['cmf_20'], i),
                    'stoch_k':         _safe(df['stoch_k'], i, 50),
                    'candle_signal':   df['candle_signal'].iloc[i],
                    'obv_trend':       df['obv_trend'].iloc[i],
                    'market_corr_60d': _safe(df['market_corr_60d'], i) if 'market_corr_60d' in df.columns else 0.0,
                    'vix_level':       vix_val,
                    'sp500_trend':     df['sp500_trend'].iloc[i] if 'sp500_trend' in df.columns else 'neutral',
                    'sp500_rsi':       _safe(df['sp500_rsi'], i, 50) if 'sp500_rsi' in df.columns else 50.0,
                    # XGB scores → 0 (no historical data)
                    'score_macro': 0.0, 'score_fundamental': 0.0,
                    'score_sentimiento': 0.0, 'score_tendencia': 0.0, 'score_momentum': 0.0,
                    'created_month': cur_date.month,
                    'next_earnings_days': None,  # → earn_norm = 0
                }

                feats = _extract_daily_features(row)
                if len(feats) != len(DAILY_FEATURE_NAMES):
                    continue

                # Future returns for each horizon
                for h in HORIZONS:
                    target_cal_date = cur_date + timedelta(days=h)
                    # Find first trading date >= target_cal_date in this ticker
                    future_idx = None
                    for j in range(i + 1, min(i + h + 10, n)):
                        if date_arr[j] >= target_cal_date:
                            future_idx = j
                            break
                    if future_idx is None:
                        continue
                    future_close = close_arr[future_idx]
                    if future_close <= 0:
                        continue
                    actual_ret = (future_close - cur_close) / cur_close * 100

                    # SPY return over same period (for beta adjustment)
                    future_spy = spy_close_map.get(date_arr[future_idx])
                    cur_spy    = spy_close_map.get(cur_date)
                    spy_ret_val = ((future_spy - cur_spy) / cur_spy * 100
                                   if future_spy and cur_spy and cur_spy > 0 else None)

                    all_samples.append({
                        'ticker':           ticker,
                        'horizon_bucket':   h,
                        'actual_signed_pct': actual_ret,
                        'spy_actual_pct':    spy_ret_val,
                        'created_at':       cur_date.isoformat(),
                        '_features':        feats,  # pre-computed, skip _extract_daily_features
                        '_is_historical':   True,
                    })
        except Exception as e:
            print(f'[hist] {ticker} error: {e}', flush=True)
            continue

    print(f'[hist] generated {len(all_samples)} historical samples', flush=True)
    return all_samples


# ── Daily signed Ridge training ───────────────────────────────────────────────

DAILY_FEATURE_NAMES = [
    # Price vs moving averages (3)
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
    # Oscillators (6)
    'rsi_norm', 'macd_norm', 'stoch_rsi_norm',
    'mfi_norm', 'cci_norm', 'williams_norm',
    # Volatility / bands (4)
    'bb_pct_b_norm', 'bb_squeeze', 'atr_pct_norm', 'hist_vol_norm',
    # Long-window volatility (1)
    'hist_vol_60_norm',
    # Trend strength (4)
    'adx_norm', 'roc_5_norm', 'roc_10_norm', 'roc_20_norm',
    # Structure levels (2)
    'dist_to_support_norm', 'dist_to_resistance_norm',
    # Volume / candle (5)
    'volume_ratio_norm', 'cmf_norm', 'stoch_k_norm',
    'candle_signal', 'obv_trend',
    # Market correlation (1)
    'market_corr_norm',
    # Macro context (3)
    'vix_norm', 'sp500_trend_enc', 'sp500_rsi_norm',
    # Pre-computed XGB scores (5)
    'score_macro', 'score_fundamental', 'score_sentimiento',
    'score_tendencia', 'score_momentum',
    # Seasonality (2)
    'month_sin', 'month_cos',
    # Earnings proximity (1)
    'earnings_days_norm',
    # VIX regime interactions (3) — derived, no extra SQL needed
    'vix_x_rsi', 'vix_x_momentum', 'vix_x_score_macro',
]


def _extract_daily_features(row: dict) -> list:
    def cl3(v, lo=-3.0, hi=3.0): return max(lo, min(hi, float(v)))
    # Price vs MA
    vs20  = float(row.get('price_vs_sma20',  0) or 0)
    vs50  = float(row.get('price_vs_sma50',  0) or 0)
    vs200 = float(row.get('price_vs_sma200', 0) or 0)
    # Oscillators
    rsi      = float(row.get('rsi_14', 50) or 50)
    macdH    = float(row.get('macd_histogram', 0) or 0)
    stochrsi = float(row.get('stoch_rsi', 50) or 50)
    mfi      = float(row.get('mfi_14', 50) or 50)
    cci      = float(row.get('cci_20', 0) or 0)
    willr    = float(row.get('williams_r_14', -50) or -50)
    # Volatility
    bbB   = float(row.get('bb_pct_b', 0.5) or 0.5)
    bbs   = 1.0 if row.get('bb_squeeze') else 0.0
    atrP  = float(row.get('atr_pct', 1) or 1)
    hv    = float(row.get('hist_vol_20', 20) or 20)
    hv60  = float(row.get('hist_vol_60', 20) or 20)
    # Trend
    adx   = float(row.get('adx_14', 20) or 20)
    roc5  = float(row.get('roc_5',  0) or 0)
    roc10 = float(row.get('roc_10', 0) or 0)
    roc20 = float(row.get('roc_20', 0) or 0)
    # Structure
    dist_sup = float(row.get('dist_to_support_pct',  5) or 5)
    dist_res = float(row.get('dist_to_resistance_pct', 5) or 5)
    # Volume / candle
    vol_r  = float(row.get('volume_ratio', 1) or 1)
    cmf    = float(row.get('cmf_20', 0) or 0)
    stoch_k = float(row.get('stoch_k', 50) or 50)
    cand_s = (row.get('candle_signal') or 'neutral').lower()
    cand  = 1.0 if cand_s == 'bullish' else (-1.0 if cand_s == 'bearish' else 0.0)
    obv_s = (row.get('obv_trend') or 'flat').lower()
    obv   = 1.0 if obv_s == 'rising' else (-1.0 if obv_s == 'falling' else 0.0)
    # Market correlation
    mcorr = float(row.get('market_corr_60d', 0) or 0)
    # Macro
    vix      = float(row.get('vix_level', 20) or 20)
    sp500_t  = (row.get('sp500_trend') or 'neutral').lower()
    sp500_enc = 1.0 if sp500_t == 'bullish' else (-1.0 if sp500_t == 'bearish' else 0.0)
    sp500_rsi = float(row.get('sp500_rsi', 50) or 50)
    # XGB scores
    sc_macro = float(row.get('score_macro', 0) or 0)
    sc_fund  = float(row.get('score_fundamental', 0) or 0)
    sc_sent  = float(row.get('score_sentimiento', 0) or 0)
    sc_tend  = float(row.get('score_tendencia', 0) or 0)
    sc_mom   = float(row.get('score_momentum', 0) or 0)
    month    = int(row.get('created_month') or 1)
    earn_raw = row.get('next_earnings_days')
    earn_norm = cl3(float(earn_raw) / 30) if earn_raw is not None else 0.0
    # Derived normalized scalars for interactions
    vix_n    = cl3((vix - 20) / 10)
    rsi_n    = (rsi - 50) / 25
    roc20_n  = cl3(roc20 / 20)
    sc_macro_n = cl3(sc_macro)
    return [
        # Price vs MA (3)
        cl3(vs20 / 5), cl3(vs50 / 10), cl3(vs200 / 20),
        # Oscillators (6)
        rsi_n,
        cl3(macdH / (abs(macdH) + 0.01)),
        (stochrsi - 50) / 50,
        (mfi - 50) / 25,
        cl3(cci / 100),
        (willr + 50) / 25,
        # Volatility (4)
        bbB * 2 - 1, bbs,
        cl3(atrP / 3, 0.0, 3.0),
        cl3(hv / 50, 0.0, 3.0),
        # Long vol (1)
        cl3(hv60 / 60, 0.0, 3.0),
        # Trend (4)
        cl3(adx / 50 - 0.4),
        cl3(roc5 / 5), cl3(roc10 / 10), roc20_n,
        # Structure (2)
        cl3(dist_sup / 5, 0.0, 3.0),
        cl3(dist_res / 5, 0.0, 3.0),
        # Volume / candle (5)
        cl3(vol_r / 2.0, 0.0, 3.0),
        cl3(cmf, -1.0, 1.0),
        (stoch_k - 50) / 25,
        cand, obv,
        # Market correlation (1)
        cl3(mcorr, -1.0, 1.0),
        # Macro (3)
        vix_n, sp500_enc,
        (sp500_rsi - 50) / 25,
        # XGB scores (5)
        cl3(sc_macro_n), cl3(sc_fund), cl3(sc_sent),
        cl3(sc_tend), cl3(sc_mom),
        # Seasonality (2)
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        # Earnings (1)
        earn_norm,
        # VIX interactions (3)
        cl3(vix_n * rsi_n),
        cl3(vix_n * roc20_n),
        cl3(vix_n * sc_macro_n),
    ]


daily_training_jobs: dict = {}


def _run_lr_training_daily(job_id: str):
    from supabase import create_client
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    import lightgbm as lgb
    from datetime import timezone

    job = daily_training_jobs[job_id]
    HALF_LIFE_DAYS = 180
    lam = math.log(2) / HALF_LIFE_DAYS

    def _parse_ts(ts):
        if not ts:
            return datetime.min.replace(tzinfo=timezone.utc)
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        return ts if getattr(ts, 'tzinfo', None) else ts.replace(tzinfo=timezone.utc)

    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        job['status'] = 'fetching'

        resp = sb.rpc('get_daily_training_data', {'p_limit': 100000}).execute()
        real_rows = resp.data or []
        print(f'[lr_train_daily] fetched {len(real_rows)} real rows', flush=True)

        # Build historical samples from price_history (walk-forward, no lookahead)
        hist_samples = _build_historical_samples(sb)

        # Merge: real rows (weight 3x) + historical rows (weight 1x)
        # Historical rows carry pre-computed features in '_features' key
        all_rows = real_rows + hist_samples
        job['total_samples'] = len(all_rows)
        print(f'[lr_train_daily] total rows: {len(all_rows)} ({len(real_rows)} real + {len(hist_samples)} historical)', flush=True)

        if not all_rows:
            job['status'] = 'error'
            job['error'] = 'No training data found'
            return

        now_utc = datetime.now(timezone.utc)
        all_rows.sort(key=lambda r: _parse_ts(r.get('created_at')))
        holdout_cutoff = now_utc - timedelta(days=30)

        def decay_w(ts_str):
            age = (now_utc - _parse_ts(ts_str)).total_seconds() / 86400
            return math.exp(-lam * max(0.0, age))

        BUCKETS = [1, 7, 14, 30, 60, 90]
        groups: dict = {b: {'X': [], 'y': [], 'y_spy': [], 'w': [], 'ts': [], 'ticker': [], 'atr': []} for b in BUCKETS}
        earnings_filtered = 0
        for row in all_rows:
            h = int(row.get('horizon_bucket') or 0)
            if h not in groups:
                continue
            signed_pct = row.get('actual_signed_pct')
            if signed_pct is None:
                continue
            is_hist = row.get('_is_historical', False)
            if is_hist:
                # Historical sample: features already computed, no earnings filter
                feats = row['_features']
                base_weight = 1.0  # historical weight 1x
            else:
                # Real prediction: earnings filter + compute features
                earn_days = row.get('next_earnings_days')
                if earn_days is not None and abs(int(earn_days)) <= 3:
                    earnings_filtered += 1
                    continue
                feats = _extract_daily_features(row)
                base_weight = 3.0  # real predictions weight 3x (more reliable)
            if len(feats) != len(DAILY_FEATURE_NAMES):
                continue
            ts_str = row.get('created_at', '')
            groups[h]['X'].append(feats)
            groups[h]['y'].append(float(signed_pct))
            groups[h]['y_spy'].append(row.get('spy_actual_pct'))
            groups[h]['w'].append(decay_w(ts_str) * base_weight)
            groups[h]['ts'].append(ts_str)
            groups[h]['ticker'].append(row.get('ticker', ''))
            groups[h]['atr'].append(float(row.get('atr_pct', 1) or 1))
        print(f'[lr_train_daily] earnings filter removed {earnings_filtered} rows', flush=True)

        job['status'] = 'training'
        job['models_total'] = len(BUCKETS)
        job['models_done'] = 0

        upserts = []
        for bucket in BUCKETS:
            X_raw = groups[bucket]['X']
            if len(X_raw) < 20:
                print(f'[lr_train_daily] H={bucket}: {len(X_raw)} samples — skip', flush=True)
                job['models_done'] += 1
                continue

            X_np = np.array(X_raw, dtype=float)
            y_np = np.array(groups[bucket]['y'], dtype=float)
            spy_np = np.array([float(v) if v is not None else float('nan') for v in groups[bucket]['y_spy']])
            w_np = np.array(groups[bucket]['w'], dtype=float)
            ticker_arr = np.array(groups[bucket]['ticker'])
            atr_arr = np.array(groups[bucket]['atr'], dtype=float)
            ts_arr = np.array(groups[bucket]['ts'])

            # Paso A: winsorize outliers at p99 (removes corrupt extremes like MRVL 72%)
            p99 = float(np.percentile(np.abs(y_np), 99))
            keep = np.abs(y_np) <= p99
            n_winsor = int((~keep).sum())
            if n_winsor > 0:
                X_np, y_np, spy_np, w_np = X_np[keep], y_np[keep], spy_np[keep], w_np[keep]
                ticker_arr, atr_arr, ts_arr = ticker_arr[keep], atr_arr[keep], ts_arr[keep]
                print(f'[daily] H={bucket}: winsorized {n_winsor} outliers (p99={p99:.2f}%)', flush=True)

            # Walk-forward: holdout = last 30 days (uses ts_arr post-winsorization)
            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in ts_arr])
            tv_mask = ~holdout_mask
            if tv_mask.sum() >= 20:
                X_tv, y_tv, spy_tv, w_tv = X_np[tv_mask], y_np[tv_mask], spy_np[tv_mask], w_np[tv_mask]
                atr_tv = np.clip(atr_arr[tv_mask], 0.1, 20.0)
            else:
                X_tv, y_tv, spy_tv, w_tv = X_np, y_np, spy_np, w_np
                atr_tv = np.clip(atr_arr, 0.1, 20.0)

            # OLS beta_spy per bucket: y_total = beta * spy + idio
            # Paso E: allow negative beta (defensive stocks have beta < 0 legitimately)
            beta_spy = 0.0
            vb = ~np.isnan(y_tv) & ~np.isnan(spy_tv)
            if vb.sum() >= 20:
                yb = y_tv[vb]; spy_b = spy_tv[vb]
                sc = spy_b - spy_b.mean(); d = float(np.dot(sc, sc))
                if d > 1e-10:
                    beta_spy = float(np.clip(np.dot(yb - yb.mean(), sc) / d, -0.5, 3.0))

            split = max(10, int(len(X_tv) * 0.8))
            X_train, X_val = X_tv[:split], X_tv[split:]
            # Apply beta adjustment to target (train on idiosyncratic return)
            y_train = apply_beta_adj(y_tv[:split], spy_tv[:split], beta_spy)
            y_val   = apply_beta_adj(y_tv[split:], spy_tv[split:], beta_spy)
            w_train = w_tv[:split]
            atr_train = atr_tv[:split]
            atr_val   = atr_tv[split:] if split < len(atr_tv) else np.ones(0)

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, X_train.shape[1]))

            tm = ~np.isnan(y_train)
            if tm.sum() < 20:
                job['models_done'] += 1
                continue

            # Ridge signed — kept for backward compat
            reg = Ridge(alpha=1.0)
            reg.fit(X_train_s[tm], y_train[tm], sample_weight=w_train[tm])
            r2_val = float(r2_score(y_train[tm], reg.predict(X_train_s[tm])))
            avg_mag = float(np.mean(np.abs(y_tv)))
            median_mag = float(np.median(np.abs(y_tv)))
            vm = ~np.isnan(y_val) if len(X_val_s) > 0 else np.zeros(0, dtype=bool)
            val_mae_ridge = None
            if vm.sum() > 0:
                val_mae_ridge = float(np.mean(np.abs(y_val[vm] - reg.predict(X_val_s[vm]))))

            # ATR normalization: train LightGBM on (return / ATR) so scale is uniform across tickers
            atr_tr_valid = np.clip(atr_train[tm], 0.1, 20.0)
            atr_train_mean = float(np.mean(atr_tr_valid))
            y_train_norm = y_train / atr_train  # shape matches y_train; NaN rows excluded by tm mask

            # Paso C: LightGBM daily — Optuna-tuned hyperparams + early stopping
            lgbm_model_b64 = lgbm_val_mae = None
            if tm.sum() >= 30:
                import optuna as _optuna
                _optuna.logging.set_verbosity(_optuna.logging.WARNING)
                has_val_d = vm.sum() >= 5
                atr_val_valid = np.clip(atr_val[vm], 0.1, 20.0) if vm.sum() > 0 else np.ones(0)

                def _lgbm_daily_obj(trial):
                    p = dict(
                        num_leaves=trial.suggest_int('num_leaves', 15, 127),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        min_child_samples=trial.suggest_int('min_child_samples', 5, 50),
                        max_depth=trial.suggest_int('max_depth', 3, 8),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
                        n_estimators=300, random_state=42, verbose=-1,
                        objective='regression_l1',
                    )
                    m = lgb.LGBMRegressor(**p)
                    m.fit(X_train_s[tm], y_train_norm[tm], sample_weight=w_train[tm])
                    if has_val_d:
                        # evaluate in real % (denormalize) so MAE is comparable
                        pred_real = m.predict(X_val_s[vm]) * atr_val_valid
                        return float(np.mean(np.abs(y_val[vm] - pred_real)))
                    pred_real_tr = m.predict(X_train_s[tm]) * atr_tr_valid
                    return float(np.mean(np.abs(y_train[tm] - pred_real_tr)))

                _study_d = _optuna.create_study(
                    direction='minimize', sampler=_optuna.samplers.TPESampler(seed=42)
                )
                _n_trials_d = 50 if int(tm.sum()) >= 30 else 25
                _study_d.optimize(_lgbm_daily_obj, n_trials=_n_trials_d, show_progress_bar=False)
                best_p_d = _study_d.best_params
                print(f'[optuna_daily] H={bucket} best={best_p_d} val_mae={_study_d.best_value:.4f}', flush=True)

                # Eval set uses ATR-normalized targets so early stopping monitors normalized loss
                eval_set_d = [(X_val_s[vm], y_val[vm] / atr_val_valid)] if has_val_d else None
                callbacks_d = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_d else None
                lgb_reg = lgb.LGBMRegressor(
                    **best_p_d, n_estimators=600, random_state=42, verbose=-1,
                    objective='regression_l1',
                )
                lgb_reg.fit(X_train_s[tm], y_train_norm[tm], sample_weight=w_train[tm],
                            eval_set=eval_set_d, callbacks=callbacks_d)
                if vm.sum() > 0:
                    # Denormalize predictions to get real % MAE
                    val_preds_real = lgb_reg.predict(X_val_s[vm]) * atr_val_valid
                    val_residuals = np.abs(y_val[vm] - val_preds_real)
                    lgbm_val_mae = float(np.mean(val_residuals))
                    lgbm_error_p25 = float(np.percentile(val_residuals, 25))
                    lgbm_error_p50 = float(np.percentile(val_residuals, 50))
                    lgbm_error_p75 = float(np.percentile(val_residuals, 75))
                    lgbm_error_p90 = float(np.percentile(val_residuals, 90))
                else:
                    lgbm_error_p25 = lgbm_error_p50 = lgbm_error_p75 = lgbm_error_p90 = None
                lgbm_model_b64 = base64.b64encode(pickle.dumps(lgb_reg)).decode('utf-8')

            upserts.append({
                'horizon_bucket': bucket,
                'feature_names': DAILY_FEATURE_NAMES,
                'feature_means': scaler.mean_.tolist(),
                'feature_stds': scaler.scale_.tolist(),
                'signed_coefficients': reg.coef_.tolist(),
                'signed_bias': float(reg.intercept_),
                'signed_r2': round(r2_val, 4),
                'avg_actual_mag': round(avg_mag, 4),
                'median_actual_mag': round(median_mag, 4),
                'train_samples': int(tm.sum()),
                'lgbm_model': lgbm_model_b64,
                'lgbm_val_mae': lgbm_val_mae,
                'val_mae_ridge': val_mae_ridge,
                'beta_spy': beta_spy,
                'error_p25': lgbm_error_p25,
                'error_p50': lgbm_error_p50,
                'error_p75': lgbm_error_p75,
                'error_p90': lgbm_error_p90,
                'atr_normalized': True,
                'atr_train_mean': round(atr_train_mean, 4),
            })
            job['models_done'] += 1
            print(
                f'[lr_train_daily] H={bucket}: n={tm.sum()} beta={beta_spy:.3f} r2={r2_val:.3f} '
                f'val_mae_ridge={val_mae_ridge} lgbm_val_mae={lgbm_val_mae}',
                flush=True,
            )

            # Paso C: train per-cluster LGBM models for this bucket
            import optuna as _optuna_cl
            _optuna_cl.logging.set_verbosity(_optuna_cl.logging.WARNING)
            cluster_upserts = []
            tv_ticker = ticker_arr[tv_mask] if tv_mask.sum() >= 20 else ticker_arr
            tv_X = X_np[tv_mask] if tv_mask.sum() >= 20 else X_np
            tv_y = y_np[tv_mask] if tv_mask.sum() >= 20 else y_np
            tv_spy = spy_np[tv_mask] if tv_mask.sum() >= 20 else spy_np
            tv_w = w_np[tv_mask] if tv_mask.sum() >= 20 else w_np
            for cl_name in set(TICKER_CLUSTERS.values()):
                cl_mask = np.array([TICKER_CLUSTERS.get(t, '') == cl_name for t in tv_ticker])
                if cl_mask.sum() < 50:
                    continue
                cl_X = tv_X[cl_mask]; cl_y = tv_y[cl_mask]
                cl_spy = tv_spy[cl_mask]; cl_w = tv_w[cl_mask]
                vb_cl = ~np.isnan(cl_y) & ~np.isnan(cl_spy)
                cl_beta = 0.0
                if vb_cl.sum() >= 20:
                    yb = cl_y[vb_cl]; spb = cl_spy[vb_cl]
                    sc_cl = spb - spb.mean(); d_cl = float(np.dot(sc_cl, sc_cl))
                    if d_cl > 1e-10:
                        cl_beta = float(np.clip(np.dot(yb - yb.mean(), sc_cl) / d_cl, -0.5, 3.0))
                split_cl = max(10, int(len(cl_X) * 0.8))
                cl_Xtr, cl_Xvl = cl_X[:split_cl], cl_X[split_cl:]
                cl_ytr = apply_beta_adj(cl_y[:split_cl], cl_spy[:split_cl], cl_beta)
                cl_yvl = apply_beta_adj(cl_y[split_cl:], cl_spy[split_cl:], cl_beta)
                cl_wtr = cl_w[:split_cl]
                scaler_cl = StandardScaler()
                cl_Xtr_s = scaler_cl.fit_transform(cl_Xtr)
                cl_Xvl_s = scaler_cl.transform(cl_Xvl) if len(cl_Xvl) > 0 else np.empty((0, cl_Xtr.shape[1]))
                tm_cl = ~np.isnan(cl_ytr)
                vm_cl = ~np.isnan(cl_yvl) if len(cl_Xvl_s) > 0 else np.zeros(0, dtype=bool)
                if tm_cl.sum() < 30:
                    continue

                def _cl_obj(trial, Xtr=cl_Xtr_s, ytr=cl_ytr, wtr=cl_wtr, Xvl=cl_Xvl_s, yvl=cl_yvl, tmk=tm_cl, vmk=vm_cl):
                    p = dict(
                        num_leaves=trial.suggest_int('num_leaves', 15, 63),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        min_child_samples=trial.suggest_int('min_child_samples', 5, 30),
                        max_depth=trial.suggest_int('max_depth', 3, 7),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
                        n_estimators=300, random_state=42, verbose=-1, objective='regression_l1',
                    )
                    m = lgb.LGBMRegressor(**p)
                    m.fit(Xtr[tmk], ytr[tmk], sample_weight=wtr[tmk])
                    if vmk.sum() >= 5:
                        return float(np.mean(np.abs(yvl[vmk] - m.predict(Xvl[vmk]))))
                    return float(np.mean(np.abs(ytr[tmk] - m.predict(Xtr[tmk]))))

                _study_cl = _optuna_cl.create_study(direction='minimize', sampler=_optuna_cl.samplers.TPESampler(seed=42))
                _study_cl.optimize(_cl_obj, n_trials=25, show_progress_bar=False)
                best_cl = _study_cl.best_params
                lgb_cl = lgb.LGBMRegressor(**best_cl, n_estimators=600, random_state=42, verbose=-1, objective='regression_l1')
                eval_cl = [(cl_Xvl_s[vm_cl], cl_yvl[vm_cl])] if vm_cl.sum() >= 5 else None
                cbs_cl = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_cl else None
                lgb_cl.fit(cl_Xtr_s[tm_cl], cl_ytr[tm_cl], sample_weight=cl_wtr[tm_cl], eval_set=eval_cl, callbacks=cbs_cl)
                cl_val_mae = float(np.mean(np.abs(cl_yvl[vm_cl] - lgb_cl.predict(cl_Xvl_s[vm_cl])))) if vm_cl.sum() > 0 else None
                cl_b64 = base64.b64encode(pickle.dumps(lgb_cl)).decode('utf-8')
                cluster_upserts.append({
                    'cluster_name': cl_name, 'horizon_bucket': bucket,
                    'lgbm_model': cl_b64, 'lgbm_val_mae': cl_val_mae,
                    'beta_spy': cl_beta, 'train_samples': int(tm_cl.sum()),
                    'feature_names': DAILY_FEATURE_NAMES,
                    'feature_means': scaler_cl.mean_.tolist(),
                    'feature_stds': scaler_cl.scale_.tolist(),
                })
                print(f'[daily_cluster] H={bucket} {cl_name}: n={tm_cl.sum()} beta={cl_beta:.3f} val_mae={cl_val_mae}', flush=True)
            for cu in cluster_upserts:
                sb.rpc('upsert_daily_cluster_model', {'p_params': [cu]}).execute()

        for u in upserts:
            sb.rpc('upsert_daily_signed_params', {'p_params': [u]}).execute()

        job['status'] = 'done'
        job['models_trained'] = len(upserts)
        print(f'[lr_train_daily] done: {len(upserts)} buckets trained', flush=True)

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        print(f'[lr_train_daily] ERROR: {e}', flush=True)


@app.route('/api/train_lr_daily', methods=['POST', 'OPTIONS'])
def train_lr_daily():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job_id = str(uuid.uuid4())[:12]
    daily_training_jobs[job_id] = {
        'status': 'starting',
        'models_done': 0,
        'models_total': 5,
        'models_trained': 0,
        'total_samples': 0,
        'start_time': time.time(),
        'error': None,
    }
    threading.Thread(target=_run_lr_training_daily, args=(job_id,), daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/api/lr_train_daily_status/<job_id>', methods=['GET', 'OPTIONS'])
def lr_train_daily_status(job_id):
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    job = daily_training_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'Job not found'}), 404

    return jsonify({
        'ok': True,
        'status': job['status'],
        'models_done': job.get('models_done', 0),
        'models_total': job.get('models_total', 5),
        'models_trained': job.get('models_trained', 0),
        'total_samples': job.get('total_samples', 0),
        'elapsed': int(time.time() - job['start_time']),
        'error': job.get('error'),
    })


# ── LightGBM model cache (refreshed every 10 min to avoid per-request DB hits) ──

_lgbm_cache: dict = {}
_lgbm_cache_ts: float = 0.0

_lgbm_session_cache: dict = {}
_lgbm_session_cache_ts: float = 0.0

_lgbm_ticker_cache: dict = {}
_lgbm_ticker_cache_ts: float = 0.0

_lgbm_cluster_cache: dict = {}
_lgbm_cluster_cache_ts: float = 0.0


def _load_lgbm_models_cached():
    """Returns dict[key] = (model, beta_spy). beta_spy=0 for old models without it."""
    global _lgbm_cache, _lgbm_cache_ts
    if time.time() - _lgbm_cache_ts < 600 and _lgbm_cache:
        return _lgbm_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('model_learned_params_intraday').select(
        'model_name,horizon_minutes,lgbm_model,beta_spy'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['model_name']}:{row['horizon_minutes']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                new_cache[key] = (model, beta)
            except Exception:
                pass
    _lgbm_cache = new_cache
    _lgbm_cache_ts = time.time()
    return _lgbm_cache


def _load_lgbm_session_models_cached():
    """Load per-session LGBM models. Keys: 'model_name:horizon:session'. Values: (model, beta_spy)."""
    global _lgbm_session_cache, _lgbm_session_cache_ts
    if time.time() - _lgbm_session_cache_ts < 600 and _lgbm_session_cache:
        return _lgbm_session_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('lgbm_session_models_intraday').select(
        'model_name,horizon_minutes,market_session,lgbm_model,beta_spy'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['model_name']}:{row['horizon_minutes']}:{row['market_session']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                new_cache[key] = (model, beta)
            except Exception:
                pass
    _lgbm_session_cache = new_cache
    _lgbm_session_cache_ts = time.time()
    return _lgbm_session_cache


def _load_lgbm_ticker_models_cached():
    """Per-ticker LGBM models. Keys: 'TICKER:horizon'. Values: (model, beta_spy, train_samples)."""
    global _lgbm_ticker_cache, _lgbm_ticker_cache_ts
    if time.time() - _lgbm_ticker_cache_ts < 600 and _lgbm_ticker_cache:
        return _lgbm_ticker_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('lgbm_ticker_models_intraday').select(
        'ticker,horizon_minutes,lgbm_model,beta_spy,train_samples'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['ticker']}:{row['horizon_minutes']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                n_samples = int(row.get('train_samples') or 0)
                new_cache[key] = (model, beta, n_samples)
            except Exception:
                pass
    _lgbm_ticker_cache = new_cache
    _lgbm_ticker_cache_ts = time.time()
    return _lgbm_ticker_cache


def _load_lgbm_cluster_models_cached():
    """Per-cluster LGBM models. Keys: 'cluster_name:horizon'. Values: (model, beta_spy)."""
    global _lgbm_cluster_cache, _lgbm_cluster_cache_ts
    if time.time() - _lgbm_cluster_cache_ts < 600 and _lgbm_cluster_cache:
        return _lgbm_cluster_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('lgbm_cluster_models_intraday').select(
        'cluster_name,horizon_minutes,lgbm_model,beta_spy'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['cluster_name']}:{row['horizon_minutes']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                new_cache[key] = (model, beta)
            except Exception:
                pass
    _lgbm_cluster_cache = new_cache
    _lgbm_cluster_cache_ts = time.time()
    return _lgbm_cluster_cache


_lgbm_daily_cache: dict = {}
_lgbm_daily_cache_ts: float = 0.0
_lgbm_daily_cluster_cache: dict = {}
_lgbm_daily_cluster_cache_ts: float = 0.0


def _load_lgbm_daily_models_cached():
    """Daily LGBM global models. Keys: horizon_bucket str. Values: (model, scaler, beta, avg_mag)."""
    global _lgbm_daily_cache, _lgbm_daily_cache_ts
    if time.time() - _lgbm_daily_cache_ts < 600 and _lgbm_daily_cache:
        return _lgbm_daily_cache
    from supabase import create_client
    from sklearn.preprocessing import StandardScaler as _SS
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('model_signed_params_daily').select(
        'horizon_bucket,lgbm_model,feature_means,feature_stds,beta_spy,avg_actual_mag,train_samples,atr_normalized,atr_train_mean'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if not row.get('lgbm_model'):
            continue
        key = str(row['horizon_bucket'])
        try:
            m = pickle.loads(base64.b64decode(row['lgbm_model']))
            means = row.get('feature_means') or []
            stds  = row.get('feature_stds')  or []
            if len(means) != len(DAILY_FEATURE_NAMES):
                print(f'[lgbm_daily] H={key}: feature dim {len(means)} != {len(DAILY_FEATURE_NAMES)} — skip', flush=True)
                continue
            sc = _SS()
            sc.mean_ = np.array(means); sc.scale_ = np.array(stds)
            sc.var_ = sc.scale_ ** 2; sc.n_samples_seen_ = int(row.get('train_samples') or 1)
            sc.n_features_in_ = len(means)
            atr_norm = bool(row.get('atr_normalized') or False)
            atr_mean = float(row.get('atr_train_mean') or 1.5)
            new_cache[key] = (m, sc, float(row.get('beta_spy') or 0), float(row.get('avg_actual_mag') or 2.0), atr_norm, atr_mean)
        except Exception as e:
            print(f'[lgbm_daily] error H={key}: {e}', flush=True)
    _lgbm_daily_cache = new_cache
    _lgbm_daily_cache_ts = time.time()
    print(f'[lgbm_daily] loaded {len(new_cache)} global models', flush=True)
    return _lgbm_daily_cache


def _load_lgbm_daily_cluster_models_cached():
    """Daily LGBM cluster models. Keys: 'cluster:horizon'. Values: (model, scaler, beta)."""
    global _lgbm_daily_cluster_cache, _lgbm_daily_cluster_cache_ts
    if time.time() - _lgbm_daily_cluster_cache_ts < 600 and _lgbm_daily_cluster_cache:
        return _lgbm_daily_cluster_cache
    from supabase import create_client
    from sklearn.preprocessing import StandardScaler as _SS
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('lgbm_cluster_models_daily').select(
        'cluster_name,horizon_bucket,lgbm_model,feature_means,feature_stds,beta_spy,train_samples'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if not row.get('lgbm_model'):
            continue
        key = f"{row['cluster_name']}:{row['horizon_bucket']}"
        try:
            m = pickle.loads(base64.b64decode(row['lgbm_model']))
            means = row.get('feature_means') or []
            stds  = row.get('feature_stds')  or []
            if not means or not stds:
                continue
            sc = _SS()
            sc.mean_ = np.array(means); sc.scale_ = np.array(stds)
            sc.var_ = sc.scale_ ** 2; sc.n_samples_seen_ = int(row.get('train_samples') or 1)
            sc.n_features_in_ = len(means)
            new_cache[key] = (m, sc, float(row.get('beta_spy') or 0))
        except Exception as e:
            print(f'[lgbm_daily_cluster] error {key}: {e}', flush=True)
    _lgbm_daily_cluster_cache = new_cache
    _lgbm_daily_cluster_cache_ts = time.time()
    print(f'[lgbm_daily_cluster] loaded {len(new_cache)} cluster models', flush=True)
    return _lgbm_daily_cluster_cache


@app.route('/api/predict_lgbm_daily', methods=['POST', 'OPTIONS'])
def predict_lgbm_daily():
    """Daily LGBM inference endpoint. Called by crear-prediccion edge function."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json() or {}
    indicators  = body.get('indicators', {})
    horizon_bucket = body.get('horizon_bucket')
    ticker  = body.get('ticker', '')
    spy_pct = float(body.get('spy_pct') or 0)
    if horizon_bucket is None:
        return jsonify({'ok': False, 'error': 'horizon_bucket required'}), 400
    try:
        global_models  = _load_lgbm_daily_models_cached()
        cluster_models = _load_lgbm_daily_cluster_models_cached()
        # Snap any horizon to the nearest trained bucket (e.g. h=1 → 7, h=8 → 14)
        h_int = int(horizon_bucket)
        snapped = next((b for b in [1, 7, 14, 30, 60, 90] if h_int <= b), 90)
        h_key = str(snapped)
        if h_key not in global_models:
            # Fall back to the smallest available bucket
            h_key = min(global_models.keys(), key=lambda k: int(k)) if global_models else None
        if not h_key:
            return jsonify({'ok': False, 'error': f'No LGBM daily model trained yet'}), 404
        g_model, g_scaler, g_beta, avg_mag, g_atr_norm, g_atr_mean = global_models[h_key]
        feats = _extract_daily_features(indicators)
        if len(feats) != len(DAILY_FEATURE_NAMES):
            return jsonify({'ok': False, 'error': f'Feature dim mismatch: {len(feats)} vs {len(DAILY_FEATURE_NAMES)}'}), 500
        X = np.array([feats], dtype=float)
        current_atr = float(indicators.get('atr_pct') or g_atr_mean or 1.5)
        cluster = TICKER_CLUSTERS.get(ticker, '')
        c_key = f'{cluster}:{int(horizon_bucket)}' if cluster else None
        if c_key and c_key in cluster_models:
            c_model, c_scaler, c_beta = cluster_models[c_key]
            raw = float(c_model.predict(c_scaler.transform(X))[0])
            pred = (raw * current_atr if g_atr_norm else raw) + c_beta * spy_pct
            model_used = 'cluster'
        else:
            raw = float(g_model.predict(g_scaler.transform(X))[0])
            pred = (raw * current_atr if g_atr_norm else raw) + g_beta * spy_pct
            model_used = 'global'
        return jsonify({
            'ok': True,
            'predicted_pct': round(pred, 4),
            'horizon_bucket': int(horizon_bucket),
            'avg_actual_mag': round(avg_mag, 4),
            'model_used': model_used,
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def _get_market_session(minutes_since_open: float) -> str:
    if minutes_since_open < 30:  return 'open'
    if minutes_since_open < 120: return 'morning'
    if minutes_since_open < 270: return 'midday'
    return 'close'


# ── LightGBM inference endpoint ───────────────────────────────────────────────

@app.route('/api/predict_lgbm_intraday', methods=['POST', 'OPTIONS'])
def predict_lgbm_intraday():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json() or {}
    model_name = body.get('model_name')
    horizon_minutes = body.get('horizon_minutes')
    indicators = body.get('indicators', {})
    if not model_name or horizon_minutes is None:
        return jsonify({'ok': False, 'error': 'model_name and horizon_minutes required'}), 400
    try:
        models = _load_lgbm_models_cached()
        key = f'{model_name}:{int(horizon_minutes)}'
        if key not in models:
            return jsonify({'ok': False, 'error': 'No LightGBM model found for this model/horizon'}), 404
        m, beta = models[key]
        X = np.array([[float(indicators.get(fn) or 0) for fn in LR_FEATURE_NAMES]])
        atr_idx = LR_FEATURE_NAMES.index('atr_pct')
        atr_scale = max(0.1, float(X[0, atr_idx]))
        spy_r15 = float(indicators.get('spy_return_15m') or 0)
        pred = float(m.predict(X)[0]) * atr_scale + beta * spy_r15
        return jsonify({'ok': True, 'predicted_pct': round(pred, 4)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/predict_lgbm_all', methods=['POST', 'OPTIONS'])
def predict_lgbm_all():
    """Batch LightGBM inference — all (model, horizon) pairs in one call.
    Called once per asset by the edge function to avoid 39 individual HTTP requests."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json() or {}
    indicators = body.get('indicators', {})
    ticker = body.get('ticker', '')
    try:
        models = _load_lgbm_models_cached()
        if not models:
            return jsonify({'ok': True, 'predictions': {}, 'models_loaded': 0})
        X = np.array([[float(indicators.get(fn) or 0) for fn in LR_FEATURE_NAMES]])
        atr_idx = LR_FEATURE_NAMES.index('atr_pct')
        atr_scale = max(0.1, float(X[0, atr_idx]))
        spy_r15 = float(indicators.get('spy_return_15m') or 0)
        session_models = _load_lgbm_session_models_cached()
        ticker_models = _load_lgbm_ticker_models_cached()
        cluster_models = _load_lgbm_cluster_models_cached()
        mso = float(indicators.get('minutes_since_open') or 0)
        session = _get_market_session(mso)
        cluster = TICKER_CLUSTERS.get(ticker, '') if ticker else ''
        ticker_used = cluster_used = 0
        predictions = {}
        for key, (global_m, global_beta) in models.items():
            # Priority: ticker (blended w/ cluster if data-starved) > cluster > session > global
            horizon = key.split(':')[1]
            t_key = f'{ticker}:{horizon}' if ticker else None
            c_key = f'{cluster}:{horizon}' if cluster else None
            blended = False
            if t_key and t_key in ticker_models:
                t_model, t_beta, t_n = ticker_models[t_key]
                ticker_used += 1
                if c_key and c_key in cluster_models and t_n < 200:
                    # Continuous blend: weight ticker by its sample richness
                    c_model, c_beta = cluster_models[c_key]
                    w_t = t_n / 200.0
                    pred_t = float(t_model.predict(X)[0]) * atr_scale + t_beta * spy_r15
                    pred_c = float(c_model.predict(X)[0]) * atr_scale + c_beta * spy_r15
                    predictions[key] = round(w_t * pred_t + (1.0 - w_t) * pred_c, 4)
                    blended = True
                else:
                    m, beta = t_model, t_beta
            elif c_key and c_key in cluster_models:
                m, beta = cluster_models[c_key]
                cluster_used += 1
            else:
                sess_key = f'{key}:{session}'
                sess_entry = session_models.get(sess_key)
                if sess_entry:
                    m, beta = sess_entry
                else:
                    m, beta = global_m, global_beta
            if not blended:
                pred_idio = float(m.predict(X)[0]) * atr_scale
                predictions[key] = round(pred_idio + beta * spy_r15, 4)
        return jsonify({
            'ok': True, 'predictions': predictions,
            'models_loaded': len(models),
            'session': session,
            'ticker_models_used': ticker_used,
            'cluster_models_used': cluster_used,
            'session_models_used': sum(1 for k in models if f'{k}:{session}' in session_models),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── APScheduler: auto-train daily at 21:30 UTC ────────────────────────────────

def _keep_alive_loop(stop_event: threading.Event):
    """Pings our own /api/health every 4 min to prevent Render free-tier spin-down."""
    import urllib.request
    self_url = os.environ.get('RENDER_EXTERNAL_URL', '').rstrip('/')
    if not self_url:
        print('[keep_alive] RENDER_EXTERNAL_URL not set — skipping', flush=True)
        return
    ping_url = f'{self_url}/api/health'
    print(f'[keep_alive] started, pinging {ping_url} every 4 min', flush=True)
    while not stop_event.wait(timeout=240):  # wake every 4 minutes
        try:
            urllib.request.urlopen(ping_url, timeout=10)
            print('[keep_alive] ping ok', flush=True)
        except Exception as e:
            print(f'[keep_alive] ping failed: {e}', flush=True)
    print('[keep_alive] stopped', flush=True)


def _auto_train_all():
    """Run intraday + daily training sequentially — called by APScheduler or /api/auto_train."""
    stop_event = threading.Event()
    ka_thread = threading.Thread(target=_keep_alive_loop, args=(stop_event,), daemon=True)
    ka_thread.start()

    try:
        intra_id = str(uuid.uuid4())[:12]
        lr_training_jobs[intra_id] = {
            'status': 'starting', 'models_done': 0, 'models_total': 0,
            'models_trained': 0, 'total_samples': 0, 'results': {},
            'start_time': time.time(), 'error': None,
        }
        _run_lr_training(intra_id)

        daily_id = str(uuid.uuid4())[:12]
        daily_training_jobs[daily_id] = {
            'status': 'starting', 'models_done': 0, 'models_total': 5,
            'models_trained': 0, 'total_samples': 0,
            'start_time': time.time(), 'error': None,
        }
        _run_lr_training_daily(daily_id)
    finally:
        stop_event.set()  # stop keep-alive regardless of success/failure


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'ok': True})


# ── Historical OHLCV loader ───────────────────────────────────────────────────

def _run_load_historical_ohlcv(job_id: str):
    """Download 3+ years of daily OHLCV from yfinance and INSERT (ignore dups) into price_history."""
    import yfinance as yf
    import pandas as pd
    from supabase import create_client

    job = historical_load_jobs[job_id]
    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        a_resp = sb.from_('assets').select('id, ticker').execute()
        assets = a_resp.data or []
        ticker_to_id = {a['ticker']: a['id'] for a in assets}
        tickers = [a['ticker'] for a in assets]

        job['tickers_total'] = len(tickers)
        job['status'] = 'downloading'
        print(f'[hist_load] Starting for {len(tickers)} tickers', flush=True)

        # Load from 2022-01-01 up to (not including) the first date of existing live data
        START_DATE = '2022-01-01'
        END_DATE   = '2025-01-27'  # exclusive — live data starts here
        BATCH_SIZE = 500
        rows_inserted = 0
        errors = []

        for i, ticker in enumerate(tickers):
            job['tickers_done'] = i
            try:
                df = yf.download(
                    ticker, start=START_DATE, end=END_DATE,
                    auto_adjust=True, progress=False,
                )
                if df is None or df.empty:
                    print(f'[hist_load] {ticker}: no data', flush=True)
                    continue

                # yfinance v0.2 may return MultiIndex columns — flatten
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] for col in df.columns]

                asset_id = ticker_to_id.get(ticker)
                if not asset_id:
                    print(f'[hist_load] {ticker}: no asset_id in DB', flush=True)
                    continue

                def _safe_float(val, fallback):
                    try:
                        f = float(val)
                        return f if f > 0 and not math.isnan(f) else fallback
                    except (TypeError, ValueError):
                        return fallback

                rows = []
                for date, row in df.iterrows():
                    raw_close = row.get('Close') or row.get('close') or 0
                    try:
                        close_val = float(raw_close)
                    except (TypeError, ValueError):
                        continue
                    if close_val <= 0 or math.isnan(close_val):
                        continue

                    vol = row.get('Volume') or row.get('volume') or 0
                    try:
                        vol_int = int(float(vol)) if vol else 0
                    except (TypeError, ValueError):
                        vol_int = 0

                    rows.append({
                        'asset_id':   asset_id,
                        'trade_date': date.strftime('%Y-%m-%d'),
                        'open':       _safe_float(row.get('Open'), close_val),
                        'high':       _safe_float(row.get('High'), close_val),
                        'low':        _safe_float(row.get('Low'), close_val),
                        'close':      close_val,
                        'volume':     vol_int,
                        'adj_close':  close_val,
                    })

                if not rows:
                    continue

                for bs in range(0, len(rows), BATCH_SIZE):
                    batch = rows[bs:bs + BATCH_SIZE]
                    sb.table('price_history').insert(batch, ignore_duplicates=True).execute()
                    rows_inserted += len(batch)

                job['rows_inserted'] = rows_inserted
                print(f'[hist_load] {ticker}: +{len(rows)} rows (total {rows_inserted})', flush=True)

            except Exception as e:
                msg = f'{ticker}: {e}'
                errors.append(msg)
                print(f'[hist_load] ERROR {msg}', flush=True)

        job['tickers_done'] = len(tickers)
        job['rows_inserted'] = rows_inserted
        job['errors'] = errors
        job['status'] = 'done'
        print(f'[hist_load] Done — {rows_inserted} rows, {len(errors)} errors', flush=True)

    except Exception as e:
        import traceback
        job['status'] = 'error'
        job['error'] = str(e)
        job['trace'] = traceback.format_exc()[-2000:]
        print(f'[hist_load] FATAL: {e}', flush=True)


@app.route('/api/load_historical_ohlcv', methods=['POST', 'OPTIONS'])
def load_historical_ohlcv():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    job_id = str(uuid.uuid4())[:12]
    historical_load_jobs[job_id] = {
        'status': 'starting', 'tickers_done': 0, 'tickers_total': 0,
        'rows_inserted': 0, 'errors': [], 'start_time': time.time(),
    }
    threading.Thread(target=_run_load_historical_ohlcv, args=(job_id,), daemon=True).start()
    return jsonify({'ok': True, 'job_id': job_id})


@app.route('/api/load_historical_ohlcv_status/<job_id>', methods=['GET'])
def load_historical_ohlcv_status(job_id):
    job = historical_load_jobs.get(job_id)
    if not job:
        return jsonify({'ok': False, 'error': 'not found'}), 404
    elapsed = round(time.time() - job['start_time'], 1)
    return jsonify({'ok': True, 'elapsed_s': elapsed, **job})


@app.route('/api/auto_train', methods=['POST', 'OPTIONS'])
def auto_train():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    threading.Thread(target=_auto_train_all, daemon=True).start()
    return jsonify({'ok': True, 'message': 'intraday + daily training started'})


@app.route('/api/hist_sample_test', methods=['POST', 'OPTIONS'])
def hist_sample_test():
    """Diagnóstico paso a paso de _build_historical_samples."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403

    import traceback
    result = {'step': 'start'}
    try:
        from supabase import create_client as _cc
        sb2 = _cc(SUPABASE_URL, SUPABASE_KEY)
        result['step'] = 'supabase_ok'

        # Step 1: fetch assets
        a_resp = sb2.from_('assets').select('id, ticker').execute()
        asset_map2 = {a['id']: a['ticker'] for a in (a_resp.data or [])}
        result['assets_count'] = len(asset_map2)
        result['step'] = 'assets_ok'

        # Step 2: fetch first page of price_history
        ph_resp = sb2.from_('price_history').select(
            'asset_id, trade_date, open, high, low, close, volume'
        ).range(0, 999).execute()
        ph_rows = ph_resp.data or []
        result['ph_first_page'] = len(ph_rows)
        result['step'] = 'ph_page1_ok'

        # Step 3: count tickers in first page
        tickers_in_page = set(asset_map2.get(r['asset_id']) for r in ph_rows if r.get('asset_id'))
        result['tickers_in_page1'] = len(tickers_in_page)
        result['sample_tickers'] = list(tickers_in_page)[:5]
        result['step'] = 'count_ok'

        # Step 4: run full historical build
        samples = _build_historical_samples(sb2)
        from collections import Counter
        by_h = Counter(s['horizon_bucket'] for s in samples)
        result['total_samples'] = len(samples)
        result['by_horizon'] = dict(by_h)
        result['step'] = 'done'

    except Exception as e:
        result['error'] = str(e)
        result['trace'] = traceback.format_exc()[-1000:]

    return jsonify({'ok': result.get('step') == 'done', **result})


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_auto_train_all, 'cron', hour=21, minute=30, timezone='UTC', id='auto_train_daily')
    _scheduler.start()
    print('[scheduler] APScheduler started — auto-training at 21:30 UTC daily', flush=True)
except Exception as _sched_err:
    print(f'[scheduler] WARNING: could not start APScheduler: {_sched_err}', flush=True)
