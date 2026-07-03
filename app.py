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

training_jobs: dict = {}   # job_id -> live status dict
prediction_jobs: dict = {} # job_id -> live status dict

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
    # Phase 2 features re-added once >=200 live samples with non-null values exist
    # 'spy_return_15m', 'spy_return_session', 'premarket_gap', 'prev_day_return',
]

lr_training_jobs: dict = {}


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

    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)

        all_rows = []
        batch_size = 1000
        offset = 0
        job['status'] = 'fetching'
        while True:
            resp = sb.rpc('get_intraday_training_data').range(offset, offset + batch_size - 1).execute()
            batch = resp.data or []
            all_rows.extend(batch)
            print(f'[lr_train] fetched offset={offset} got={len(batch)} total={len(all_rows)}', flush=True)
            if len(batch) < batch_size:
                break
            offset += batch_size

        job['total_samples'] = len(all_rows)
        if not all_rows:
            job['status'] = 'done'
            job['models_trained'] = 0
            return

        now_utc = datetime.now(timezone.utc)
        all_rows.sort(key=lambda r: _parse_ts(r.get('created_at')))
        holdout_cutoff = now_utc - timedelta(days=30)

        def decay_w(ts_str):
            age = (now_utc - _parse_ts(ts_str)).total_seconds() / 86400
            return math.exp(-lam * max(0.0, age))

        # Group chronologically by (model_name, horizon_minutes)
        groups: dict = {}
        for row in all_rows:
            key = (row['model_name'], int(row['horizon_minutes']))
            if key not in groups:
                groups[key] = {'X': [], 'y_dir': [], 'y_signed': [], 'y_mag': [], 'w': [], 'ts': []}
            groups[key]['X'].append([float(row.get(fn) or 0) for fn in LR_FEATURE_NAMES])
            groups[key]['y_dir'].append(1 if row['direction_correct'] else 0)
            groups[key]['y_signed'].append(row.get('actual_signed_pct'))
            groups[key]['y_mag'].append(row.get('actual_magnitude'))
            groups[key]['w'].append(decay_w(row.get('created_at')))
            groups[key]['ts'].append(row.get('created_at'))

        job['status'] = 'training'
        job['models_total'] = len(groups)
        upserts = []
        results = {}

        for (model_name, horizon_minutes), data in groups.items():
            n = len(data['X'])
            if n < 20:
                continue

            X_np = np.array(data['X'], dtype=float)
            y_dir_np = np.array(data['y_dir'], dtype=float)
            y_signed_np = np.array([float(v) if v is not None else float('nan') for v in data['y_signed']])
            y_mag_np = np.array([float(v) if v is not None else float('nan') for v in data['y_mag']])
            w_np = np.array(data['w'], dtype=float)

            # Walk-forward: holdout = last 30 days (never used for training)
            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in data['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() >= 20:
                X_tv = X_np[tv_mask]; y_dir_tv = y_dir_np[tv_mask]
                y_signed_tv = y_signed_np[tv_mask]; y_mag_tv = y_mag_np[tv_mask]; w_tv = w_np[tv_mask]
            else:
                X_tv, y_dir_tv, y_signed_tv, y_mag_tv, w_tv = X_np, y_dir_np, y_signed_np, y_mag_np, w_np

            split = max(10, int(len(X_tv) * 0.8))
            X_train, X_val = X_tv[:split], X_tv[split:]
            y_dir_train = y_dir_tv[:split]
            y_signed_train = y_signed_tv[:split]
            y_signed_val = y_signed_tv[split:]
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

            # LightGBM — target normalized by ATR so model learns in ATR units.
            # Inference denormalizes: pred_pct = model.predict(X) * current_atr_pct.
            # This makes predictions comparable across high/low volatility regimes.
            lgbm_model_b64 = lgbm_val_mae = lgbm_importance = None
            if Xs is not None and len(Xs) >= 30:
                atr_idx = LR_FEATURE_NAMES.index('atr_pct')
                # Raw ATR for train samples (before scaling), clipped to [0.1, 10]
                atr_train_raw = np.clip(X_train[train_signed_mask][:, atr_idx], 0.1, 10.0)
                ys_norm = ys / atr_train_raw

                val_sm = ~np.isnan(y_signed_val) if len(X_val_s) > 0 else np.zeros(0, dtype=bool)
                eval_set_norm = None
                if val_sm.sum() >= 5:
                    atr_val_raw = np.clip(X_val[:, atr_idx][val_sm], 0.1, 10.0)
                    y_val_norm = y_signed_val[val_sm] / atr_val_raw
                    eval_set_norm = [(X_val_s[val_sm], y_val_norm)]

                callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_norm else None
                lgb_reg = lgb.LGBMRegressor(
                    n_estimators=500, learning_rate=0.05, num_leaves=31,
                    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1,
                    objective='regression',
                )
                lgb_reg.fit(Xs, ys_norm, sample_weight=ws, eval_set=eval_set_norm, callbacks=callbacks)
                if val_sm.sum() > 0:
                    atr_val_raw = np.clip(X_val[:, atr_idx][val_sm], 0.1, 10.0)
                    preds_denorm = lgb_reg.predict(X_val_s[val_sm]) * atr_val_raw
                    lgbm_val_mae = float(np.mean(np.abs(y_signed_val[val_sm] - preds_denorm)))
                lgbm_model_b64 = base64.b64encode(pickle.dumps(lgb_reg)).decode('utf-8')
                lgbm_importance = dict(zip(LR_FEATURE_NAMES, lgb_reg.feature_importances_.tolist()))

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
            })
            results[f'{model_name}:{horizon_minutes}'] = {
                'samples': len(X_tv), 'accuracy': round(accuracy, 3),
                'avg_mag': round(avg_mag, 3) if avg_mag else None,
                'val_mae_ridge': round(val_mae_ridge, 3) if val_mae_ridge else None,
                'lgbm_val_mae': round(lgbm_val_mae, 3) if lgbm_val_mae else None,
            }
            job['models_done'] = len(upserts)
            print(
                f'[lr_train] {model_name}:{horizon_minutes} n={len(X_tv)} acc={accuracy:.3f} '
                f'val_mae_ridge={val_mae_ridge} lgbm_val_mae={lgbm_val_mae}',
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

        session_groups: dict = {}
        for row in all_rows:
            mso = float(row.get('minutes_since_open') or 0)
            sess = _session_from_mso(mso)
            key = (row['model_name'], int(row['horizon_minutes']), sess)
            if key not in session_groups:
                session_groups[key] = {'X': [], 'y_signed': [], 'w': [], 'ts': []}
            session_groups[key]['X'].append([float(row.get(fn) or 0) for fn in LR_FEATURE_NAMES])
            session_groups[key]['y_signed'].append(row.get('actual_signed_pct'))
            session_groups[key]['w'].append(decay_w(row.get('created_at')))
            session_groups[key]['ts'].append(row.get('created_at'))

        session_upserts = []
        atr_idx = LR_FEATURE_NAMES.index('atr_pct')
        for (model_name, horizon_minutes, session), sdata in session_groups.items():
            X_np = np.array(sdata['X'], dtype=float)
            y_np = np.array([float(v) if v is not None else float('nan') for v in sdata['y_signed']])
            w_np = np.array(sdata['w'], dtype=float)

            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in sdata['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() < 20:
                tv_mask = np.ones(len(X_np), dtype=bool)

            X_tv = X_np[tv_mask]; y_tv = y_np[tv_mask]; w_tv = w_np[tv_mask]
            split = max(10, int(len(X_tv) * 0.8))
            X_tr, X_v = X_tv[:split], X_tv[split:]
            y_tr, y_v = y_tv[:split], y_tv[split:]
            w_tr = w_tv[:split]

            tr_mask = ~np.isnan(y_tr)
            if tr_mask.sum() < 20:
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
                objective='quantile', alpha=0.5,
            )
            lgb_s.fit(Xs_s, ys_norm_s, sample_weight=ws_s, eval_set=eval_set_s, callbacks=cbs)

            sess_val_mae = None
            if val_sm_s.sum() > 0:
                atr_v2 = np.clip(X_v[:, atr_idx][val_sm_s], 0.1, 10.0)
                preds_d = lgb_s.predict(X_v_s[val_sm_s]) * atr_v2
                sess_val_mae = float(np.mean(np.abs(y_v[val_sm_s] - preds_d)))

            session_upserts.append({
                'model_name': model_name,
                'horizon_minutes': horizon_minutes,
                'market_session': session,
                'lgbm_model': base64.b64encode(pickle.dumps(lgb_s)).decode('utf-8'),
                'lgbm_val_mae': sess_val_mae,
                'lgbm_feature_importance': dict(zip(LR_FEATURE_NAMES, lgb_s.feature_importances_.tolist())),
                'train_samples': int(tr_mask.sum()),
                'last_updated': now_utc.isoformat(),
            })
            print(f'[lr_train:session] {model_name}:{horizon_minutes}:{session} n={tr_mask.sum()} val_mae={sess_val_mae}', flush=True)

        for su in session_upserts:
            sb.table('lgbm_session_models_intraday').upsert(
                su, on_conflict='model_name,horizon_minutes,market_session'
            ).execute()
        print(f'[lr_train] session models: {len(session_upserts)} trained', flush=True)
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


# ── Daily signed Ridge training ───────────────────────────────────────────────

DAILY_FEATURE_NAMES = [
    'price_vs_sma20', 'price_vs_sma50', 'price_vs_sma200',
    'rsi_norm', 'macd_norm', 'bb_pct_b_norm', 'bb_squeeze',
    'atr_pct_norm', 'hist_vol_norm', 'adx_norm',
    'roc_5_norm', 'roc_10_norm', 'roc_20_norm',
    'candle_signal', 'obv_trend',
    'month_sin', 'month_cos',
]


def _extract_daily_features(row: dict) -> list:
    def cl3(v, lo=-3.0, hi=3.0): return max(lo, min(hi, float(v)))
    vs20  = float(row.get('price_vs_sma20',  0) or 0)
    vs50  = float(row.get('price_vs_sma50',  0) or 0)
    vs200 = float(row.get('price_vs_sma200', 0) or 0)
    rsi   = float(row.get('rsi_14', 50) or 50)
    macdH = float(row.get('macd_histogram', 0) or 0)
    bbB   = float(row.get('bb_pct_b', 0.5) or 0.5)
    bbs   = 1.0 if row.get('bb_squeeze') else 0.0
    atrP  = float(row.get('atr_pct', 1) or 1)
    hv    = float(row.get('hist_vol_20', 20) or 20)
    adx   = float(row.get('adx_14', 20) or 20)
    roc5  = float(row.get('roc_5',  0) or 0)
    roc10 = float(row.get('roc_10', 0) or 0)
    roc20 = float(row.get('roc_20', 0) or 0)
    cand_s = (row.get('candle_signal') or 'neutral').lower()
    cand  = 1.0 if cand_s == 'bullish' else (-1.0 if cand_s == 'bearish' else 0.0)
    obv_s = (row.get('obv_trend') or 'flat').lower()
    obv   = 1.0 if obv_s == 'rising' else (-1.0 if obv_s == 'falling' else 0.0)
    month = int(row.get('created_month') or 1)
    return [
        cl3(vs20 / 5), cl3(vs50 / 10), cl3(vs200 / 20),
        (rsi - 50) / 25,
        cl3(macdH / (abs(macdH) + 0.01)),
        bbB * 2 - 1, bbs,
        cl3(atrP / 3, 0.0, 3.0),
        cl3(hv / 50, 0.0, 3.0),
        cl3(adx / 50 - 0.4),
        cl3(roc5 / 5), cl3(roc10 / 10), cl3(roc20 / 20),
        cand, obv,
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
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
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return ts if getattr(ts, 'tzinfo', None) else ts.replace(tzinfo=timezone.utc)

    try:
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        job['status'] = 'fetching'

        resp = sb.rpc('get_daily_training_data', {'p_limit': 100000}).execute()
        all_rows = resp.data or []
        job['total_samples'] = len(all_rows)
        print(f'[lr_train_daily] fetched {len(all_rows)} rows', flush=True)

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

        BUCKETS = [7, 14, 30, 60, 90]
        groups: dict = {b: {'X': [], 'y': [], 'w': [], 'ts': []} for b in BUCKETS}
        for row in all_rows:
            h = int(row.get('horizon_bucket') or 0)
            if h not in groups:
                continue
            signed_pct = row.get('actual_signed_pct')
            if signed_pct is None:
                continue
            feats = _extract_daily_features(row)
            if len(feats) != len(DAILY_FEATURE_NAMES):
                continue
            groups[h]['X'].append(feats)
            groups[h]['y'].append(float(signed_pct))
            groups[h]['w'].append(decay_w(row.get('created_at')))
            groups[h]['ts'].append(row.get('created_at'))

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
            w_np = np.array(groups[bucket]['w'], dtype=float)

            # Walk-forward: holdout = last 30 days
            holdout_mask = np.array([_parse_ts(ts) >= holdout_cutoff for ts in groups[bucket]['ts']])
            tv_mask = ~holdout_mask
            if tv_mask.sum() >= 20:
                X_tv, y_tv, w_tv = X_np[tv_mask], y_np[tv_mask], w_np[tv_mask]
            else:
                X_tv, y_tv, w_tv = X_np, y_np, w_np

            split = max(10, int(len(X_tv) * 0.8))
            X_train, X_val = X_tv[:split], X_tv[split:]
            y_train, y_val = y_tv[:split], y_tv[split:]
            w_train = w_tv[:split]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_val_s = scaler.transform(X_val) if len(X_val) > 0 else np.empty((0, X_train.shape[1]))

            # Ridge signed — kept for backward compat
            reg = Ridge(alpha=1.0)
            reg.fit(X_train_s, y_train, sample_weight=w_train)
            r2_val = float(r2_score(y_train, reg.predict(X_train_s)))
            avg_mag = float(np.mean(np.abs(y_tv)))
            median_mag = float(np.median(np.abs(y_tv)))
            val_mae_ridge = None
            if len(X_val_s) > 0:
                val_mae_ridge = float(np.mean(np.abs(y_val - reg.predict(X_val_s))))

            # LightGBM daily — early stopping against val set, lower lr for stability
            lgbm_model_b64 = lgbm_val_mae = None
            if len(X_train_s) >= 30:
                eval_set_d = [(X_val_s, y_val)] if len(X_val_s) >= 5 else None
                callbacks_d = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_d else None
                lgb_reg = lgb.LGBMRegressor(
                    n_estimators=600, learning_rate=0.03, num_leaves=31,
                    min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbose=-1,
                    objective='regression',
                )
                lgb_reg.fit(X_train_s, y_train, sample_weight=w_train, eval_set=eval_set_d, callbacks=callbacks_d)
                if len(X_val_s) > 0:
                    lgbm_val_mae = float(np.mean(np.abs(y_val - lgb_reg.predict(X_val_s))))
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
                'train_samples': len(X_tv),
                'lgbm_model': lgbm_model_b64,
                'lgbm_val_mae': lgbm_val_mae,
                'val_mae_ridge': val_mae_ridge,
            })
            job['models_done'] += 1
            print(
                f'[lr_train_daily] H={bucket}: n={len(X_tv)} r2={r2_val:.3f} '
                f'val_mae_ridge={val_mae_ridge} lgbm_val_mae={lgbm_val_mae}',
                flush=True,
            )

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


def _load_lgbm_models_cached():
    global _lgbm_cache, _lgbm_cache_ts
    if time.time() - _lgbm_cache_ts < 600 and _lgbm_cache:
        return _lgbm_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('model_learned_params_intraday').select('model_name,horizon_minutes,lgbm_model').execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['model_name']}:{row['horizon_minutes']}"
            try:
                new_cache[key] = pickle.loads(base64.b64decode(row['lgbm_model']))
            except Exception:
                pass
    _lgbm_cache = new_cache
    _lgbm_cache_ts = time.time()
    return _lgbm_cache


def _load_lgbm_session_models_cached():
    """Load per-session LGBM models. Keys: 'model_name:horizon:session'."""
    global _lgbm_session_cache, _lgbm_session_cache_ts
    if time.time() - _lgbm_session_cache_ts < 600 and _lgbm_session_cache:
        return _lgbm_session_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('lgbm_session_models_intraday').select(
        'model_name,horizon_minutes,market_session,lgbm_model'
    ).execute()
    new_cache: dict = {}
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['model_name']}:{row['horizon_minutes']}:{row['market_session']}"
            try:
                new_cache[key] = pickle.loads(base64.b64decode(row['lgbm_model']))
            except Exception:
                pass
    _lgbm_session_cache = new_cache
    _lgbm_session_cache_ts = time.time()
    return _lgbm_session_cache


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
        X = np.array([[float(indicators.get(fn) or 0) for fn in LR_FEATURE_NAMES]])
        pred = float(models[key].predict(X)[0])
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
    try:
        models = _load_lgbm_models_cached()
        if not models:
            return jsonify({'ok': True, 'predictions': {}, 'models_loaded': 0})
        X = np.array([[float(indicators.get(fn) or 0) for fn in LR_FEATURE_NAMES]])
        # Denormalize: models trained on y/atr_pct, so pred_pct = pred_norm * atr_pct
        atr_idx = LR_FEATURE_NAMES.index('atr_pct')
        atr_scale = max(0.1, float(X[0, atr_idx]))
        # Prefer session-specific model when available (Step 7: modelos por sesión)
        session_models = _load_lgbm_session_models_cached()
        mso = float(indicators.get('minutes_since_open') or 0)
        session = _get_market_session(mso)
        predictions = {}
        for key, global_m in models.items():
            sess_key = f'{key}:{session}'
            m = session_models.get(sess_key, global_m)
            predictions[key] = round(float(m.predict(X)[0]) * atr_scale, 4)
        return jsonify({
            'ok': True, 'predictions': predictions,
            'models_loaded': len(models),
            'session': session,
            'session_models_used': sum(1 for k in models if f'{k}:{session}' in session_models),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── APScheduler: auto-train daily at 21:30 UTC ────────────────────────────────

def _auto_train_all():
    """Run intraday + daily training sequentially — called by APScheduler."""
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


try:
    from apscheduler.schedulers.background import BackgroundScheduler
    _scheduler = BackgroundScheduler()
    _scheduler.add_job(_auto_train_all, 'cron', hour=21, minute=30, timezone='UTC', id='auto_train_daily')
    _scheduler.start()
    print('[scheduler] APScheduler started — auto-training at 21:30 UTC daily', flush=True)
except Exception as _sched_err:
    print(f'[scheduler] WARNING: could not start APScheduler: {_sched_err}', flush=True)
