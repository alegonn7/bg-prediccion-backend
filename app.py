from flask import Flask, request, jsonify
import os
import pickle
import base64
import numpy as np
import math
from datetime import datetime, timedelta
from collections import defaultdict

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

def train_model(model_name: str) -> dict:
    from supabase import create_client
    import xgboost as xgb

    print(f'[train] START model={model_name}', flush=True)
    print(f'[train] SUPABASE_URL set={bool(SUPABASE_URL)} KEY set={bool(SUPABASE_KEY)}', flush=True)

    if model_name not in MODEL_FEATURE_NAMES:
        raise ValueError(f'Unknown model: {model_name}')

    feature_names = MODEL_FEATURE_NAMES[model_name]
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch all price history (paginated — Supabase caps at 1000 rows/request)
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

    # Group and sort by asset; normalize trade_date to string
    asset_rows: dict = defaultdict(list)
    for row in all_rows:
        if not isinstance(row['trade_date'], str):
            row['trade_date'] = str(row['trade_date'])
        asset_rows[row['asset_id']].append(row)
    for aid in asset_rows:
        asset_rows[aid].sort(key=lambda r: r['trade_date'])

    print(f'[train] assets grouped: {len(asset_rows)}', flush=True)
    # Free raw list to save RAM
    del all_rows

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

        train_acc  = float((model.predict(X) == y).mean())
        model_b64  = base64.b64encode(pickle.dumps(model)).decode()

        # Fetch previous accuracy before overwriting
        try:
            old_row = sb.table('xgb_models').select(
                'train_accuracy,train_samples'
            ).eq('model_name', model_name).eq('horizon_bucket', bucket).maybe_single().execute()
            old_acc  = float(old_row.data['train_accuracy']) if old_row.data and old_row.data.get('train_accuracy') is not None else None
            old_samp = int(old_row.data['train_samples'])    if old_row.data and old_row.data.get('train_samples')   is not None else None
        except Exception:
            old_acc, old_samp = None, None

        sb.table('xgb_models').upsert({
            'model_name':     model_name,
            'horizon_bucket': bucket,
            'model_data':     model_b64,
            'feature_names':  feature_names,
            'train_accuracy': train_acc,
            'train_samples':  len(X_rows),
        }, on_conflict='model_name,horizon_bucket').execute()

        # Record training history
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
        }

    return {'model_name': model_name, 'buckets': bucket_results}


# ── Prediction ────────────────────────────────────────────────────────────────

def run_predictions() -> dict:
    from supabase import create_client

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    models_rows = sb.table('xgb_models').select(
        'model_name,horizon_bucket,model_data,feature_names'
    ).execute().data or []

    if not models_rows:
        return {'predictions': 0, 'reason': 'No trained XGBoost models found'}

    models: dict = {}
    for row in models_rows:
        key = (row['model_name'], int(row['horizon_bucket']))
        model = pickle.loads(base64.b64decode(row['model_data']))
        models[key] = {'model': model, 'feature_names': row['feature_names']}

    cols = (
        'asset_id,computed_date,price_close,'
        'price_vs_sma20,price_vs_sma50,price_vs_sma200,'
        'rsi_14,macd_histogram,macd_signal,bb_pct_b,bb_squeeze,'
        'atr_pct,hist_vol_20,obv_trend,roc_5,roc_10,roc_20,'
        'candle_signal,adx_14,assets(ticker,is_active)'
    )
    today     = datetime.now().strftime('%Y-%m-%d')
    indicators = sb.table('indicators').select(cols).eq('computed_date', today).execute().data or []

    if not indicators:
        yesterday  = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        indicators = sb.table('indicators').select(cols).eq('computed_date', yesterday).execute().data or []

    if not indicators:
        return {'predictions': 0, 'reason': 'No indicator data for today'}

    pred_date = indicators[0].get('computed_date', today)

    rows_to_upsert = []
    for ind_row in indicators:
        asset_info = ind_row.get('assets') or {}
        ticker = asset_info.get('ticker', '')
        if not ticker:
            continue

        feats_all = extract_features(ind_row)

        for (model_name, bucket), model_info in models.items():
            feature_names = model_info['feature_names']
            model         = model_info['model']
            x = np.array([[feats_all.get(f, 0.0) for f in feature_names]], dtype=np.float32)
            prob_up = float(model.predict_proba(x)[0][1])

            rows_to_upsert.append({
                'ticker':          ticker,
                'model_name':      model_name,
                'horizon_bucket':  bucket,
                'probability_up':  round(prob_up, 6),
                'prediction_date': pred_date,
            })

    CHUNK = 500
    for i in range(0, len(rows_to_upsert), CHUNK):
        sb.table('xgb_daily_predictions').upsert(
            rows_to_upsert[i:i + CHUNK],
            on_conflict='ticker,model_name,horizon_bucket,prediction_date'
        ).execute()

    return {
        'predictions': len(rows_to_upsert),
        'date':        pred_date,
        'assets':      len(indicators),
        'models':      len(models),
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


@app.route('/api/predict_xgb', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    try:
        result = run_predictions()
        return jsonify({'ok': True, **result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
