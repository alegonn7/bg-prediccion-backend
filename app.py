from flask import Flask, request, jsonify
import os
import pickle
import base64
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import math

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

MIN_MOVE_PCT = {7: 0.3, 14: 0.5, 30: 0.8, 60: 1.2, 90: 1.5}
HORIZON_BUCKETS = [7, 14, 30, 60, 90]


def cl(v, scale):
    return max(-1.0, min(1.0, float(v or 0) / scale)) if scale else 0.0


def cl3(v, lo=-3.0, hi=3.0):
    return max(lo, min(hi, float(v or 0)))


# ── Indicator helpers ────────────────────────────────────────────────────────

def _ema(prices, period):
    k = 2.0 / (period + 1)
    val = float(prices[0])
    for p in prices[1:]:
        val = float(p) * k + val * (1 - k)
    return val


def _rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    avg_gain = np.maximum(deltas, 0).mean()
    avg_loss = np.maximum(-deltas, 0).mean()
    if avg_loss < 1e-10:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


def compute_features_from_ohlcv(rows, trade_date):
    """Compute all model features from raw OHLCV rows (oldest→newest)."""
    n = len(rows)
    if n < 21:
        return None

    closes  = np.array([float(r.get('adj_close') or r.get('close') or 0) for r in rows], dtype=np.float64)
    highs   = np.array([float(r.get('high') or closes[i]) for i, r in enumerate(rows)], dtype=np.float64)
    lows    = np.array([float(r.get('low') or closes[i]) for i, r in enumerate(rows)], dtype=np.float64)
    opens   = np.array([float(r.get('open') or closes[i]) for i, r in enumerate(rows)], dtype=np.float64)
    volumes = np.array([float(r.get('volume') or 0) for r in rows], dtype=np.float64)

    close = closes[-1]
    if close <= 0:
        return None

    # SMA ratios
    sma20  = closes[-min(20, n):].mean()
    sma50  = closes[-min(50, n):].mean()
    sma200 = closes[-min(200, n):].mean()
    vs20  = cl((close - sma20)  / sma20  * 100 if sma20  > 0 else 0, 20)
    vs50  = cl((close - sma50)  / sma50  * 100 if sma50  > 0 else 0, 20)
    vs200 = cl((close - sma200) / sma200 * 100 if sma200 > 0 else 0, 30)

    # RSI
    rsi_norm = (_rsi(closes) - 50) / 25.0

    # ROC
    roc5  = cl((close / closes[-6]  - 1) * 100 if n >= 6  and closes[-6]  > 0 else 0, 20)
    roc10 = cl((close / closes[-11] - 1) * 100 if n >= 11 and closes[-11] > 0 else 0, 30)
    roc20 = cl((close / closes[-21] - 1) * 100 if n >= 21 and closes[-21] > 0 else 0, 40)

    # MACD (EMA12 - EMA26)
    ema_window = closes[-min(52, n):]
    ema12 = _ema(ema_window, 12)
    ema26 = _ema(ema_window, 26) if len(ema_window) >= 26 else _ema(ema_window, len(ema_window))
    macd_h = ema12 - ema26
    macd_norm = cl3(macd_h / (abs(macd_h) + 0.01)) * 0.33

    # Bollinger Bands (20-day, 2σ)
    w = min(20, n)
    std20 = closes[-w:].std() if w >= 2 else 0.0
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    rng = upper - lower
    bb_pct_b = (close - lower) / rng if rng > 1e-10 else 0.5
    bb_pos = bb_pct_b * 2 - 1
    if n >= 40:
        avg_std = np.array([closes[-40 + i: -40 + i + 20].std() for i in range(20)]).mean()
        bb_squeeze = 1.0 if std20 < 0.75 * avg_std and avg_std > 0 else 0.0
    else:
        bb_squeeze = 0.0

    # ATR (14-day)
    w14 = min(14, n - 1)
    if w14 > 0:
        tr = np.maximum(
            highs[-w14:] - lows[-w14:],
            np.maximum(
                np.abs(highs[-w14:] - closes[-w14 - 1:-1]),
                np.abs(lows[-w14:]  - closes[-w14 - 1:-1])
            )
        )
        atr_pct = tr.mean() / close * 100
    else:
        atr_pct = 1.0
    atr_norm = cl(atr_pct, 5)

    # Historical volatility (20-day annualised)
    w20 = min(21, n)
    hv = closes[-w20:].std() / close * math.sqrt(252) * 100 if w20 >= 2 else 20.0
    hv_norm = cl(hv, 100)

    # OBV direction (5-day)
    obv = 0.0
    for i in range(max(-5, -n + 1), 0):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]
        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]
    obv_dir = 1.0 if obv > 0 else (-1.0 if obv < 0 else 0.0)

    # Candle direction
    candle = 1.0 if closes[-1] > opens[-1] else -1.0

    # ADX (simplified 14-day)
    if n >= 15:
        h14 = highs[-15:]
        l14 = lows[-15:]
        c14 = closes[-15:]
        pdm = np.maximum(np.diff(h14), 0)
        mdm = np.maximum(-np.diff(l14), 0)
        tr14 = np.maximum(
            h14[1:] - l14[1:],
            np.maximum(np.abs(h14[1:] - c14[:-1]), np.abs(l14[1:] - c14[:-1]))
        )
        atr14 = tr14.mean()
        if atr14 > 0:
            pdi = 100 * pdm.mean() / atr14
            mdi = 100 * mdm.mean() / atr14
            adx_val = 100 * abs(pdi - mdi) / (pdi + mdi + 1e-10)
        else:
            adx_val = 20.0
        adx_norm = cl(adx_val, 50) - 0.4
    else:
        adx_norm = -0.4
    adx_dir = adx_norm

    # Seasonality
    try:
        month = datetime.strptime(trade_date, '%Y-%m-%d').month
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
        'adx_norm': adx_norm, 'adx_dir': adx_dir,
        'neg_vs20': -vs20, 'neg_bb': -bb_pos,
        'neg_rsi': -rsi_norm, 'neg_vs50': -vs50,
        'sin_month': sin_month, 'cos_month': cos_month,
    }


# ── Extract features from indicators row (for daily prediction) ──────────────

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

    bb_b   = float(ind.get('bb_pct_b', 0.5) or 0.5)
    bb_pos = bb_b * 2 - 1
    bb_squeeze = 1.0 if ind.get('bb_squeeze') else 0.0

    atr_norm = cl(ind.get('atr_pct', 1), 5)
    hv_norm  = cl(ind.get('hist_vol_20', 20), 100)

    obv_t = ind.get('obv_trend', '')
    obv_dir = 1.0 if obv_t == 'rising' else (-1.0 if obv_t == 'falling' else 0.0)

    roc5  = cl(ind.get('roc_5', 0), 20)
    roc10 = cl(ind.get('roc_10', 0), 30)
    roc20 = cl(ind.get('roc_20', 0), 40)

    candle = 1.0 if ind.get('candle_signal') == 'bullish' else -1.0

    adx_norm = cl(ind.get('adx_14', 20), 50) - 0.4
    adx_dir  = adx_norm

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
        'adx_norm': adx_norm, 'adx_dir': adx_dir,
        'neg_vs20': -vs20, 'neg_bb': -bb_pos,
        'neg_rsi': -rsi_norm, 'neg_vs50': -vs50,
        'sin_month': sin_month, 'cos_month': cos_month,
    }


# ── Training (uses price_history) ────────────────────────────────────────────

def train_model(model_name: str) -> dict:
    from supabase import create_client
    import xgboost as xgb

    if model_name not in MODEL_FEATURE_NAMES:
        raise ValueError(f'Unknown model: {model_name}')

    feature_names = MODEL_FEATURE_NAMES[model_name]
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Fetch all price history
    all_rows = []
    offset = 0
    while True:
        resp = sb.table('price_history').select(
            'asset_id,trade_date,open,high,low,close,volume,adj_close'
        ).order('trade_date').range(offset, offset + 999).execute()
        rows = resp.data or []
        all_rows.extend(rows)
        if len(rows) < 1000:
            break
        offset += 1000

    if not all_rows:
        raise ValueError('No price history data found')

    # Group by asset_id, sorted by date
    asset_rows: dict = defaultdict(list)
    for row in all_rows:
        asset_rows[row['asset_id']].append(row)
    for aid in asset_rows:
        asset_rows[aid].sort(key=lambda r: r['trade_date'])

    # Compute features for every (asset, date) with enough history
    feat_cache: dict = {}       # (asset_id, trade_date) -> feature dict
    price_cache: dict = {}      # (asset_id, trade_date) -> close price

    for aid, rows in asset_rows.items():
        date_to_close = {
            r['trade_date']: float(r.get('adj_close') or r.get('close') or 0)
            for r in rows
        }
        for i in range(21, len(rows)):
            date_str = rows[i]['trade_date']
            close_p  = date_to_close.get(date_str, 0)
            if close_p <= 0:
                continue
            feats = compute_features_from_ohlcv(rows[:i + 1], date_str)
            if feats is None:
                continue
            feat_cache[(aid, date_str)]  = feats
            price_cache[(aid, date_str)] = close_p

    bucket_results = {}
    for bucket in HORIZON_BUCKETS:
        min_move = MIN_MOVE_PCT[bucket]
        X_rows, y_rows = [], []

        for aid, rows in asset_rows.items():
            date_to_close = {
                r['trade_date']: float(r.get('adj_close') or r.get('close') or 0)
                for r in rows
            }
            dates = [r['trade_date'] for r in rows]

            for i in range(21, len(rows)):
                date_str = dates[i]
                feats = feat_cache.get((aid, date_str))
                if feats is None:
                    continue
                close_p = price_cache[(aid, date_str)]

                # Forward price
                try:
                    target_dt = datetime.strptime(date_str, '%Y-%m-%d') + timedelta(days=int(bucket * 1.45))
                except Exception:
                    continue

                future_price = None
                for delta in range(-3, 8):
                    check = (target_dt + timedelta(days=delta)).strftime('%Y-%m-%d')
                    fp = date_to_close.get(check, 0)
                    if fp > 0:
                        future_price = fp
                        break

                if future_price is None:
                    continue

                pct = (future_price - close_p) / close_p * 100
                X_rows.append([feats.get(f, 0.0) for f in feature_names])
                y_rows.append(1 if pct >= min_move else 0)

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

        train_acc = float((model.predict(X) == y).mean())
        model_b64 = base64.b64encode(pickle.dumps(model)).decode()

        sb.table('xgb_models').upsert({
            'model_name':    model_name,
            'horizon_bucket': bucket,
            'model_data':    model_b64,
            'feature_names': feature_names,
            'train_accuracy': train_acc,
            'train_samples': len(X_rows),
        }, on_conflict='model_name,horizon_bucket').execute()

        bucket_results[bucket] = {
            'samples':  len(X_rows),
            'accuracy': round(train_acc, 4),
            'pos_rate': round(pos_rate, 4),
        }

    return {'model_name': model_name, 'buckets': bucket_results}


# ── Prediction (uses indicators table) ──────────────────────────────────────

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
    today = datetime.now().strftime('%Y-%m-%d')
    indicators = sb.table('indicators').select(cols).eq('computed_date', today).execute().data or []

    if not indicators:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
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
            model = model_info['model']
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


# ── Flask app ────────────────────────────────────────────────────────────────

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
    body = request.get_json(silent=True) or {}
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
