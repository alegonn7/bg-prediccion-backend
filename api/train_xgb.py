from http.server import BaseHTTPRequestHandler
import json
import os
import pickle
import base64
import numpy as np
from datetime import datetime, timedelta
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


def train_model(model_name: str) -> dict:
    from supabase import create_client
    import xgboost as xgb

    if model_name not in MODEL_FEATURE_NAMES:
        raise ValueError(f'Unknown model: {model_name}')

    feature_names = MODEL_FEATURE_NAMES[model_name]
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    cutoff = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    indicators = []
    offset = 0
    cols = (
        'asset_id,computed_date,price_close,'
        'price_vs_sma20,price_vs_sma50,price_vs_sma200,'
        'rsi_14,macd_histogram,macd_signal,bb_pct_b,bb_squeeze,'
        'atr_pct,hist_vol_20,obv_trend,roc_5,roc_10,roc_20,'
        'candle_signal,adx_14'
    )
    while True:
        resp = sb.table('indicators').select(cols).gte(
            'computed_date', cutoff
        ).order('computed_date').range(offset, offset + 999).execute()
        rows = resp.data or []
        indicators.extend(rows)
        if len(rows) < 1000:
            break
        offset += 1000

    if not indicators:
        raise ValueError('No indicator data found')

    # Build asset_id -> date -> close_price lookup
    price_map: dict = {}
    for row in indicators:
        aid = row['asset_id']
        dt  = row['computed_date']
        pc  = float(row.get('price_close') or 0)
        if pc > 0:
            if aid not in price_map:
                price_map[aid] = {}
            price_map[aid][dt] = pc

    bucket_results = {}
    for bucket in HORIZON_BUCKETS:
        min_move = MIN_MOVE_PCT[bucket]
        X_rows, y_rows = [], []

        for ind_row in indicators:
            aid = ind_row['asset_id']
            dt  = ind_row['computed_date']
            close_price = float(ind_row.get('price_close') or 0)
            if close_price <= 0:
                continue

            feats = extract_features(ind_row)
            x = [feats[f] for f in feature_names]

            # Approximate forward date (bucket trading days ≈ bucket*1.45 calendar days)
            try:
                dt_obj = datetime.strptime(dt, '%Y-%m-%d')
                target_dt = dt_obj + timedelta(days=int(bucket * 1.45))
            except Exception:
                continue

            # Find closest available price near target_dt
            prices = price_map.get(aid, {})
            future_price = None
            for delta in range(-3, 8):
                check = (target_dt + timedelta(days=delta)).strftime('%Y-%m-%d')
                if check in prices:
                    future_price = prices[check]
                    break

            if future_price is None:
                continue

            pct_change = (future_price - close_price) / close_price * 100
            label = 1 if pct_change >= min_move else 0

            X_rows.append(x)
            y_rows.append(label)

        if len(X_rows) < 50:
            bucket_results[bucket] = {'skipped': True, 'samples': len(X_rows)}
            continue

        X = np.array(X_rows, dtype=np.float32)
        y = np.array(y_rows, dtype=np.float32)

        pos_rate = float(y.mean())
        scale_pos_weight = (1 - pos_rate) / pos_rate if pos_rate > 0.01 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=150,
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
            'model_name': model_name,
            'horizon_bucket': bucket,
            'model_data': model_b64,
            'feature_names': feature_names,
            'train_accuracy': train_acc,
            'train_samples': len(X_rows),
        }, on_conflict='model_name,horizon_bucket').execute()

        bucket_results[bucket] = {
            'samples': len(X_rows),
            'accuracy': round(train_acc, 4),
            'pos_rate': round(pos_rate, 4),
        }

    return {'model_name': model_name, 'buckets': bucket_results}


class handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'content-type, authorization, x-internal-secret')

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        # Verify internal secret
        secret = self.headers.get('x-internal-secret', '')
        if INTERNAL_SECRET and secret != INTERNAL_SECRET:
            self.send_response(403)
            self.send_header('Content-Type', 'application/json')
            self._cors()
            self.end_headers()
            self.wfile.write(b'{"ok":false,"error":"forbidden"}')
            return

        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
        model_name = body.get('model_name', 'tendencia')

        try:
            result = train_model(model_name)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({'ok': True, **result}).encode())
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self._cors()
            self.end_headers()
            self.wfile.write(json.dumps({'ok': False, 'error': str(e)}).encode())
