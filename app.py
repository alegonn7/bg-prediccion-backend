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



def cl(v, scale):
    return max(-1.0, min(1.0, float(v or 0) / scale)) if scale else 0.0


def cl3(v, lo=-3.0, hi=3.0):
    return max(lo, min(hi, float(v or 0)))


def magnitude_weight(y_real_pct, floor=0.1):
    """Sample weight by |actual return| — LightGBM pays more for missing big moves instead
    of being scored equally on days the asset barely budged. Applied only to LGBM .fit()
    calls (not Ridge/LogisticRegression), multiplied on top of the existing recency weight.
    floor avoids zero-weighting near-flat samples entirely."""
    return np.maximum(np.abs(np.asarray(y_real_pct, dtype=float)), floor)


def lgbm_trial_score(mae, dir_acc, dir_acc_floor=0.5, penalty_scale=2.0):
    """Optuna objective score (lower=better). Inflates MAE when validation-fold directional
    accuracy is at/below a coin flip, so hyperparam search can't win purely by shrinking
    predictions toward zero (great MAE, useless direction)."""
    if dir_acc is None:
        return mae
    shortfall = max(0.0, dir_acc_floor - dir_acc)
    return mae * (1.0 + penalty_scale * shortfall)


def directional_accuracy(y_true, y_pred):
    """Fraction of matching signs — None if there's nothing to compare."""
    if len(y_true) == 0:
        return None
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def capture_ratio_segments(y_true, y_pred, top_frac=0.2, min_abs=0.05):
    """'% de movimiento real capturado' (predicho / real, sólo cuando el signo coincide),
    medido por separado en el top `top_frac` de movimientos reales más grandes vs. el resto.
    Mejorar el segmento grande típicamente empeora el MAE promedio en el segmento chico/tranquilo
    — es el trade-off esperado, no una regresión; por eso se trackea separado en vez de un único
    número agregado. Devuelve (pct_top, pct_rest, n_top, n_rest); None cuando no hay muestra
    suficiente en ese segmento."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = (np.abs(y_true) >= min_abs) & (np.sign(y_true) == np.sign(y_pred))
    if mask.sum() < 5:
        return None, None, int(mask.sum()), 0
    yt, yp = y_true[mask], y_pred[mask]
    order = np.argsort(-np.abs(yt))
    n_top = max(1, int(round(len(yt) * top_frac)))
    top_idx, rest_idx = order[:n_top], order[n_top:]
    ratio = np.clip(yp / yt, -5, 5)  # guard against blowups when yt is near the min_abs floor
    pct_top = float(np.mean(ratio[top_idx]) * 100) if len(top_idx) else None
    pct_rest = float(np.mean(ratio[rest_idx]) * 100) if len(rest_idx) else None
    return pct_top, pct_rest, len(top_idx), len(rest_idx)


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


def _check_secret() -> bool:
    if not INTERNAL_SECRET:
        return False
    secret = request.headers.get('x-internal-secret', '')
    return secret == INTERNAL_SECRET


@app.after_request
def _cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'content-type, authorization, x-internal-secret'
    return response


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
    # New features added 2026-07-06 — returned by get_intraday_training_data v2
    'macd_hist',          # MACD histogram (numeric)
    'orb_breakout_num',   # orb_breakout text → +1 up / -1 down / 0 none
    'bb_squeeze_num',     # bb_squeeze bool → 1/0
    'macd_cross_num',     # macd_cross text → +1 bullish / -1 bearish / 0 none
]

lr_training_jobs: dict = {}

# Etapa 29.1 — con la RPC paginada, este límite ahora se cumple de verdad (antes el techo real
# era 1000, el cap de PostgREST). _INTRADAY_PAGE_SIZE no puede superar 1000: es el cap del
# transporte, pedir más no trae más, sólo lo esconde.
#
# Por qué 30.000 y no las ~72.000 filas elegibles que hay: Render corre en free tier (512 MB) y
# ya tuvo un crash-loop por memoria (ver STATUS.md, 28/07). 30.000 filas ≈ 3 ruedas ≈ 10.000
# eventos únicos, contra los 318 con los que entrenaba ridge:60 antes de este arreglo.
#
# Optimización pendiente, anotada acá porque es donde se nota: las filas vienen triplicadas. Para
# un mismo (evento, horizonte) hay una fila por model_name (lgbm/ridge/reversion) con features
# IDÉNTICAS. El modelo signed y LGBM entrenan contra actual_signed_pct, que es del evento y no del
# modelo, así que 2 de cada 3 filas no aportan información a esos dos — sólo el clasificador de
# dirección usa direction_correct, que sí es por modelo. Deduplicar del lado SQL triplicaría la
# ventana efectiva sin costar memoria.
_INTRADAY_FETCH_LIMIT = 30000
_INTRADAY_PAGE_SIZE = 1000


def _log_training_event(sb, event: str, summary: str, samples: int | None = None) -> None:
    """Etapa 27.6 — deja rastro en model_changelog de lo que antes fallaba en silencio.

    Motivo (auditoría del 11/08/2026): el entrenamiento diario del 10/08 escribió
    lgbm_cluster_models_daily a las 21:17 y NUNCA llegó a upsert_daily_signed_params, que es la
    línea siguiente del mismo bucle. Los parámetros Ridge diarios quedaron congelados tres días y
    no había forma de notarlo: el dashboard muestra 'última actualización', que con un fallo a
    mitad de camino sigue mostrando una fecha vieja sin decir que hubo un intento fallido.

    Nunca lanza: si el logging falla, el entrenamiento tiene que seguir igual.
    """
    try:
        sb.table('model_changelog').insert({
            'model_name': '__training__',
            'change_type': 'lr_params',
            'trigger': event,
            'new_samples': samples,
            'summary': summary[:2000],
        }).execute()
    except Exception as e:
        print(f'[changelog] no se pudo registrar {event}: {e}', flush=True)

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
    # arg_ars: TODO lo que cotiza en pesos en BYMA (33 CEDEARs + 20 acciones argentinas).
    # Van juntos a propósito, aunque un CEDEAR siga a una empresa extranjera y una acción local a
    # una argentina: lo que comparten es estar denominados en pesos, o sea que arrastran el
    # componente cambiario y la liquidez de BYMA. Eso es exactamente lo que el modelo global —
    # entrenado casi todo con datos de EEUU — no puede capturar.
    # Motivo concreto de este cluster (ver STATUS.md, sesión del 08/08/2026): las 3 features de la
    # Etapa 23 (underlying_pred_norm, underlying_conf_norm, ccl_momentum_norm) quedaron con
    # coeficiente EXACTAMENTE 0 en el modelo global, porque sólo valen distinto de cero en el 0,8%
    # de las muestras (649 filas en pesos contra ~79.000 totales) y la regularización las anula.
    # En un modelo entrenado sólo con instrumentos en pesos tienen valor real en el 100% de las
    # muestras, así que recién ahí el modelo puede aprender cuánto pesa la predicción en dólares.
    # OJO: separarlos en dos clusters (CEDEAR vs acción local) parte la muestra al medio y ninguno
    # llegaría al mínimo de 50 — reevaluar cuando haya bastante más histórico cerrado.
    'MU.BA': 'arg_ars', 'MSFT.BA': 'arg_ars', 'MELI.BA': 'arg_ars', 'SNDK.BA': 'arg_ars',
    'NVDA.BA': 'arg_ars', 'SPY.BA': 'arg_ars', 'GOOGL.BA': 'arg_ars', 'NU.BA': 'arg_ars',
    'QQQ.BA': 'arg_ars', 'IBM.BA': 'arg_ars', 'AMZN.BA': 'arg_ars', 'ORCL.BA': 'arg_ars',
    'AMD.BA': 'arg_ars', 'PLTR.BA': 'arg_ars', 'TSLA.BA': 'arg_ars', 'MSTR.BA': 'arg_ars',
    'AAPL.BA': 'arg_ars', 'NBIS.BA': 'arg_ars', 'INTC.BA': 'arg_ars', 'GLD.BA': 'arg_ars',
    'KO.BA': 'arg_ars', 'BRKB.BA': 'arg_ars', 'ASTS.BA': 'arg_ars', 'SATL.BA': 'arg_ars',
    'EWZ.BA': 'arg_ars', 'MCD.BA': 'arg_ars', 'V.BA': 'arg_ars', 'JPM.BA': 'arg_ars',
    'UBER.BA': 'arg_ars', 'GLOB.BA': 'arg_ars', 'NFLX.BA': 'arg_ars', 'META.BA': 'arg_ars',
    'VIST.BA': 'arg_ars',
    'YPFD.BA': 'arg_ars', 'GGAL.BA': 'arg_ars', 'PAMP.BA': 'arg_ars', 'BMA.BA': 'arg_ars',
    'BBAR.BA': 'arg_ars', 'TGSU2.BA': 'arg_ars', 'SUPV.BA': 'arg_ars', 'CEPU.BA': 'arg_ars',
    'LOMA.BA': 'arg_ars', 'TECO2.BA': 'arg_ars', 'TXAR.BA': 'arg_ars', 'TRAN.BA': 'arg_ars',
    'BYMA.BA': 'arg_ars', 'EDN.BA': 'arg_ars', 'METR.BA': 'arg_ars', 'VALO.BA': 'arg_ars',
    'ECOG.BA': 'arg_ars', 'TGNO4.BA': 'arg_ars', 'CRES.BA': 'arg_ars', 'COME.BA': 'arg_ars',
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
        # Etapa 29.1 — Paginación por keyset. Historia de por qué es así y no de otra forma:
        #
        #   1. La versión original paginaba con .range() de PostgREST. get_intraday_training_data()
        #      es plpgsql (no inlineable), así que PostgREST reejecutaba la función ENTERA por
        #      página en vez de empujar OFFSET/LIMIT adentro: hasta 16 corridas del join+sort por
        #      job, y con la tabla en +140k filas eso superaba el statement_timeout.
        #   2. El arreglo fue pasar p_limit a la función y llamarla una sola vez. Eso resolvió el
        #      timeout y creó un problema peor y silencioso: PostgREST cap-ea toda respuesta a
        #      1000 filas pase lo que pida el cliente (el mismo tope que la Etapa 16 ya había
        #      encontrado del lado del dashboard). Resultado real medido el 10/08: ridge:60 con
        #      318 muestras, ridge:120 con 174, ridge:240 sin entrenar — y 31 features.
        #   3. Un OFFSET adentro de la función tiene el mismo defecto de fondo que (1): la página
        #      N reescanea las N-1 anteriores. Por eso keyset: cada página hace un seek por el
        #      índice mpid_created y cuesta O(página).
        #
        # El desempate por id NO es cosmético: created_at no es único (las predicciones se insertan
        # en lote y miles comparten timestamp), así que un cursor sólo por created_at saltearía y
        # duplicaría filas en cada corte de página.
        all_rows = []
        before_ts, before_id = None, None
        pages = 0
        while len(all_rows) < _INTRADAY_FETCH_LIMIT:
            page = None
            for attempt in range(3):
                try:
                    resp = sb.rpc('get_intraday_training_data_page', {
                        'p_limit': _INTRADAY_PAGE_SIZE,
                        'p_before_created_at': before_ts,
                        'p_before_id': before_id,
                    }).execute()
                    page = resp.data or []
                    break
                except Exception as e:
                    if attempt == 2:
                        raise
                    print(f'[lr_train] page {pages} attempt {attempt + 1} failed: {e} — retrying', flush=True)
                    time.sleep(5 * (attempt + 1))

            if not page:
                break
            all_rows.extend(page)
            pages += 1
            # cursor = última fila de la página (la función ordena created_at DESC, id DESC)
            before_ts, before_id = page[-1]['created_at'], page[-1]['id']
            if len(page) < _INTRADAY_PAGE_SIZE:
                break   # se acabaron los datos, no el límite

        print(f'[lr_train] fetched {len(all_rows)} rows in {pages} pages', flush=True)

        # Etapa 27.6.3 — que un fetch corto no vuelva a pasar desapercibido. Con la paginación
        # arreglada, quedarse exactamente en un múltiplo redondo del tamaño de página es la firma
        # de un tope escondido, no de que se hayan acabado los datos.
        if pages > 0 and len(all_rows) == pages * _INTRADAY_PAGE_SIZE and len(all_rows) < _INTRADAY_FETCH_LIMIT:
            _log_training_event(
                sb, 'lr_train_intraday_truncated',
                f'El fetch intradiario cortó en {len(all_rows)} filas ({pages} páginas completas) '
                f'sin llegar al límite de {_INTRADAY_FETCH_LIMIT} y sin devolver una página '
                f'parcial. Sospechar un tope nuevo del lado del transporte.',
                samples=len(all_rows))

        job['total_samples'] = len(all_rows)
        if not all_rows:
            job['status'] = 'done'
            job['models_trained'] = 0
            return

        now_utc = datetime.now(timezone.utc)
        all_rows.sort(key=lambda r: _parse_ts(r.get('created_at')))

        # Etapa 29.2 — El holdout de 30 días nunca existió, y el código no se enteraba.
        #
        # `limpieza-dominical-entrenamiento` (cron 51) borra model_predictions_intraday a los 7
        # días, así que NUNCA hay filas de más de 30 días: holdout_mask daba todo True, tv_mask
        # todo False, y el bloque de más abajo caía al fallback de "usar todo". O sea el
        # entrenamiento intradiario venía entrenando y validando sobre el mismo período, sin
        # holdout real — funcionando por accidente del fallback, no por diseño.
        #
        # Ahora el corte es proporcional cuando no hay 30 días de historia: se aparta el tramo más
        # reciente (20% del período cubierto) como holdout. Con la ventana real de ~3 ruedas eso
        # son las últimas horas: poco, pero es un holdout de verdad — y se agranda solo si sube la
        # retención del cron 51.
        _oldest = _parse_ts(all_rows[0].get('created_at'))
        _span_days = max((now_utc - _oldest).total_seconds() / 86400, 0.0)
        if _span_days >= 37.5:                       # historia de sobra: el corte fijo sirve
            holdout_cutoff = now_utc - timedelta(days=30)
        else:                                        # ventana corta: corte proporcional
            holdout_cutoff = now_utc - timedelta(days=_span_days * 0.20)
        print(f'[lr_train] ventana={_span_days:.1f}d holdout_desde={holdout_cutoff.isoformat()}', flush=True)

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
            lgbm_error_p50 = lgbm_error_p75 = lgbm_error_p90 = None
            capture_pct_top20 = capture_pct_rest = None
            capture_n_top20 = capture_n_rest = None
            if horizon_minutes in lgbm_horizon_cache:
                cached = lgbm_horizon_cache[horizon_minutes]
                lgbm_model_b64 = cached['model_b64']
                lgbm_val_mae = cached['val_mae']
                lgbm_importance = cached['importance']
                beta_spy = cached['beta_spy']
                lgbm_error_p50 = cached.get('error_p50')
                lgbm_error_p75 = cached.get('error_p75')
                lgbm_error_p90 = cached.get('error_p90')
                capture_pct_top20 = cached.get('capture_pct_top20')
                capture_pct_rest = cached.get('capture_pct_rest')
                capture_n_top20 = cached.get('capture_n_top20')
                capture_n_rest = cached.get('capture_n_rest')
            elif Xs is not None and len(Xs) >= 30:
                import optuna
                optuna.logging.set_verbosity(optuna.logging.WARNING)

                atr_idx = LR_FEATURE_NAMES.index('atr_pct')
                atr_train_raw = np.clip(X_train[train_signed_mask][:, atr_idx], 0.1, 10.0)
                ys_norm = ys / atr_train_raw
                # LGBM-only: pay more for missing large moves, on top of the recency weight.
                ws_lgbm = ws * magnitude_weight(ys)

                val_sm = ~np.isnan(y_signed_val) if len(X_val_s) > 0 else np.zeros(0, dtype=bool)
                has_val = val_sm.sum() >= 5
                if has_val:
                    atr_val_raw = np.clip(X_val[:, atr_idx][val_sm], 0.1, 10.0)
                    y_val_norm = y_signed_val[val_sm] / atr_val_raw

                def _lgbm_objective(trial):
                    params = dict(
                        num_leaves=trial.suggest_int('num_leaves', 15, 90),
                        learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                        min_child_samples=trial.suggest_int('min_child_samples', 10, 60),
                        max_depth=trial.suggest_int('max_depth', 3, 7),
                        min_split_gain=trial.suggest_float('min_split_gain', 0.0, 0.5),
                        subsample=trial.suggest_float('subsample', 0.6, 1.0),
                        colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        reg_alpha=trial.suggest_float('reg_alpha', 0.0, 1.0),
                        reg_lambda=trial.suggest_float('reg_lambda', 0.0, 5.0),
                        n_estimators=300, random_state=42, verbose=-1,
                        objective='regression_l1',
                    )
                    m = lgb.LGBMRegressor(**params)
                    m.fit(Xs, ys_norm, sample_weight=ws_lgbm)
                    if has_val:
                        preds_d = m.predict(X_val_s[val_sm]) * atr_val_raw
                        mae = float(np.mean(np.abs(y_signed_val[val_sm] - preds_d)))
                        dir_acc = directional_accuracy(y_signed_val[val_sm], preds_d)
                    else:
                        preds_tr = m.predict(Xs) * atr_train_raw
                        mae = float(np.mean(np.abs(ys_norm - m.predict(Xs))))
                        dir_acc = directional_accuracy(ys, preds_tr)
                    return lgbm_trial_score(mae, dir_acc)

                study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                n_trials = 50 if len(Xs) >= 30 else 25
                study.optimize(_lgbm_objective, n_trials=n_trials, show_progress_bar=False)
                best_p = study.best_params
                print(f'[optuna] H={horizon_minutes} best={best_p} score={study.best_value:.4f}', flush=True)

                # Train final model with best params + early stopping
                eval_set_norm = [(X_val_s[val_sm], y_val_norm)] if has_val else None
                callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_norm else None
                lgb_reg = lgb.LGBMRegressor(
                    **best_p, n_estimators=500, random_state=42, verbose=-1,
                    objective='regression_l1',
                )
                lgb_reg.fit(Xs, ys_norm, sample_weight=ws_lgbm, eval_set=eval_set_norm, callbacks=callbacks)
                if val_sm.sum() > 0:
                    atr_val_raw = np.clip(X_val[:, atr_idx][val_sm], 0.1, 10.0)
                    preds_denorm = lgb_reg.predict(X_val_s[val_sm]) * atr_val_raw
                    val_residuals = np.abs(y_signed_val[val_sm] - preds_denorm)
                    lgbm_val_mae = float(np.mean(val_residuals))
                    lgbm_error_p50 = float(np.percentile(val_residuals, 50))
                    lgbm_error_p75 = float(np.percentile(val_residuals, 75))
                    lgbm_error_p90 = float(np.percentile(val_residuals, 90))
                    capture_pct_top20, capture_pct_rest, capture_n_top20, capture_n_rest = \
                        capture_ratio_segments(y_signed_val[val_sm], preds_denorm)
                    print(f'[capture] H={horizon_minutes} top20%={capture_pct_top20} (n={capture_n_top20}) '
                          f'resto={capture_pct_rest} (n={capture_n_rest})', flush=True)
                lgbm_model_b64 = base64.b64encode(pickle.dumps(lgb_reg)).decode('utf-8')
                lgbm_importance = dict(zip(LR_FEATURE_NAMES, lgb_reg.feature_importances_.tolist()))
                lgbm_horizon_cache[horizon_minutes] = {
                    'model_b64': lgbm_model_b64,
                    'val_mae': lgbm_val_mae,
                    'importance': lgbm_importance,
                    'beta_spy': beta_spy,
                    'error_p50': lgbm_error_p50,
                    'error_p75': lgbm_error_p75,
                    'error_p90': lgbm_error_p90,
                    'capture_pct_top20': capture_pct_top20,
                    'capture_pct_rest': capture_pct_rest,
                    'capture_n_top20': capture_n_top20,
                    'capture_n_rest': capture_n_rest,
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
                'error_p50': lgbm_error_p50,
                'error_p75': lgbm_error_p75,
                'error_p90': lgbm_error_p90,
                'capture_pct_top20': capture_pct_top20,
                'capture_pct_rest': capture_pct_rest,
                'capture_n_top20': capture_n_top20,
                'capture_n_rest': capture_n_rest,
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

        # Bootstrap 'lgbm'/'ridge' under their canonical D4 vote names (backlog fix, post
        # Etapa 12). model_learned_params_intraday is keyed by (model_name, horizon_minutes)
        # from the legacy 13-strategy schema — 'lgbm'/'ridge' never existed as historical
        # model_names before crear-prediccion-intraday started asking for them (Etapa 4), so
        # they can never appear via the grouping above: a catch-22 where the vote can't fire
        # without a trained row, and no row gets trained without the vote having fired first.
        # The X/y pairs are identical across model_names for the same horizon (indicators +
        # actual outcome, independent of which named strategy logged the guess — see the LGBM
        # dedup cache above), so cloning an existing horizon's trained artifacts under the
        # canonical names isn't fabricating a different model, just exposing the one already
        # trained under the name the new roster looks up. Self-healing: once the votes fire for
        # real, future runs group genuine 'lgbm'/'ridge' rows and this stops being needed.
        upserted_keys = {(u['model_name'], u['horizon_minutes']) for u in upserts}
        by_horizon: dict = defaultdict(list)
        for u in upserts:
            by_horizon[u['horizon_minutes']].append(u)
        for horizon_minutes, group_upserts in by_horizon.items():
            for canonical, needs_field in (('lgbm', 'lgbm_model'), ('ridge', 'signed_coefficients')):
                if (canonical, horizon_minutes) in upserted_keys:
                    continue
                candidates = [u for u in group_upserts if u.get(needs_field)]
                if not candidates:
                    continue
                template = max(candidates, key=lambda u: u['train_samples'])
                clone = dict(template)
                clone['model_name'] = canonical
                upserts.append(clone)
                print(f'[lr_train] bootstrap {canonical}:{horizon_minutes} cloned from '
                      f'{template["model_name"]} (n={template["train_samples"]})', flush=True)

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
            ws_s_lgbm = ws_s * magnitude_weight(ys_s)

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
            lgb_s.fit(Xs_s, ys_norm_s, sample_weight=ws_s_lgbm, eval_set=eval_set_s, callbacks=cbs)

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

            # Store single canonical row per (horizon, session) — avoids 13× storage bloat
            session_upserts.append({
                'model_name': '__session__',
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
            ws_t_lgbm = ws_t * magnitude_weight(ys_t)

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
            lgb_t.fit(Xs_t, ys_tn, sample_weight=ws_t_lgbm, eval_set=eval_t, callbacks=cbs_t)

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
        ridge_cluster_upserts = []  # Etapa 29.6.3
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
            ws_c_lgbm = ws_c * magnitude_weight(ys_c)

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
            lgb_c.fit(Xs_c, ys_cn, sample_weight=ws_c_lgbm, eval_set=eval_c, callbacks=cbs_c)

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

            # Etapa 29.6.3 — Ridge propio por cluster, empezando por arg_ars. Reusa Xs_c/ys_c/sc_c
            # ya calculados arriba para el LGBM de este mismo cluster (mismo holdout, mismo beta,
            # mismo escalado) — nada se recalcula. A diferencia de LGBM (ys_cn, normalizado por
            # ATR), Ridge usa ys_c sin normalizar, igual que el Ridge global unas líneas más arriba
            # en esta misma función (ver 'Ridge signed — kept for backward compat').
            #
            # Por qué hace falta: hasta ahora sólo LGBM se entrenaba por cluster (Etapa "Hallazgo
            # grande", 08/08) — Ridge seguía siendo un único modelo global, entrenado casi todo con
            # datos de EEUU. arg_ars mide 1.484-1.992 muestras/horizonte (verificado el 14/08,
            # sobre el mínimo de 50), así que no nace con coeficientes en cero como sí habría
            # pasado si se hubiera intentado esto antes de que el universo argentino acumulara
            # historia — mismo mecanismo que dejó en 0 la feature híbrida del lado diario.
            reg_ridge_c = Ridge(alpha=1.0)
            reg_ridge_c.fit(Xs_c, ys_c, sample_weight=ws_c)
            ridge_c_val_mae = None
            if vm_c.sum() > 0:
                ridge_c_val_mae = float(np.mean(np.abs(y_v[vm_c] - reg_ridge_c.predict(X_v_c[vm_c]))))
            mag_valid_c = np.abs(y_tv[~np.isnan(y_tv)])
            ridge_cluster_upserts.append({
                'cluster_name': cluster_name,
                'horizon_minutes': horizon_minutes,
                'feature_names': LR_FEATURE_NAMES,
                'signed_coefficients': reg_ridge_c.coef_.tolist(),
                'signed_bias': float(reg_ridge_c.intercept_),
                'feature_means': sc_c.mean_.tolist(),
                'feature_stds': sc_c.scale_.tolist(),
                'avg_actual_mag': float(np.mean(mag_valid_c)) if len(mag_valid_c) > 0 else None,
                'median_actual_mag': float(np.median(mag_valid_c)) if len(mag_valid_c) > 0 else None,
                'val_mae_ridge': ridge_c_val_mae,
                'train_samples': int(tm.sum()),
                'last_updated': now_utc.isoformat(),
            })
            print(f'[lr_train:ridge_cluster] {cluster_name}:{horizon_minutes} n={tm.sum()} val_mae={ridge_c_val_mae}', flush=True)

        for cu in cluster_upserts:
            sb.table('lgbm_cluster_models_intraday').upsert(
                cu, on_conflict='cluster_name,horizon_minutes'
            ).execute()
        print(f'[lr_train] cluster models: {len(cluster_upserts)} trained', flush=True)

        for rcu in ridge_cluster_upserts:
            sb.table('ridge_cluster_models_intraday').upsert(
                rcu, on_conflict='cluster_name,horizon_minutes'
            ).execute()
        print(f'[lr_train] ridge cluster models: {len(ridge_cluster_upserts)} trained', flush=True)
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

    print('[hist] Fetching pre-live price_history (paginated)...', flush=True)
    # Only fetch data BEFORE live predictions started (2025-01-27).
    # This gives the 2022-2024 data loaded separately — clean non-overlapping training window.
    # With MAX_HIST_DAYS=150 per ticker this stays well under Render's 512 MB limit.
    LIVE_CUTOFF_DATE = '2025-01-27'
    rows = []
    PAGE = 1000
    offset = 0
    while True:
        resp = sb.from_('price_history').select(
            'asset_id, trade_date, open, high, low, close, volume'
        ).lt('trade_date', LIVE_CUTOFF_DATE).order('trade_date').range(offset, offset + PAGE - 1).execute()
        chunk = resp.data or []
        rows.extend(chunk)
        if len(chunk) < PAGE:
            break
        offset += PAGE
    print(f'[hist] fetched {len(rows)} pre-live rows from price_history', flush=True)
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

            # Only generate samples from the last 150 trading days to avoid OOM.
            # SMA200 still computed on full history above (correct lookback).
            MAX_HIST_DAYS = 150
            start_i = max(MIN_LOOKBACK, n - MAX_HIST_DAYS)

            for i in range(start_i, n):
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
                        'atr_pct':          float(df['atr_pct'].iloc[i]) if not pd.isna(df['atr_pct'].iloc[i]) else 1.0,
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
    # Etapa 23: par USD↔ARS (CEDEARs y acciones argentinas locales) — 0 para todo lo demás (3)
    'underlying_pred_norm', 'underlying_conf_norm', 'ccl_momentum_norm',
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
    # Etapa 23: predicción USD del subyacente (mismo horizonte) + momentum del CCL. Ausentes para
    # cualquier ticker sin par (todo lo que no sea cedear_arg/accion_arg_local) → quedan en 0, sin
    # efecto en esos modelos. Presentes sólo cuando quien arma `row` los agrega explícitamente
    # (crear-prediccion vía indicators, o el enrichment de get_daily_training_data en Python).
    underlying_pred = row.get('underlying_pred_pct')
    underlying_conf = row.get('underlying_pred_conf')
    ccl_mom         = row.get('ccl_momentum_5d')
    underlying_pred_norm = cl3(float(underlying_pred) / 5) if underlying_pred is not None else 0.0
    underlying_conf_norm = cl3((float(underlying_conf) - 0.5) * 2, -1.0, 1.0) if underlying_conf is not None else 0.0
    ccl_momentum_norm    = cl3(float(ccl_mom) / 3) if ccl_mom is not None else 0.0
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
        # Etapa 23: par USD↔ARS (3)
        underlying_pred_norm, underlying_conf_norm, ccl_momentum_norm,
    ]


daily_training_jobs: dict = {}


def _enrich_underlying_pair_features(sb, rows: list) -> None:
    """Etapa 23: para filas reales de tickers con par USD↔ARS (cedear_arg/accion_arg_local),
    agrega 'underlying_pred_pct'/'underlying_pred_conf'/'ccl_momentum_5d' con el valor real que
    existía en ese momento (as-of, no lookahead: la predicción USD más reciente con
    created_at <= la de esta fila real, mismo horizon_bucket). Muta `rows` in place. Tickers sin
    par, o fechas sin muestra as-of disponible, quedan sin estas keys — _extract_daily_features ya
    los trata como 0 vía row.get(..., None)."""
    a_resp = sb.from_('assets').select('id, ticker, underlying_ticker').not_.is_('underlying_ticker', 'null').execute()
    pair_assets = a_resp.data or []
    if not pair_assets:
        return
    underlying_ticker_of = {a['ticker']: a['underlying_ticker'] for a in pair_assets}
    tickers_with_pair = set(underlying_ticker_of.keys())
    relevant_rows = [r for r in rows if r.get('ticker') in tickers_with_pair]
    if not relevant_rows:
        return

    underlying_tickers = set(underlying_ticker_of.values())
    u_resp = sb.from_('assets').select('id, ticker').in_('ticker', list(underlying_tickers)).execute()
    underlying_id_of = {a['ticker']: a['id'] for a in (u_resp.data or [])}

    cp_resp = sb.from_('consensus_predictions') \
        .select('asset_id, horizon_days, created_at, final_pct_predicted, confidence') \
        .in_('asset_id', list(underlying_id_of.values())) \
        .order('created_at').execute()
    cp_rows = cp_resp.data or []
    # (underlying_ticker, horizon_bucket) -> lista ordenada de (created_at, pct, conf)
    id_to_ticker = {v: k for k, v in underlying_id_of.items()}
    cp_series: dict = {}
    for r in cp_rows:
        t = id_to_ticker.get(r['asset_id'])
        if not t:
            continue
        key = (t, int(r['horizon_days']))
        cp_series.setdefault(key, []).append((r['created_at'], r['final_pct_predicted'], r['confidence']))

    ccl_resp = sb.from_('dolar_ccl_history').select('fecha, venta').order('fecha').execute()
    ccl_rows = [(r['fecha'], float(r['venta'])) for r in (ccl_resp.data or []) if r.get('venta') is not None]

    def _as_of(series: list, ts: str):
        # series ya viene ordenada por created_at asc — busca la última entrada <= ts.
        best = None
        for entry_ts, *_rest in series:
            if entry_ts <= ts:
                best = entry_ts, *_rest
            else:
                break
        return best

    ccl_dates = [d for d, _ in ccl_rows]
    ccl_vals  = [v for _, v in ccl_rows]

    def _ccl_momentum_as_of(date_str: str):
        # último índice con fecha <= date_str
        idx = None
        for i, d in enumerate(ccl_dates):
            if str(d) <= date_str[:10]:
                idx = i
            else:
                break
        if idx is None or idx < 5:
            return None
        base = ccl_vals[idx - 5]
        if not base:
            return None
        return (ccl_vals[idx] - base) / base * 100

    enriched = 0
    for r in relevant_rows:
        u_ticker = underlying_ticker_of.get(r['ticker'])
        h = int(r.get('horizon_bucket') or 0)
        ts = r.get('created_at')
        if not u_ticker or not h or not ts:
            continue
        series = cp_series.get((u_ticker, h))
        if series:
            hit = _as_of(series, ts)
            if hit:
                _, pct, conf = hit
                r['underlying_pred_pct']  = pct
                r['underlying_pred_conf'] = conf
        ccl_mom = _ccl_momentum_as_of(ts)
        if ccl_mom is not None:
            r['ccl_momentum_5d'] = ccl_mom
        if 'underlying_pred_pct' in r or 'ccl_momentum_5d' in r:
            enriched += 1
    print(f'[lr_train_daily] Etapa 23: {enriched}/{len(relevant_rows)} filas con par enriquecidas con dato real as-of', flush=True)


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

    # Keep-alive to prevent Render from spinning down during long training
    _ka_stop = threading.Event()
    threading.Thread(target=_keep_alive_loop, args=(_ka_stop,), daemon=True).start()

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
        try:
            _enrich_underlying_pair_features(sb, real_rows)
        except Exception as e:
            print(f'[lr_train_daily] Etapa 23 enrichment failed (non-fatal, features default to 0): {e}', flush=True)

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
            # LGBM-only: pay more for missing large moves, on top of the recency/reliability weight.
            w_train_lgbm = w_train * magnitude_weight(y_train)

            # Paso C: LightGBM daily — Optuna-tuned hyperparams + early stopping
            # (all pre-declared: tm.sum() in [20,30) skips the block below but the upsert dict
            # further down still references these names — pre-existing gap, not touching the
            # 20-sample floor itself, just making sure it doesn't NameError on that gap.)
            lgbm_model_b64 = lgbm_val_mae = None
            lgbm_error_p25 = lgbm_error_p50 = lgbm_error_p75 = lgbm_error_p90 = None
            capture_pct_top20 = capture_pct_rest = capture_n_top20 = capture_n_rest = None
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
                    m.fit(X_train_s[tm], y_train_norm[tm], sample_weight=w_train_lgbm[tm])
                    if has_val_d:
                        # evaluate in real % (denormalize) so MAE/dir-acc are comparable
                        pred_real = m.predict(X_val_s[vm]) * atr_val_valid
                        mae = float(np.mean(np.abs(y_val[vm] - pred_real)))
                        dir_acc = directional_accuracy(y_val[vm], pred_real)
                    else:
                        pred_real_tr = m.predict(X_train_s[tm]) * atr_tr_valid
                        mae = float(np.mean(np.abs(y_train[tm] - pred_real_tr)))
                        dir_acc = directional_accuracy(y_train[tm], pred_real_tr)
                    return lgbm_trial_score(mae, dir_acc)

                _study_d = _optuna.create_study(
                    direction='minimize', sampler=_optuna.samplers.TPESampler(seed=42)
                )
                _n_trials_d = 50 if int(tm.sum()) >= 30 else 25
                _study_d.optimize(_lgbm_daily_obj, n_trials=_n_trials_d, show_progress_bar=False)
                best_p_d = _study_d.best_params
                print(f'[optuna_daily] H={bucket} best={best_p_d} score={_study_d.best_value:.4f}', flush=True)

                # Eval set uses ATR-normalized targets so early stopping monitors normalized loss
                eval_set_d = [(X_val_s[vm], y_val[vm] / atr_val_valid)] if has_val_d else None
                callbacks_d = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_set_d else None
                lgb_reg = lgb.LGBMRegressor(
                    **best_p_d, n_estimators=600, random_state=42, verbose=-1,
                    objective='regression_l1',
                )
                lgb_reg.fit(X_train_s[tm], y_train_norm[tm], sample_weight=w_train_lgbm[tm],
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
                    capture_pct_top20, capture_pct_rest, capture_n_top20, capture_n_rest = \
                        capture_ratio_segments(y_val[vm], val_preds_real)
                    print(f'[capture_daily] H={bucket} top20%={capture_pct_top20} (n={capture_n_top20}) '
                          f'resto={capture_pct_rest} (n={capture_n_rest})', flush=True)
                else:
                    lgbm_error_p25 = lgbm_error_p50 = lgbm_error_p75 = lgbm_error_p90 = None
                    capture_pct_top20 = capture_pct_rest = capture_n_top20 = capture_n_rest = None
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
                'capture_pct_top20': capture_pct_top20,
                'capture_pct_rest': capture_pct_rest,
                'capture_n_top20': capture_n_top20,
                'capture_n_rest': capture_n_rest,
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
                cl_wtr_lgbm = cl_wtr * magnitude_weight(cl_ytr)
                scaler_cl = StandardScaler()
                cl_Xtr_s = scaler_cl.fit_transform(cl_Xtr)
                cl_Xvl_s = scaler_cl.transform(cl_Xvl) if len(cl_Xvl) > 0 else np.empty((0, cl_Xtr.shape[1]))
                tm_cl = ~np.isnan(cl_ytr)
                vm_cl = ~np.isnan(cl_yvl) if len(cl_Xvl_s) > 0 else np.zeros(0, dtype=bool)
                if tm_cl.sum() < 30:
                    continue

                def _cl_obj(trial, Xtr=cl_Xtr_s, ytr=cl_ytr, wtr=cl_wtr_lgbm, Xvl=cl_Xvl_s, yvl=cl_yvl, tmk=tm_cl, vmk=vm_cl):
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
                        preds = m.predict(Xvl[vmk])
                        mae = float(np.mean(np.abs(yvl[vmk] - preds)))
                        dir_acc = directional_accuracy(yvl[vmk], preds)
                    else:
                        preds = m.predict(Xtr[tmk])
                        mae = float(np.mean(np.abs(ytr[tmk] - preds)))
                        dir_acc = directional_accuracy(ytr[tmk], preds)
                    return lgbm_trial_score(mae, dir_acc)

                _study_cl = _optuna_cl.create_study(direction='minimize', sampler=_optuna_cl.samplers.TPESampler(seed=42))
                _study_cl.optimize(_cl_obj, n_trials=25, show_progress_bar=False)
                best_cl = _study_cl.best_params
                lgb_cl = lgb.LGBMRegressor(**best_cl, n_estimators=600, random_state=42, verbose=-1, objective='regression_l1')
                eval_cl = [(cl_Xvl_s[vm_cl], cl_yvl[vm_cl])] if vm_cl.sum() >= 5 else None
                cbs_cl = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)] if eval_cl else None
                lgb_cl.fit(cl_Xtr_s[tm_cl], cl_ytr[tm_cl], sample_weight=cl_wtr_lgbm[tm_cl], eval_set=eval_cl, callbacks=cbs_cl)
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
        # Etapa 27.6.1 — registrar el ÉXITO explícitamente. Sin esto no hay forma de distinguir
        # "no corrió" de "corrió y se cortó a la mitad", que es exactamente lo que pasó el 10/08.
        _log_training_event(
            sb, 'lr_train_daily_ok',
            f'Entrenamiento diario completo: {len(upserts)} buckets con parámetros escritos en '
            f'model_signed_params_daily.',
            samples=job.get('total_samples'))

    except Exception as e:
        job['status'] = 'error'
        job['error'] = str(e)
        print(f'[lr_train_daily] ERROR: {e}', flush=True)
        # Etapa 27.6.1 — y registrar el FALLO. El caso del 10/08 murió entre
        # upsert_daily_cluster_model y upsert_daily_signed_params (líneas consecutivas), así que
        # la mitad de los modelos quedó fresca y la otra mitad de tres días atrás, sin aviso.
        try:
            _log_training_event(
                sb, 'lr_train_daily_error',
                f'Entrenamiento diario ABORTADO: {e}. Los parámetros de model_signed_params_daily '
                f'pueden haber quedado desactualizados respecto de lgbm_cluster_models_daily — '
                f'comparar last_updated de ambas antes de confiar en las predicciones.',
                samples=job.get('total_samples'))
        except Exception:
            pass

    finally:
        _ka_stop.set()


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

# Etapa 29.3-fix2 (18/08/2026) — ver comentario largo más abajo. TTL tiene que superar el
# intervalo del cron (15 min) con margen, si no el progreso incremental nunca sobrevive de una
# corrida a la siguiente.
_LGBM_CACHE_TTL = 1800
# Tope de cuántos tickers NUEVOS se deserializan por request — acota el costo de una sola
# llamada para que no vuelva a matar el worker, sin bloquear la predicción (ver fallback
# ticker > cluster > sesión > global en _predict_lgbm_one, siempre devuelve algo).
_LGBM_MAX_NEW_TICKERS_PER_CALL = 15


def _enrich_indicators(ind: dict) -> dict:
    """Compute numeric versions of categorical/boolean indicator fields for inference."""
    out = dict(ind)
    orb = out.get('orb_breakout') or ''
    out['orb_breakout_num'] = 1.0 if orb == 'up' else -1.0 if orb == 'down' else 0.0
    out['bb_squeeze_num'] = 1.0 if out.get('bb_squeeze') else 0.0
    mc = out.get('macd_cross') or ''
    out['macd_cross_num'] = 1.0 if mc == 'bullish' else -1.0 if mc == 'bearish' else 0.0
    return out


def _lgbm_predict(m, X: 'np.ndarray') -> float:
    """Shape-safe predict: slices X to model's expected feature count for backward compat."""
    n = getattr(m, 'n_features_', X.shape[1])
    return float(m.predict(X[:, :n])[0])


def _load_lgbm_models_cached():
    """Returns dict[key] = (model, beta_spy). beta_spy=0 for old models without it.
    Deduplicates by horizon: all model_names sharing a horizon reuse the same object."""
    global _lgbm_cache, _lgbm_cache_ts
    if time.time() - _lgbm_cache_ts < 600 and _lgbm_cache:
        return _lgbm_cache
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    resp = sb.table('model_learned_params_intraday').select(
        'model_name,horizon_minutes,lgbm_model,beta_spy'
    ).execute()
    new_cache: dict = {}
    horizon_models: dict = {}  # horizon_minutes -> (model, beta) — unpickled once per horizon
    for row in resp.data or []:
        if row.get('lgbm_model'):
            h = row['horizon_minutes']
            key = f"{row['model_name']}:{h}"
            try:
                if h not in horizon_models:
                    model = pickle.loads(base64.b64decode(row['lgbm_model']))
                    beta = float(row.get('beta_spy') or 0.0)
                    horizon_models[h] = (model, beta)
                new_cache[key] = horizon_models[h]
            except Exception:
                pass
    _lgbm_cache = new_cache
    _lgbm_cache_ts = time.time()
    print(f'[lgbm_cache] loaded {len(new_cache)} keys, {len(horizon_models)} unique models', flush=True)
    return _lgbm_cache


# Etapa 29.3-fix (14/08/2026) — INCIDENTE DE PRODUCCIÓN, no cosmético.
#
# Las tres funciones de abajo cargaban SIEMPRE la tabla entera y deserializaban cada fila con
# pickle.loads()/LightGBM nativo, sin filtrar por lo que la request realmente necesitaba. Mientras
# lgbm_ticker_models_intraday tuvo pocas filas (0 en el audit del 11/08) esto no se notaba. La
# Etapa 29.1 (paginación real del entrenamiento) hizo que esa tabla pasara a 181 filas de la noche
# a la mañana — y una carga en frío de ~210 modelos (181 ticker + 18 cluster + 11 session, éstos
# últimos ~263 KB promedio, más pesados de deserializar que los de ticker) empezó a superar el
# tiempo que el worker de gunicorn tolera sin heartbeat. Confirmado en logs reales de Render:
#   [CRITICAL] WORKER TIMEOUT (pid:65) ... pickle.loads(...) -> LGBM_BoosterLoadModelFromString
#   Worker (pid:65) was sent SIGKILL!
# Y como el worker muere A MITAD de poblar el caché, el caché NUNCA llega a calentarse: cada
# worker nuevo repite la carga completa desde cero y vuelve a morir — un bucle que no se autocura.
#
# El arreglo: cachear de forma INCREMENTAL, filtrando por lo que la request pide. Cada llamada
# sólo trae y deserializa lo que todavía no tiene cacheado, y lo suma a lo ya cacheado (no lo
# reemplaza) — así el costo real queda acotado al universo REALMENTE consultado en la práctica
# (~78 activos intradiarios activos, llamados una y otra vez desde el mismo cron cada 15 min),
# no a los ~181-210 que acumula el histórico de 90 días con tickers que ya ni están activos.
# `filter_values=None` preserva el comportamiento viejo (traer todo) para quien lo necesite a
# propósito — hoy nadie lo usa así.
#
# Etapa 29.3-fix2 (18/08/2026) — el fix de arriba no alcanzaba, confirmado con logs reales de
# Render en mercado abierto: `[CRITICAL] WORKER TIMEOUT` dentro de este mismo
# `pickle.loads`/`LGBM_BoosterLoadModelFromString`, en TODAS las corridas del día (0/78 lgbm en
# `model_changelog` cada 15 min desde el 11/08, no sólo en frío). Dos motivos, los dos vigentes
# a la vez:
#   1. El TTL (600s) es MENOR al intervalo del cron (`intraday-prediccion-15min`, cada 900s) —
#      el caché siempre estaba vencido antes de la próxima corrida, así que "incremental" nunca
#      tenía nada previo sobre lo cual incrementar: cada llamada volvía a pedir los ~78 tickers
#      enteros de cero.
#   2. Ese cold-load de ~78 tickers (hasta 234 filas con los 3 horizontes) por sí solo ya tarda
#      más de lo que el worker tolera en producción — el `--timeout 600` del Procfile no explica
#      por qué muere en segundos, no en 10 minutos; probablemente el Start Command real en el
#      dashboard de Render no sea el del Procfile (no verificable desde acá, sin acceso a esa
#      cuenta — si una sesión futura tiene ese acceso, vale la pena confirmarlo y de paso
#      simplificar este workaround).
# Con el worker muriendo a mitad de carga, el caché nunca terminaba de poblarse — un bucle que
# no se autocuraba, igual que el incidente original que motivó el fix de arriba.
#
# Mitigación (sin depender de arreglar Render): `_LGBM_MAX_NEW_TICKERS_PER_CALL` acota cuántos
# tickers NUEVOS se deserializan por request, y `_LGBM_CACHE_TTL` (30 min) supera el intervalo
# del cron con margen. El progreso ahora sí sobrevive de una corrida a la siguiente — el universo
# de 78 activos se termina de cachear en ~5-6 corridas (~75-90 min) en vez de una sola que nunca
# llegaba a completarse. Mientras un ticker todavía no tiene su modelo propio cacheado,
# `_predict_lgbm_one` ya cae solo al fallback cluster/sesión/global (ver esa función) — la
# predicción sale igual, sólo menos personalizada hasta que le toque su turno. Verificar con
# `select model_name, count(*) from model_predictions_intraday where created_at > now() -
# interval '1 day' group by 1` que `lgbm` deja de estar en 0.

def _load_lgbm_session_models_cached(sessions: set | None = None):
    """Per-session LGBM models. Keys: 'model_name:horizon:session'. Values: (model, beta_spy)."""
    global _lgbm_session_cache, _lgbm_session_cache_ts
    if time.time() - _lgbm_session_cache_ts > _LGBM_CACHE_TTL:
        _lgbm_session_cache = {}
    if sessions is not None:
        have = {k.split(':')[2] for k in _lgbm_session_cache}
        missing = sorted(set(sessions) - have)
        if not missing:
            return _lgbm_session_cache
    else:
        missing = None
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    q = sb.table('lgbm_session_models_intraday').select(
        'model_name,horizon_minutes,market_session,lgbm_model,beta_spy'
    )
    if missing is not None:
        q = q.in_('market_session', missing)
    resp = q.execute()
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['model_name']}:{row['horizon_minutes']}:{row['market_session']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                _lgbm_session_cache[key] = (model, beta)
            except Exception:
                pass
    _lgbm_session_cache_ts = time.time()
    return _lgbm_session_cache


def _load_lgbm_ticker_models_cached(tickers: set | None = None):
    """Per-ticker LGBM models. Keys: 'TICKER:horizon'. Values: (model, beta_spy, train_samples)."""
    global _lgbm_ticker_cache, _lgbm_ticker_cache_ts
    if time.time() - _lgbm_ticker_cache_ts > _LGBM_CACHE_TTL:
        _lgbm_ticker_cache = {}
    if tickers is not None:
        have = {k.split(':')[0] for k in _lgbm_ticker_cache}
        missing = sorted(set(tickers) - have)
        if not missing:
            return _lgbm_ticker_cache
        # Etapa 29.3-fix2: acota el trabajo de ESTA request — ver comentario largo arriba de
        # _load_lgbm_session_models_cached. Los tickers que quedan afuera este turno siguen
        # recibiendo predicción vía el fallback de _predict_lgbm_one, sólo que menos
        # personalizada hasta que les toque cachearse.
        missing = missing[:_LGBM_MAX_NEW_TICKERS_PER_CALL]
    else:
        missing = None
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    q = sb.table('lgbm_ticker_models_intraday').select(
        'ticker,horizon_minutes,lgbm_model,beta_spy,train_samples'
    )
    if missing is not None:
        q = q.in_('ticker', missing)
    resp = q.execute()
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['ticker']}:{row['horizon_minutes']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                n_samples = int(row.get('train_samples') or 0)
                _lgbm_ticker_cache[key] = (model, beta, n_samples)
            except Exception:
                pass
    _lgbm_ticker_cache_ts = time.time()
    if missing is not None:
        cached_tickers = {k.split(':')[0] for k in _lgbm_ticker_cache}
        print(f'[lgbm_ticker_cache] +{len(missing)} tickers this call, '
              f'{len(cached_tickers)} cached total', flush=True)
    return _lgbm_ticker_cache


def _load_lgbm_cluster_models_cached(clusters: set | None = None):
    """Per-cluster LGBM models. Keys: 'cluster_name:horizon'. Values: (model, beta_spy)."""
    global _lgbm_cluster_cache, _lgbm_cluster_cache_ts
    if time.time() - _lgbm_cluster_cache_ts > _LGBM_CACHE_TTL:
        _lgbm_cluster_cache = {}
    if clusters is not None:
        have = {k.split(':')[0] for k in _lgbm_cluster_cache}
        missing = sorted(set(clusters) - have)
        if not missing:
            return _lgbm_cluster_cache
    else:
        missing = None
    from supabase import create_client
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    q = sb.table('lgbm_cluster_models_intraday').select(
        'cluster_name,horizon_minutes,lgbm_model,beta_spy'
    )
    if missing is not None:
        q = q.in_('cluster_name', missing)
    resp = q.execute()
    for row in resp.data or []:
        if row.get('lgbm_model'):
            key = f"{row['cluster_name']}:{row['horizon_minutes']}"
            try:
                model = pickle.loads(base64.b64decode(row['lgbm_model']))
                beta = float(row.get('beta_spy') or 0.0)
                _lgbm_cluster_cache[key] = (model, beta)
            except Exception:
                pass
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
        # Etapa 17 (backlog): a horizontes largos (visto sobre todo en 60/90d, peor en modelos
        # 'cluster' — muestra de entrenamiento más chica que el global) el modelo ocasionalmente
        # extrapola muy por fuera de lo que él mismo considera típico para este bucket — casos
        # reales vistos: +95.84% a 90d con avg_actual_mag=20.2% (4.7x — el caso que motivó este
        # fix), universo completo con máximos de hasta 420% a 90d. No es un fix de la causa (no se
        # investigó por qué el modelo extrapola así — candidatos: overfitting en clusters con pocas
        # muestras, falta de regularización a horizontes largos) — es un piso de sanidad basado en
        # la propia calibración del modelo (avg_actual_mag, dato real medido en entrenamiento, no
        # un número inventado). 3x (no 5x — con 5x el caso real de arriba, 4.7x, casi no se hubiera
        # tocado, que es exactamente lo que este fix tiene que evitar) sigue siendo generoso: ya es
        # más que el promedio real de movimiento en ese horizonte. Ver REDISENO/STATUS.md.
        MAG_CLAMP_MULT = 3.0
        mag_cap = MAG_CLAMP_MULT * avg_mag
        pred_clamped = max(-mag_cap, min(mag_cap, pred))
        return jsonify({
            'ok': True,
            'predicted_pct': round(pred_clamped, 4),
            'predicted_pct_raw': round(pred, 4) if pred_clamped != pred else None,
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
        indicators = _enrich_indicators(indicators)
        X = np.array([[float(indicators.get(fn) or 0) for fn in LR_FEATURE_NAMES]])
        atr_idx = LR_FEATURE_NAMES.index('atr_pct')
        atr_scale = max(0.1, float(X[0, atr_idx]))
        spy_r15 = float(indicators.get('spy_return_15m') or 0)
        pred = _lgbm_predict(m, X) * atr_scale + beta * spy_r15
        return jsonify({'ok': True, 'predicted_pct': round(pred, 4)})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


def _predict_lgbm_one(indicators: dict, ticker: str, models, session_models,
                      ticker_models, cluster_models) -> dict:
    """Etapa 29.3 — el núcleo de predict_lgbm_all, extraído para que el endpoint batch no
    duplique la lógica de prioridad ticker > cluster > sesión > global ni recargue los modelos
    una vez por activo. Devuelve {predictions, session, ticker_models_used, cluster_models_used}."""
    indicators = _enrich_indicators(indicators)
    X = np.array([[float(indicators.get(fn) or 0) for fn in LR_FEATURE_NAMES]])
    atr_idx = LR_FEATURE_NAMES.index('atr_pct')
    atr_scale = max(0.1, float(X[0, atr_idx]))
    spy_r15 = float(indicators.get('spy_return_15m') or 0)
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
                pred_t = _lgbm_predict(t_model, X) * atr_scale + t_beta * spy_r15
                pred_c = _lgbm_predict(c_model, X) * atr_scale + c_beta * spy_r15
                predictions[key] = round(w_t * pred_t + (1.0 - w_t) * pred_c, 4)
                blended = True
            else:
                m, beta = t_model, t_beta
        elif c_key and c_key in cluster_models:
            m, beta = cluster_models[c_key]
            cluster_used += 1
        else:
            # New storage: '__session__:horizon:session'; old rows keyed by model_name
            sess_key = f'__session__:{horizon}:{session}'
            sess_entry = session_models.get(sess_key) or session_models.get(f'{key}:{session}')
            if sess_entry:
                m, beta = sess_entry
            else:
                m, beta = global_m, global_beta
        if not blended:
            pred_idio = _lgbm_predict(m, X) * atr_scale
            predictions[key] = round(pred_idio + beta * spy_r15, 4)
    return {
        'predictions': predictions,
        'session': session,
        'ticker_models_used': ticker_used,
        'cluster_models_used': cluster_used,
    }


@app.route('/api/predict_lgbm_all', methods=['POST', 'OPTIONS'])
def predict_lgbm_all():
    """LightGBM para UN activo, todos los (modelo, horizonte) en una llamada.
    Etapa 29.3: conservado por compatibilidad — crear-prediccion-intraday ahora usa
    /api/predict_lgbm_batch, que hace lo mismo para los ~78 activos de una sola vez."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json() or {}
    ticker = body.get('ticker', '')
    indicators = body.get('indicators', {}) or {}
    try:
        models = _load_lgbm_models_cached()
        if not models:
            return jsonify({'ok': True, 'predictions': {}, 'models_loaded': 0})
        # Etapa 29.3-fix: filtrar por lo que ESTA request necesita, no traer las tablas enteras.
        mso = float(indicators.get('minutes_since_open') or 0)
        session = _get_market_session(mso)
        cluster = TICKER_CLUSTERS.get(ticker, '') if ticker else ''
        out = _predict_lgbm_one(
            indicators, ticker, models,
            _load_lgbm_session_models_cached({session}),
            _load_lgbm_ticker_models_cached({ticker} if ticker else set()),
            _load_lgbm_cluster_models_cached({cluster} if cluster else set()))
        return jsonify({'ok': True, 'models_loaded': len(models), **out})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/predict_lgbm_batch', methods=['POST', 'OPTIONS'])
def predict_lgbm_batch():
    """Etapa 29.3 — LightGBM para TODOS los activos de la corrida en una sola llamada.

    Por qué existe. crear-prediccion-intraday hacía una llamada HTTP por activo con
    AbortSignal.timeout(6000): 78 activos en tandas de 15 contra un free tier de un solo worker.
    La mayoría timeouteaba y el `catch { return {} }` lo convertía en "este activo no tiene voto
    lgbm", sin dejar rastro. Medido: el 10/08 se crearon 16 filas de lgbm contra 5.226 de ridge
    (0,3%), y el 07/08 CERO — siendo lgbm el único voto intradiario con acierto real (63-65% a 60
    y 120 min). O sea el mejor voto del roster estaba efectivamente apagado por un timeout.

    Entrada: {"assets": [{"ticker": "...", "indicators": {...}}, ...]}
    Salida:  {"ok": true, "results": {"<ticker>": {"predictions": {...}, ...}}, "failed": {...}}

    Los modelos se cargan UNA vez para todo el lote. Un activo que falla no tumba el lote: se
    reporta en `failed` para que el llamador pueda contarlo en vez de perderlo en silencio.
    """
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json() or {}
    assets = body.get('assets') or []
    if not isinstance(assets, list):
        return jsonify({'ok': False, 'error': 'assets must be a list'}), 400
    if len(assets) > 200:
        return jsonify({'ok': False, 'error': 'too many assets (max 200)'}), 400
    try:
        models = _load_lgbm_models_cached()
        if not models:
            return jsonify({'ok': True, 'results': {}, 'failed': {}, 'models_loaded': 0})

        # Etapa 29.3-fix: precalcular la UNIÓN de tickers/sesiones/clusters que este lote
        # necesita, y pedir sólo eso — ver el comentario largo sobre el incidente de producción
        # arriba de _load_lgbm_ticker_models_cached(). Es aritmética pura, sin red ni DB, así
        # que hacerlo antes de cargar no cuesta nada.
        needed_tickers = {(a or {}).get('ticker', '') for a in assets if (a or {}).get('ticker')}
        needed_sessions, needed_clusters = set(), set()
        for a in assets:
            ind = (a or {}).get('indicators', {}) or {}
            needed_sessions.add(_get_market_session(float(ind.get('minutes_since_open') or 0)))
            t = (a or {}).get('ticker', '')
            c = TICKER_CLUSTERS.get(t, '') if t else ''
            if c:
                needed_clusters.add(c)

        session_models = _load_lgbm_session_models_cached(needed_sessions)
        ticker_models = _load_lgbm_ticker_models_cached(needed_tickers)
        cluster_models = _load_lgbm_cluster_models_cached(needed_clusters)

        results, failed = {}, {}
        for a in assets:
            ticker = (a or {}).get('ticker', '')
            try:
                results[ticker] = _predict_lgbm_one(
                    (a or {}).get('indicators', {}), ticker, models,
                    session_models, ticker_models, cluster_models)
            except Exception as e:
                failed[ticker] = str(e)[:200]

        print(f'[predict_batch] {len(results)} ok, {len(failed)} fallidos, '
              f'{len(models)} modelos globales, filtrado a {len(needed_tickers)} tickers/'
              f'{len(needed_sessions)} sesiones/{len(needed_clusters)} clusters', flush=True)
        return jsonify({
            'ok': True, 'results': results, 'failed': failed,
            'models_loaded': len(models),
            'assets_requested': len(assets),
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


# ── Etapa 30: motor de trading automático (hoy sólo papel) ────────────────────
# Corre vía cron (mismo patrón de auth que el resto de este archivo: _check_secret() +
# XGB_INTERNAL_SECRET), determinístico — sin LLM en el loop. Decide entradas/salidas de
# `auto_trades` contra el filtro de costo IOL + el gate estadístico de `scorecard_bolsas` + los
# límites de riesgo de `auto_trading_config`. Ver REDISENO/ETAPA-30-motor-trading-automatico.md.
#
# Comisiones IOL: mismos números que `routeInstrument()`/`DEFAULT_COSTO_CONFIG` en
# dashboard/lib/tracking.ts (Fase A/B de la Etapa 30) — duplicado acá porque este código corre en
# Python vía cron y no puede importar TS. Si se recalibra uno, recalibrar el otro.
AUTO_COSTO_PCT = {
    'BYMA': {'normal': 1.33, 'intradia': 0.67},
    'US':   {'normal': 0.85, 'intradia': 0.85},
}

DEFAULT_STOP_LOSS_PCT = -2.0  # fallback si la predicción no trae stop_loss_pct sugerido


def _auto_route(core_bucket):
    """Espejo de routeInstrument() en dashboard/lib/tracking.ts. 'cedear_underlying' no resuelve
    acá la contraparte 'us' de la misma compañía (a diferencia de la versión TS con
    hasUsCounterpart) — simplificación de esta primera versión: esas filas quedan sin venue
    (informativas, no se operan) hasta que se necesite ese mapeo.

    Etapa 30 (continuación, 19/08/2026, a pedido explícito del usuario): 'cedear_arg' SACADO del
    motor de trading por completo (ni papel ni vivo) — la declaración jurada que exige IOL es por
    orden, sin forma de dejarla aceptada de antemano, así que nunca puede operar en vivo, y
    mantenerlo sólo en papel ya no aporta (probado con la cuenta real el 19/08). El sistema de
    predicciones/scorecard de CEDEARs sigue intacto, esto es sólo el motor de trading."""
    if core_bucket in ('us', 'adr_arg'):
        return 'US'
    if core_bucket == 'accion_arg_local':
        return 'BYMA'
    return None


def _auto_costo_pct(venue, is_intraday):
    return AUTO_COSTO_PCT.get(venue, {}).get('intradia' if is_intraday else 'normal')


# ── Etapa 30 (continuación, 18/08/2026): cliente REST real de IOL para operar en vivo ─────────
# Contrato confirmado contra la documentación oficial (api.invertironline.com/Help/Autenticacion)
# más una implementación de referencia real y funcionando (github.com/pgallar/iol-mcp, revisada el
# 18/08/2026) — no adivinado. Credenciales por env var (IOL_USERNAME/IOL_PASSWORD en Render, nunca
# en este repo ni en una tabla). Pedido explícito del usuario: operar en vivo YA, salteando el gate
# estadístico a sabiendas de que hoy no hay ninguna bolsa validada (ver ETAPA-30). No hay endpoint
# nativo de stop-loss/take-profit confirmado en la referencia disponible — se implementa por
# polling (mismo chequeo de precio que ya hace _run_motor_salidas cada 15 min) más una orden de
# venta REAL cuando se dispara, en vez de adivinar un endpoint no documentado para algo que manda
# plata real.
IOL_BASE_URL = 'https://api.invertironline.com'
IOL_USERNAME = os.environ.get('IOL_USERNAME', '')
IOL_PASSWORD = os.environ.get('IOL_PASSWORD', '')
_iol_token_cache = {'access_token': None, 'expiry': None}

IOL_MERCADO_ARS = 'bCBA'
_IOL_MERCADO_BY_EXCHANGE = {'NASDAQ': 'nASDAQ', 'NYSE': 'nYSE', 'AMEX': 'aMEX'}


def _iol_mercado_for(venue, exchange):
    if venue == 'BYMA':
        return IOL_MERCADO_ARS
    return _IOL_MERCADO_BY_EXCHANGE.get((exchange or '').upper(), 'nYSE')


def _iol_authenticate():
    import httpx
    resp = httpx.post(
        f'{IOL_BASE_URL}/token',
        data={'username': IOL_USERNAME, 'password': IOL_PASSWORD, 'grant_type': 'password'},
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    _iol_token_cache['access_token'] = data['access_token']
    _iol_token_cache['expiry'] = (
        datetime.utcnow() + timedelta(seconds=int(data['expires_in'])) - timedelta(minutes=2)
    )


def _iol_headers():
    if (not _iol_token_cache['access_token'] or not _iol_token_cache['expiry']
            or datetime.utcnow() >= _iol_token_cache['expiry']):
        _iol_authenticate()
    return {'Authorization': f"Bearer {_iol_token_cache['access_token']}", 'Content-Type': 'application/json'}


def _iol_request(method, path, **kwargs):
    import httpx
    headers = _iol_headers()
    resp = httpx.request(method, f'{IOL_BASE_URL}{path}', headers=headers, timeout=20, **kwargs)
    if resp.status_code == 401:
        _iol_authenticate()
        headers = _iol_headers()
        resp = httpx.request(method, f'{IOL_BASE_URL}{path}', headers=headers, timeout=20, **kwargs)
    # Etapa 30 (25/08/2026): resp.raise_for_status() (versión anterior) sólo daba el status y la
    # URL (ej. "400 Bad Request"), sin el cuerpo de la respuesta -- que es justo donde IOL manda
    # el motivo real del rechazo (ej. {"codigo":"MS-ORD-0033","mensaje":"Una orden a Precio de
    # Mercado no puede tener un precio límite"}, el error real detrás de dos bugs seguidos en esta
    # misma sesión). Diagnosticar un 400 real obligaba a reproducirlo a mano contra validate_order
    # (MCP) en vez de leerlo directo de los logs. Ahora el texto de la excepción incluye el body
    # entero -- visible en los print() de _run_motor_entradas/_run_motor_salidas (logs de Render)
    # y en el campo `live_errors` de la respuesta de /api/motor_ejecucion.
    if resp.status_code >= 400:
        raise Exception(f'{resp.status_code} {resp.reason_phrase}: {resp.text[:500]}')
    # Etapa 30 (25/08/2026, hallazgo grave, confirmado con la cuenta real vía MCP —
    # get_portfolio/get_activities/get_balance — DESPUÉS de que el motor ya había creado 2 filas
    # auto_trades modo='vivo' creyendo que la compra de LOMA.BA había salido bien): la orden real
    # devolvió éxito HTTP con el body VACÍO, y el código anterior (`resp.json() if resp.text else
    # {}`) trataba eso como una respuesta válida — nunca se movió un peso ni apareció la operación
    # en la cuenta real. La referencia funcionando (github.com/pgallar/iol-mcp,
    # src/iol/http_client.py) nunca contempla un body vacío como éxito — siempre asume JSON real
    # en 200/201. Un body vacío ahora es un error, no un `{}` silencioso — con el status code real
    # en el mensaje (200/201/202/lo que sea) para poder diagnosticar la próxima vez sin tener que
    # reproducirlo contra la cuenta real de nuevo. Filas de auto_trades ya creadas por el bug
    # viejo se borraron a mano (25/08/2026) — no son reales, no había ninguna operación detrás.
    if not resp.text:
        raise Exception(f'{resp.status_code} {resp.reason_phrase}: respuesta vacía (sin body) — no se puede confirmar la orden')
    return resp.json()


def _iol_estado_cuenta():
    return _iol_request('GET', '/api/v2/estadocuenta')


def _iol_available_cash_both():
    """Efectivo disponible real en pesos y en dólares (Cuenta Estados Unidos específicamente,
    `tipo='inversion_Estados_Unidos_Dolares'` — NO la sub-cuenta en dólares del lado Argentina,
    que es una cuenta distinta con otro `numero`) en una sola llamada a estadocuenta — el shape
    exacto no se pudo confirmar sin credenciales reales, así que busca de forma defensiva en vez de
    asumir una ruta fija. Devuelve (None, None) en error, nunca una excepción, para que el
    llamador salte la entrada en lugar de arriesgar un tamaño de posición mal calculado."""
    try:
        data = _iol_estado_cuenta()
        ars, usd = None, None
        for cuenta in data.get('cuentas', []) or []:
            moneda = str(cuenta.get('moneda', '')).lower()
            disponible = cuenta.get('disponible')
            if disponible is None:
                continue
            if ars is None and ('peso' in moneda or moneda == 'ars'):
                ars = float(disponible)
            if usd is None and cuenta.get('tipo') == 'inversion_Estados_Unidos_Dolares':
                usd = float(disponible)
        return ars, usd
    except Exception as e:
        print(f'[iol_estado_cuenta] error: {e}', flush=True)
        return None, None


def _iol_available_ars_cash():
    return _iol_available_cash_both()[0]


def _iol_available_usd_cash():
    return _iol_available_cash_both()[1]


def _iol_simbolo_for(ticker):
    """IOL espera el ticker pelado, sin sufijo de exchange — este proyecto guarda los activos
    BYMA (accion_arg_local) con sufijo '.BA' (convención estilo Yahoo Finance, usada en
    predicciones/UI), pero la API real de IOL rechaza eso con 400 Bad Request. Confirmado
    25/08/2026: 'LOMA.BA' -> 400 al comprar de verdad; el mismo ticker sin sufijo ('LOMA', mercado
    BCBA) valida OK contra la API real de IOL. Esto explica por qué auto_trades nunca tuvo una
    sola fila modo='vivo' pese a que cash/DDJJ/gate ya estaban resueltos — cada intento de compra
    real fallaba acá, silenciosamente (skipped_reasons['vivo_orden_fallo'], sin persistir el motivo
    exacto). Único caso conocido: los tickers de EEUU/ADR (venue 'US') no llevan sufijo."""
    return ticker[:-3] if ticker.endswith('.BA') else ticker


# Etapa 30 (fix 25/08/2026, tercera vuelta la misma tarde — ver REDISENO/ETAPA-30 para el
# historial completo de los 400 reales encontrados en orden): 'precio' siempre viaja (no es
# opcional, confirmado contra la referencia real github.com/pgallar/iol-mcp,
# src/iol/operar/client.py). 'cantidad' y 'monto' son ambos opcionales ahí, pero IOL en
# producción rechazó mandar los dos juntos en una orden a mercado: "Precio mercado no debe tener
# cantidad" — a mercado hay que decirle CUÁNTA PLATA gastar (monto), no cuántas acciones, y deja
# que la cotización del momento determine cuántas entran. `cantidad` sigue siendo un parámetro de
# esta función (el llamador la sigue calculando para saber cuántas acciones espera que entren y
# poder registrar auto_trades), sólo se dejó de mandar en el body.
def _iol_comprar(mercado, simbolo, precio, cantidad, plazo='t0'):
    validez = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59')
    return _iol_request('POST', '/api/v2/operar/Comprar', json={
        'mercado': mercado, 'simbolo': simbolo, 'precio': precio, 'plazo': plazo,
        'validez': validez, 'monto': round(cantidad * precio, 2),
        'tipoOrden': 'precioMercado',
    })


def _iol_vender(mercado, simbolo, precio, cantidad, plazo='t0'):
    validez = (datetime.utcnow() + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59')
    return _iol_request('POST', '/api/v2/operar/Vender', json={
        'mercado': mercado, 'simbolo': simbolo, 'cantidad': cantidad, 'precio': precio,
        'validez': validez, 'plazo': plazo, 'tipoOrden': 'precioMercado',
    })


def _iol_plazo_for(venue):
    """t0 (Contado Inmediato) vale en BCBA; NASDAQ/NYSE lo rechaza (probado con la cuenta real,
    AAPL) — requiere t1. No bloquea operar la misma posición el mismo día, es sólo cuándo liquida
    contablemente, no una restricción de day-trading."""
    return 't0' if venue == 'BYMA' else 't1'


def _run_motor_entradas(sb, source):
    """source: 'intraday' (todos los horizontes) | 'daily' (sólo h=1). Evalúa predicciones
    abiertas nuevas contra costo + gate estadístico + riesgo, y abre `auto_trades` en papel donde
    corresponda. No abre nada si el kill switch está prendido, no hay portfolio para la moneda, o
    el gate estadístico no pasa — eso es comportamiento correcto, no una falla."""
    cfg = (sb.from_('auto_trading_config').select('*').eq('id', True).single().execute().data) or {}

    # Etapa 30 (continuación, 19/08/2026, a pedido explícito del usuario): guarda una foto del
    # efectivo real en cada corrida, para que el dashboard lo muestre sin necesitar acceso directo
    # a IOL (que no tiene). Se hace ANTES del corte por kill_switch, para que la foto se mantenga
    # fresca incluso con el motor frenado.
    snapshot_ars, snapshot_usd = _iol_available_cash_both()
    if snapshot_ars is not None or snapshot_usd is not None:
        snap_update = {'last_known_cash_at': datetime.utcnow().isoformat()}
        if snapshot_ars is not None:
            snap_update['last_known_ars_cash'] = snapshot_ars
        if snapshot_usd is not None:
            snap_update['last_known_usd_cash'] = snapshot_usd
        sb.from_('auto_trading_config').update(snap_update).eq('id', True).execute()
        cfg.update(snap_update)

    if cfg.get('kill_switch', True):
        return {'ok': True, 'skipped': 'kill_switch', 'opened': 0}

    portfolios = {p['currency']: p for p in (sb.from_('auto_portfolios').select('*').execute().data or [])}
    if not portfolios:
        return {'ok': True, 'skipped': 'no_portfolios', 'opened': 0}

    # Etapa 30 (continuación, egress — el usuario ya se pasó del límite de Supabase antes, cuidado):
    # sin acotar, esto trae TODA la historia de auto_trades, creciendo sin límite con cada semana
    # que pasa. Sólo hace falta lo abierto (para slots/reparto de capital) y lo cerrado HOY (para el
    # límite de pérdida diaria) — 3 días de margen cubre de sobra cualquier trade todavía abierto
    # (ningún horizonte de este motor pasa el día) sin traer meses de historial de vuelta.
    since_cutoff = (datetime.utcnow() - timedelta(days=3)).isoformat()
    all_trades = sb.from_('auto_trades').select(
        'id, portfolio_id, status, closed_at, pnl_monto, daily_prediction_id, intraday_prediction_id, '
        'modo, prediction_type, monto_invertido, venue'
    ).gte('opened_at', since_cutoff).execute().data or []
    open_trades = [t for t in all_trades if t['status'] == 'abierta']
    already_traded_pred_ids = (
        {t['daily_prediction_id'] for t in all_trades if t.get('daily_prediction_id')} |
        {t['intraday_prediction_id'] for t in all_trades if t.get('intraday_prediction_id')}
    )

    # Etapa 30 (fix 20/08/2026, bug real encontrado en producción: 10 posiciones en papel llenaban
    # max_concurrent_positions y con eso bloqueaban también el cupo de vivo, aunque nunca hubiera
    # ninguna posición vivo abierta todavía). Cupos separados por modo — papel y vivo compiten cada
    # uno sólo contra su propio conteo, uno no le saca lugar al otro.
    max_positions = int(cfg.get('max_concurrent_positions', 10))
    slots_left_papel = max_positions - sum(1 for t in open_trades if t.get('modo') != 'vivo')
    slots_left_vivo = max_positions - sum(1 for t in open_trades if t.get('modo') == 'vivo')
    if slots_left_papel <= 0 and slots_left_vivo <= 0:
        return {'ok': True, 'skipped': 'max_concurrent_positions', 'opened': 0}

    # Etapa 30 (continuación, 19/08/2026, corregido a pedido explícito del usuario tras un
    # malentendido): en Argentina (BYMA) el reparto es por HORIZONTE ($10.000 intradiario / $2.000
    # diario, cada `source` con su propio tope — mismo mecanismo que la primera versión). En EEUU
    # (hoy deshabilitado, `live_enabled_us=false`) es un solo pool sin sub-repartir por horizonte —
    # no hace falta esa finura mientras el capital ahí sea tan chico, y así queda listo para
    # activarse solo el día que haya más dólares, sin volver a tocar este código.
    live_committed_byma_this_source = sum(
        float(t['monto_invertido']) for t in open_trades
        if t.get('modo') == 'vivo' and t.get('venue') == 'BYMA' and t.get('prediction_type') == source
    )
    live_committed_us = sum(
        float(t['monto_invertido']) for t in open_trades
        if t.get('modo') == 'vivo' and t.get('venue') == 'US'
    )
    live_capital_cap_by_venue = {
        'BYMA': float(cfg.get('live_capital_intraday_ars' if source == 'intraday' else 'live_capital_daily_ars', 0)),
        'US': float(cfg.get('live_capital_usd', 0)),
    }
    live_committed_by_venue = {'BYMA': live_committed_byma_this_source, 'US': live_committed_us}

    today = datetime.utcnow().date().isoformat()
    daily_pnl_by_portfolio = defaultdict(float)
    for t in all_trades:
        if t.get('closed_at') and str(t['closed_at'])[:10] == today and t.get('pnl_monto') is not None:
            daily_pnl_by_portfolio[t['portfolio_id']] += float(t['pnl_monto'])

    # Etapa 30 (continuación): el límite de pérdida diaria tiene que medirse contra plata REAL para
    # las monedas en vivo, no contra auto_portfolios.capital_inicial (placeholder de papel,
    # $1.000.000 — 100x más grande que el capital real de ~$10.000, lo que hacía que el freno nunca
    # se fuera a disparar de verdad). capital_base reconstruye "cuánto había al empezar hoy" como
    # efectivo actual menos lo ya ganado/perdido hoy — no incluye el valor de posiciones abiertas
    # todavía sin cerrar, así que subestima el capital total mientras algo está invertido, lo cual
    # empuja el freno a activarse ANTES de lo estrictamente necesario — más conservador, no al revés.
    # Reusa la foto ya tomada al principio de la función — no hace falta pedirle a IOL el mismo
    # dato dos veces en la misma corrida.
    real_ars_cash = snapshot_ars if cfg.get('live_enabled_byma', False) else None
    real_usd_cash = snapshot_usd if cfg.get('live_enabled_us', False) else None

    blocked_portfolio_ids = set()
    for pf in portfolios.values():
        pnl_today = daily_pnl_by_portfolio.get(pf['id'], 0.0)
        if pf['currency'] == 'ars' and real_ars_cash is not None:
            capital_base = real_ars_cash - pnl_today
        elif pf['currency'] == 'usd' and real_usd_cash is not None:
            capital_base = real_usd_cash - pnl_today
        else:
            capital_base = float(pf['capital_inicial'])
        loss_limit = -abs(float(cfg.get('max_daily_loss_pct', 3))) / 100.0 * capital_base
        if pnl_today <= loss_limit:
            blocked_portfolio_ids.add(pf['id'])

    scorecard_horizon_unit = 'minutes' if source == 'intraday' else 'days'
    scorecard = sb.from_('scorecard_bolsas').select(
        'asset_id, currency, horizon_unit, horizon_bucket, estado, expectancy_net_pct'
    ).eq('horizon_unit', scorecard_horizon_unit).execute().data or []
    scorecard_by_key = {(s['asset_id'], s['currency'], s['horizon_unit'], s['horizon_bucket']): s for s in scorecard}

    assets_by_id = {a['id']: a for a in (
        sb.from_('assets').select('id, ticker, currency, core_bucket, exchange').execute().data or [])}

    if source == 'intraday':
        preds = sb.from_('consensus_predictions_intraday').select(
            'id, asset_id, direction, horizon_minutes, status, price_at_creation, final_pct_predicted, stop_loss_pct'
        ).eq('status', 'open').eq('direction', 'up').execute().data or []
        horizon_unit = 'minutes'
    else:
        preds = sb.from_('consensus_predictions').select(
            'id, asset_id, direction, horizon_days, status, price_at_creation, final_pct_predicted, stop_loss_pct'
        ).eq('status', 'open').eq('direction', 'up').eq('horizon_days', 1).execute().data or []
        horizon_unit = 'days'

    opened, skipped_reasons, live_errors = 0, defaultdict(int), []
    for p in preds:
        if slots_left_papel <= 0 and slots_left_vivo <= 0:
            break
        pred_id = p['id']
        if pred_id in already_traded_pred_ids:
            continue
        asset = assets_by_id.get(p['asset_id'])
        if not asset:
            skipped_reasons['sin_asset'] += 1
            continue

        venue = _auto_route(asset.get('core_bucket'))
        if not venue:
            skipped_reasons['sin_venue'] += 1
            continue

        horizon_value = p['horizon_minutes'] if source == 'intraday' else p['horizon_days']
        costo_pct = _auto_costo_pct(venue, is_intraday=(source == 'intraday'))
        movimiento_esperado = abs(float(p.get('final_pct_predicted') or 0))
        if costo_pct is None or movimiento_esperado <= costo_pct:
            skipped_reasons['no_supera_costo'] += 1
            continue

        bolsa = scorecard_by_key.get((p['asset_id'], asset['currency'], horizon_unit, int(horizon_value)))
        expectancy = float(bolsa['expectancy_net_pct']) if bolsa and bolsa.get('expectancy_net_pct') is not None else None
        estado = bolsa['estado'] if bolsa else 'sin_datos'
        # Etapa 30 (continuación): override_statistical_gate, a pedido explícito del usuario, salta
        # este chequeo a sabiendas de que hoy no hay ninguna bolsa validada — el resto de los
        # filtros (costo, riesgo) se mantienen intactos. Default false: sin tocar la config, el
        # comportamiento es exactamente el de antes.
        if not cfg.get('override_statistical_gate', False):
            if estado != 'validado' or expectancy is None or expectancy <= 0:
                skipped_reasons['gate_estadistico'] += 1
                continue

        pf = portfolios.get(asset['currency'])
        if not pf:
            skipped_reasons['sin_portfolio'] += 1
            continue
        if pf['id'] in blocked_portfolio_ids:
            skipped_reasons['daily_loss_limit'] += 1
            continue

        entry_price = float(p.get('price_at_creation') or 0)
        if entry_price <= 0:
            skipped_reasons['sin_precio'] += 1
            continue

        # Etapa 30 (continuación): modo vivo, habilitado por venue (live_enabled_byma/_us). Usa
        # efectivo REAL (estado de cuenta de IOL en el momento, no el capital_inicial de papel) y
        # coloca una orden de compra real. Si algo falla (auth, DDJJ requerido, fondos
        # insuficientes, lo que sea) se salta esta entrada sin tocar `auto_trades` — nunca se
        # inserta una fila 'vivo' sin una orden real detrás.
        # Etapa 30 (continuación, 19/08/2026 — hallazgo real con la cuenta real): los CEDEARs
        # (core_bucket='cedear_arg') exigen una declaración jurada (DDJJ) POR ORDEN, no una vez
        # para siempre — se probó aceptándola para las 33 y volvió a pedirla en la siguiente orden.
        # Sin un humano confirmando cada vez, esto rompe el "zero-touch": no hay endpoint para
        # aceptarla sin que alguien lea el texto en el momento. Las acciones argentinas locales
        # (core_bucket='accion_arg_local') NO son CEDEARs y no piden nada de esto (confirmado con
        # GGAL, `validate_order` real). Por eso el modo vivo se restringe a accion_arg_local — los
        # CEDEARs se quedan en papel, no porque no califiquen sino porque la orden real fallaría
        # siempre por la DDJJ.
        live_enabled = (
            (venue == 'BYMA' and cfg.get('live_enabled_byma', False) and asset.get('core_bucket') != 'cedear_arg') or
            (venue == 'US' and cfg.get('live_enabled_us', False))
        )
        if live_enabled and slots_left_vivo <= 0:
            skipped_reasons['max_concurrent_positions_vivo'] += 1
            continue
        if not live_enabled and slots_left_papel <= 0:
            skipped_reasons['max_concurrent_positions_papel'] += 1
            continue

        modo = 'papel'
        iol_buy_order_id = None
        if live_enabled:
            cash = _iol_available_ars_cash() if venue == 'BYMA' else _iol_available_usd_cash()
            if cash is None or cash <= 0:
                skipped_reasons['vivo_sin_efectivo'] += 1
                continue
            # Reparto explícito por MERCADO (Etapa 30 continuación) — nunca comprometer más del
            # tope asignado a Argentina/EEUU, aunque sobre efectivo real de la otra moneda.
            remaining_allocation = max(0.0, live_capital_cap_by_venue.get(venue, 0.0) - live_committed_by_venue[venue])
            budget = min(remaining_allocation, cash)
            monto_invertido = budget * float(cfg.get('live_position_pct', 90)) / 100.0
            cantidad = int(monto_invertido / entry_price)  # unidades enteras — IOL no fracciona acciones/CEDEARs
            if cantidad < 1:
                skipped_reasons['vivo_monto_insuficiente'] += 1
                continue
            mercado = _iol_mercado_for(venue, asset.get('exchange'))
            plazo = _iol_plazo_for(venue)
            try:
                orden = _iol_comprar(mercado, _iol_simbolo_for(asset['ticker']), entry_price, cantidad, plazo=plazo)
                iol_buy_order_id = orden.get('numeroOperacion') or orden.get('numero') or orden.get('id')
                modo = 'vivo'
                monto_invertido = cantidad * entry_price
                live_committed_by_venue[venue] += monto_invertido
                print(f'[motor_ejecucion] VIVO: compra real {asset["ticker"]} x{cantidad} '
                      f'@ {entry_price} ({mercado}/{plazo}) — orden {iol_buy_order_id}', flush=True)
                if iol_buy_order_id is None:
                    # Etapa 30 (25/08/2026): en la primera compra real de la historia del proyecto
                    # (LOMA.BA, 17:39 UTC) ninguno de los 3 nombres de campo probados matcheó — la
                    # compra igual se registró bien (modo='vivo', monto correcto), sólo faltó el
                    # id de la orden para poder rastrearla después. Se loguea la respuesta cruda
                    # completa para ver el shape real en los logs de Render la próxima vez, en vez
                    # de seguir adivinando nombres de campo contra la cuenta real.
                    print(f'[motor_ejecucion] VIVO: orden de {asset["ticker"]} sin id reconocido, '
                          f'respuesta cruda: {orden}', flush=True)
            except Exception as e:
                print(f'[motor_ejecucion] VIVO: compra real de {asset["ticker"]} fallo: {e}', flush=True)
                skipped_reasons['vivo_orden_fallo'] += 1
                if len(live_errors) < 10:  # acotado — esto es diagnóstico, no un log completo
                    live_errors.append(f'{asset["ticker"]} ({mercado}) x{cantidad}: {str(e)[:200]}')
                continue
        else:
            monto_invertido = float(pf['capital_inicial']) * float(cfg.get('max_position_pct_capital', 2)) / 100.0
            cantidad = monto_invertido / entry_price

        stop_loss_usado = float(p['stop_loss_pct']) if p.get('stop_loss_pct') is not None else DEFAULT_STOP_LOSS_PCT

        sb.from_('auto_trades').insert({
            'portfolio_id': pf['id'], 'asset_id': p['asset_id'], 'prediction_type': source,
            'daily_prediction_id': pred_id if source == 'daily' else None,
            'intraday_prediction_id': pred_id if source == 'intraday' else None,
            'direction': 'up', 'venue': venue, 'modo': modo,
            'horizon_value': horizon_value, 'horizon_unit': horizon_unit,
            'bolsa_estado_al_entrar': estado, 'bolsa_expectancy_net_at_entry': expectancy,
            'monto_invertido': monto_invertido, 'cantidad': cantidad,
            'stop_loss_sugerido_pct': p.get('stop_loss_pct'), 'stop_loss_usado_pct': stop_loss_usado,
            'take_profit_pct': None, 'entry_price': entry_price,
            'iol_buy_order_id': iol_buy_order_id,
        }).execute()
        opened += 1
        if modo == 'vivo':
            slots_left_vivo -= 1
        else:
            slots_left_papel -= 1

    return {'ok': True, 'opened': opened, 'evaluated': len(preds), 'skipped': dict(skipped_reasons),
            'live_errors': live_errors}


def _get_latest_prices(sb, asset_ids):
    """Último precio conocido por activo: `indicators_intraday.price_close` (más fresco, cada
    ~10 min en horario de mercado) con fallback a `price_history.close` (cierre diario) para
    activos sin fila intradiaria reciente — cubre tanto auto_trades intradiarios como diarios."""
    # Etapa 30 (continuación, a pedido explícito del usuario — cuidado con el egress de Supabase,
    # ya se pasaron de eso antes): ambas consultas de acá abajo NO tenían límite ni ventana de
    # tiempo — `indicators_intraday` guarda 14 días de historial (~una fila cada 10 min por activo)
    # y `price_history` años. Sin acotar, cada corrida traía TODO ese historial de vuelta sólo para
    # quedarse con la fila más reciente. Ahora se acota a una ventana chica (2h intradía, 5 días
    # diario) + un límite duro — de sobra para encontrar el último precio, sin traer de más.
    asset_ids = list(asset_ids)
    if not asset_ids:
        return {}
    prices = {}
    recent_cutoff = (datetime.utcnow() - timedelta(hours=2)).isoformat()
    ii = sb.from_('indicators_intraday').select('asset_id, price_close, calculated_at') \
        .in_('asset_id', asset_ids).gte('calculated_at', recent_cutoff) \
        .order('calculated_at', desc=True).limit(len(asset_ids) * 6).execute().data or []
    for row in ii:
        if row['asset_id'] not in prices and row.get('price_close') is not None:
            prices[row['asset_id']] = float(row['price_close'])
    missing = [a for a in asset_ids if a not in prices]
    if missing:
        recent_days_cutoff = (datetime.utcnow() - timedelta(days=5)).date().isoformat()
        ph = sb.from_('price_history').select('asset_id, close, trade_date') \
            .in_('asset_id', missing).gte('trade_date', recent_days_cutoff) \
            .order('trade_date', desc=True).limit(len(missing) * 5).execute().data or []
        for row in ph:
            if row['asset_id'] not in prices and row.get('close') is not None:
                prices[row['asset_id']] = float(row['close'])
    return prices


def _run_motor_salidas(sb):
    """Cierra `auto_trades` abiertas por stop-loss, take-profit, o cierre de la predicción que las
    originó. pnl_pct/pnl_monto quedan netos de la comisión IOL del venue (mismo principio que el
    filtro de costo: la magnitud bruta no es el número que importa)."""
    open_trades = sb.from_('auto_trades').select(
        'id, asset_id, venue, modo, prediction_type, daily_prediction_id, intraday_prediction_id, '
        'entry_price, cantidad, monto_invertido, stop_loss_usado_pct, take_profit_pct'
    ).eq('status', 'abierta').execute().data or []
    if not open_trades:
        return {'ok': True, 'closed': 0}

    assets_by_id = {a['id']: a for a in (
        sb.from_('assets').select('id, ticker, exchange').in_(
            'id', list({t['asset_id'] for t in open_trades})).execute().data or [])}

    daily_pred_ids = [t['daily_prediction_id'] for t in open_trades if t.get('daily_prediction_id')]
    intraday_pred_ids = [t['intraday_prediction_id'] for t in open_trades if t.get('intraday_prediction_id')]
    daily_status = {p['id']: p for p in (
        sb.from_('consensus_predictions').select('id, status, actual_final_price')
        .in_('id', daily_pred_ids).execute().data or [])} if daily_pred_ids else {}
    # Etapa 30 (continuación, a pedido explícito del usuario): NO esperar a que `juez-intraday`
    # (que corre cada 5 min, en su propio cron, fuera de esta etapa) marque `status='closed'` — acá
    # se compara `target_time` directo contra la hora actual, así el cierre por vencimiento no
    # arrastra el retraso de ese otro job encima del propio de este motor.
    intraday_status = {p['id']: p for p in (
        sb.from_('consensus_predictions_intraday').select('id, status, actual_price, target_time')
        .in_('id', intraday_pred_ids).execute().data or [])} if intraday_pred_ids else {}

    latest_prices = _get_latest_prices(sb, {t['asset_id'] for t in open_trades})

    closed = 0
    for t in open_trades:
        entry_price = float(t['entry_price'])
        current_price = latest_prices.get(t['asset_id'])
        gross_pct_now = ((current_price - entry_price) / entry_price * 100.0) if current_price else None

        exit_price, close_status = None, None
        if t.get('take_profit_pct') is not None and gross_pct_now is not None and gross_pct_now >= t['take_profit_pct']:
            exit_price, close_status = current_price, 'cerrada_por_take_profit'
        elif gross_pct_now is not None and gross_pct_now <= t['stop_loss_usado_pct']:
            exit_price, close_status = current_price, 'cerrada_por_stop'
        elif t['prediction_type'] == 'intraday':
            # Etapa 30 (continuación): vencimiento contra `target_time` directo (exacto, no
            # depende del cron de juez-intraday) — pedido explícito del usuario de que el cierre
            # coincida con el momento real del horizonte, no con el de otro job.
            pred = intraday_status.get(t['intraday_prediction_id'])
            target_time_raw = pred.get('target_time') if pred else None
            if target_time_raw:
                target_time = datetime.fromisoformat(target_time_raw.replace('Z', '+00:00')).replace(tzinfo=None)
                if datetime.utcnow() >= target_time:
                    exit_price = current_price or (float(pred.get('actual_price')) if pred.get('actual_price') is not None else entry_price)
                    close_status = 'cerrada_normal'
        else:
            pred = daily_status.get(t['daily_prediction_id'])
            if pred and pred.get('status') == 'closed':
                exit_price = float(pred.get('actual_final_price') or current_price or entry_price)
                close_status = 'cerrada_normal'

        if not close_status:
            continue

        # Etapa 30 (continuación): modo vivo vende de verdad antes de marcar la fila como cerrada.
        # Si la venta real falla, la fila queda abierta (se reintenta el próximo ciclo) — nunca se
        # marca 'cerrada' sin una orden real detrás, para que auto_trades no mienta sobre lo que
        # pasó con la plata real.
        iol_sell_order_id = None
        if t.get('modo') == 'vivo':
            asset = assets_by_id.get(t['asset_id'])
            cantidad_vender = int(t['cantidad'] or 0)
            if not asset or cantidad_vender < 1:
                print(f'[motor_ejecucion] VIVO: no se pudo vender trade {t["id"]} '
                      f'(asset o cantidad inválida)', flush=True)
                continue
            mercado = _iol_mercado_for(t['venue'], asset.get('exchange'))
            plazo = _iol_plazo_for(t['venue'])
            try:
                orden = _iol_vender(mercado, _iol_simbolo_for(asset['ticker']), exit_price, cantidad_vender, plazo=plazo)
                iol_sell_order_id = orden.get('numeroOperacion') or orden.get('numero') or orden.get('id')
                print(f'[motor_ejecucion] VIVO: venta real {asset["ticker"]} x{cantidad_vender} '
                      f'@ {exit_price} ({close_status}) — orden {iol_sell_order_id}', flush=True)
            except Exception as e:
                print(f'[motor_ejecucion] VIVO: venta real de trade {t["id"]} fallo: {e} '
                      f'— queda abierta, se reintenta', flush=True)
                continue

        costo_pct = _auto_costo_pct(t['venue'], is_intraday=(t['prediction_type'] == 'intraday')) or 0.0
        gross_pct = (exit_price - entry_price) / entry_price * 100.0
        pnl_pct = gross_pct - costo_pct
        pnl_monto = pnl_pct / 100.0 * float(t['monto_invertido'])

        sb.from_('auto_trades').update({
            'status': close_status, 'exit_price': exit_price,
            'pnl_pct': pnl_pct, 'pnl_monto': pnl_monto,
            'closed_at': datetime.utcnow().isoformat(),
            'iol_sell_order_id': iol_sell_order_id,
        }).eq('id', t['id']).execute()
        closed += 1

    return {'ok': True, 'closed': closed, 'evaluated': len(open_trades)}


@app.route('/api/motor_ejecucion', methods=['POST', 'OPTIONS'])
def motor_ejecucion():
    """Etapa 30 — motor de trading automático (hoy sólo papel, ver REDISENO/ETAPA-30). Disparado
    por cron poco después de cada tanda de predicciones nuevas.
    body: {"source": "intraday"|"daily", "phase": "entradas"|"salidas"|"ambas" (default "ambas")}.
    Evalúa salidas antes que entradas cuando phase="ambas", para liberar cupo de posiciones
    concurrentes en la misma corrida."""
    if request.method == 'OPTIONS':
        return '', 200
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    body = request.get_json() or {}
    source = body.get('source', 'intraday')
    phase = body.get('phase', 'ambas')
    if source not in ('intraday', 'daily'):
        return jsonify({'ok': False, 'error': 'source debe ser intraday|daily'}), 400
    if phase not in ('entradas', 'salidas', 'ambas'):
        return jsonify({'ok': False, 'error': 'phase debe ser entradas|salidas|ambas'}), 400
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        result = {}
        if phase in ('salidas', 'ambas'):
            result['salidas'] = _run_motor_salidas(sb)
        if phase in ('entradas', 'ambas'):
            result['entradas'] = _run_motor_entradas(sb, source)
        return jsonify({'ok': True, **result})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500


@app.route('/api/iol_test', methods=['GET'])
def iol_test():
    """Diagnóstico: confirma que IOL_USERNAME/IOL_PASSWORD autentican de verdad contra la API real
    de IOL, y expone la respuesta cruda de estadocuenta para poder ajustar
    _iol_available_ars_cash() si su parseo defensivo no matchea el shape real (nunca se probó
    contra una cuenta real hasta tener credenciales) — no hace ninguna operación, sólo lee."""
    if not _check_secret():
        return jsonify({'ok': False, 'error': 'forbidden'}), 403
    try:
        raw = _iol_estado_cuenta()
        parsed_ars_cash = _iol_available_ars_cash()
        return jsonify({'ok': True, 'auth': 'ok', 'estado_cuenta_raw': raw, 'parsed_ars_cash': parsed_ars_cash})
    except Exception as e:
        return jsonify({'ok': False, 'auth': 'failed', 'error': str(e)}), 500


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
