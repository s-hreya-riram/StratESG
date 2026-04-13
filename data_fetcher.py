# data/data_fetcher.py
import time
import logging

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alpaca imports (live trading only)
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from config import STOCK_CLIENT, CRYPTO_CLIENT
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def to_utc(ts) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Red-flag / award events
# ---------------------------------------------------------------------------
@dataclass
class RedFlagEvent:
    ticker: str
    date:   str
    reason: str
    mode:   str   # "layoff" | "ceo_departure" | "protest"


CASE_STUDY_EVENTS = [
    # (ticker, date,         label,              color)
    ("NVDA", "2021-01-15", "BPTW #2\n+GPTW #12", "darkgreen"),
    ("NVDA", "2022-01-15", "BPTW #1\n+GPTW #5",  "darkgreen"),
    ("NVDA", "2023-01-15", "BPTW #5\nGPTW #6",   "darkgreen"),
    ("NVDA", "2024-01-15", "BPTW #2\nGPTW #3",   "darkgreen"),
    ("NVDA", "2025-01-15", "BPTW #4\nGPTW #5",   "darkgreen"),
    ("NVDA", "2026-01-15", "BPTW #3\nGPTW #6",   "darkgreen"),
    ("AMZN", "2022-04-01", "ALU union\nvote victory", "red"),
    ("AMZN", "2022-11-16", "10k layoffs",         "darkred"),
    ("AMZN", "2023-01-18", "18k layoffs",         "darkred"),
    ("AMZN", "2025-10-28", "14k layoffs",         "darkred"),
    ("AMZN", "2026-01-28", "16k layoffs",         "darkred"),
]

GLASSDOOR_RATINGS = {
    2016: (4.2, 3.7), 2017: (4.3, 3.6), 2018: (4.4, 3.6),
    2019: (4.5, 3.5), 2020: (4.5, 3.4), 2021: (4.6, 3.4),
    2022: (4.6, 3.3), 2023: (4.5, 3.2), 2024: (4.4, 3.3), 2025: (4.4, 3.4),
}

BPTW_APPEARANCES = {
    "NVDA": [2018, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    "AMZN": [],
}


# ---------------------------------------------------------------------------
# Data fetching — Yahoo Finance (backtesting)
# ---------------------------------------------------------------------------
def fetch_asset_data(symbol_mapping, is_backtesting, start_date=None, end_date=None):
    """Fetch OHLCV data via Yahoo Finance."""
    if is_backtesting:
        assert start_date is not None and end_date is not None, \
            "Backtesting requires start_date and end_date"
    else:
        if start_date is None:
            start_date = pd.Timestamp.now("UTC") - pd.Timedelta(days=90)
        if end_date is None:
            end_date = pd.Timestamp.now("UTC")

    all_data = {}
    for idx, (alpaca_symbol, yahoo_symbol) in enumerate(symbol_mapping):
        if idx > 0:
            time.sleep(0.02)
        try:
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError("Yahoo returned an empty dataframe")
            df.index = pd.to_datetime(df.index).tz_convert("UTC")
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            logger.info(f"Fetched Yahoo data for {yahoo_symbol} ({len(df)} rows)")
            all_data[yahoo_symbol] = df
        except Exception as exc:
            logger.error(f"Yahoo fetch failed for {yahoo_symbol}: {exc}")

    return all_data


# ---------------------------------------------------------------------------
# StratESG universe & strategy configuration
# ---------------------------------------------------------------------------

# (ticker, GICS sector, ESG rating)
# Ratings: AAA -> conviction 1.0 / AA -> conviction 0.5
STRATESG_UNIVERSE = [
    ("NVDA", "InformationTechnology",  "AAA"), # Nvidia
    ("LLY",  "HealthCare",             "AA"),  # Eli Lilly and Company
    ("MSFT", "InformationTechnology",  "AA"),  # Microsoft
    ("EXP",  "RealEstate",             "AA"),  # eXP Realty
    ("RLI",  "Financials",             "AA"),  # RLI
    ("SBGSY","Industrials",            "AA"),  # Schneider Electric
    ("ASML", "InformationTechnology",  "AA"),  # ASML
    ("BAH",  "Industrials",            "AA"),  # Booz Allen Hamilton
    ("ADSK", "InformationTechnology",  "AA"),  # Autodesk
    ("CRM",  "InformationTechnology",  "AAA"), # Salesforce
    ("SAIL", "InformationTechnology",  "AA"),  # SailPoint
    ("ADBE", "InformationTechnology",  "AAA"), # Adobe
    ("MSI",  "InformationTechnology",  "AA"),  # Motorola Solutions
    ("PG",   "ConsumerStaples",        "AA"),  # Procter & Gamble
    ("MRK",  "HealthCare",             "AAA"), # Merck & Co.
    ("EPAM", "InformationTechnology",  "AA"),  # EPAM Systems
    ("HLT",  "ConsumerDiscretionary",  "AA"),  # Hilton Worldwide
    ("SYF",  "Financials",             "AA"),  # Synchrony Financial
    ("CSCO", "InformationTechnology",  "AA"),  # Cisco Systems
    ("AXP",  "Financials",             "AA"),  # American Express
    ("ACN",  "InformationTechnology",  "AA"),  # Accenture
    ("MAR",  "ConsumerDiscretionary",  "AA"),  # Marriott International
    ("PNFP", "Financials",             "AA"),  # Pinnacle Financial Partners
    ("CDNS", "InformationTechnology",  "AA"),  # Cadence Design Systems
    ("DAL",  "Industrials",            "AA"),  # Delta Air Lines
    ("PGR",  "Financials",             "AA"),  # Progressive Corporation
    ("IHG",  "ConsumerDiscretionary",  "AA"),  # InterContinental Hotels Group
    ("CPT",  "InformationTechnology",  "AA"),  # CDW
    ("CAKE", "ConsumerDiscretionary",  "AA"),  # Cheesecake Factory
    ("DOW",  "Materials",              "AA"),  # Dow Inc.
    ("CMCSA","CommunicationServices",  "AA"),  # Comcast
    ("RKT",  "ConsumerDiscretionary",  "AA"),  # Rocket Companies
    ("NOW",  "InformationTechnology",  "AA"),  # ServiceNow
    ("MET",  "Financials",             "AA"),  # MetLife
    ("CACC", "Financials",             "AA"),  # Credit Acceptance Corporation
    ("COF",  "Financials",             "AA"),  # Capital One
    ("ABBV", "HealthCare",             "AA"),  # AbbVie
    ("Z",    "CommunicationServices",  "AA"),  # Zillow
    ("VRTX", "HealthCare",             "AA"),  # Vertex Pharmaceuticals
    ("TPH",  "RealEstate",             "AA"),  # Tri Pointe Homes
    ("ELV",  "HealthCare",             "AA"),  # Elevance Health
    ("PHM",  "RealEstate",             "AA"),  # PulteGroup
    ("RHI",  "Industrials",            "AA"),  # Robert Half International
    ("BAC",  "Financials",             "AA"),  # Bank of America
    ("SYK",  "Industrials",            "AA"),  # Stryker Corporation
    ("TGT",  "ConsumerStaples",        "AA"),  # Target Corporation
    ("ALLY", "Financials",             "AA"),  # Ally Financial
    ("BOX",  "InformationTechnology",  "AA"),  # Box
    ("CRWD", "InformationTechnology",  "AA"),  # CrowdStrike
    ("MA",   "Financials",             "AA"),  # Mastercard
    ("FAF",  "Financials",             "AA"),  # First American Financial
    ("H",    "ConsumerDiscretionary",  "AA"),  # Hyatt Hotels
    ("KMX",  "ConsumerDiscretionary",  "AA"),  # CarMax
    ("UTHR", "HealthCare",             "AA"),  # United Therapeutics
    ("V",    "Financials",             "AA"),  # Visa
    ("INTU", "InformationTechnology",  "AA"),  # Intuit
    ("WK",   "InformationTechnology",  "AA"),  # Workiva
    ("HPE",  "InformationTechnology",  "AA"),  # Hewlett Packard Enterprise
    ("HPQ",  "InformationTechnology",  "AA"),  # HP Inc.
    ("TEAM", "InformationTechnology",  "AA"),  # Atlassian
    ("WMT",  "ConsumerStaples",        "AA"),  # Walmart
]

CONVICTION          = {"AAA": 1.0, "AA": 0.5}
SECTOR_CAP          = 0.33          # 33% max per sector
CONTROVERSY_PENALTY = 0.50          # weight halved on controversy date
MIN_WEIGHT          = 0.0         # minimum weight floor per non-penalised ticker
REBALANCE_FREQ      = "MS"          # month-start rebalancing
LOOKBACK_DAYS       = 90
MOMENTUM_WINDOW     = 30            # ~1-month momentum lookback (trading days)
MOMENTUM_BLEND      = 0.75          # 75% momentum, 25% inv-vol

# Controversies within the 2025-2026 investment window (>5% workforce confirmed):
STRATESG_CONTROVERSIES = [
    # (ticker,  date,          reason,                                pct_affected)
    ("CRWD", "2025-05-07", "5% workforce reduction (500 layoffs)",    0.05),
    ("BAH",  "2025-05-23", "7% workforce reduction (2.5k layoffs)",   0.07),
    ("PG",   "2025-06-05", "6% workforce reduction (7k layoffs)",     0.06),
    ("MSFT", "2025-07-02", "7% workforce reduction (15k layoffs)",    0.07),
    ("MRK",  "2025-07-31", "8% workforce reduction (6k layoffs)",     0.08),
    ("CRM",  "2025-09-02", "5.5% workforce reduction (4k layoffs)",   0.055),
    ("ADSK", "2026-01-22", "7% workforce reduction (1k layoffs)",     0.07),
]


# ---------------------------------------------------------------------------
# StratESG engine
# ---------------------------------------------------------------------------

def _compute_inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    """Inverse-volatility weighting with conviction multipliers."""
    tickers = returns.columns.tolist()
    if returns.shape[0] < 2:
        return pd.Series(1.0 / len(tickers), index=tickers)

    vol = returns.std().replace(0, np.nan)
    inv_vol = (1.0 / vol).fillna((1.0 / vol).min())

    if inv_vol.isna().all():
        return pd.Series(1.0 / len(tickers), index=tickers)

    for ticker, _, rating in STRATESG_UNIVERSE:
        if ticker in inv_vol.index:
            inv_vol[ticker] *= CONVICTION[rating]

    total = inv_vol.sum()
    return inv_vol / total if total > 0 else pd.Series(1.0 / len(tickers), index=tickers)


def _compute_momentum_weights(returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional momentum, conviction-scaled. Negative scores zeroed out."""
    tickers = returns.columns.tolist()
    if returns.shape[0] < 2:
        return pd.Series(1.0 / len(tickers), index=tickers)

    cum_ret = (1 + returns).prod() - 1

    for ticker, _, rating in STRATESG_UNIVERSE:
        if ticker in cum_ret.index:
            cum_ret[ticker] *= CONVICTION[rating]

    scores = cum_ret.clip(lower=1e-4)
    total  = scores.sum()
    if total <= 0:
        return pd.Series(1.0 / len(tickers), index=tickers)
    return scores / total


def _apply_sector_cap(weights: pd.Series) -> pd.Series:
    """
    Enforce SECTOR_CAP with proportional scaling inside over-cap sectors.
    Freed excess redistributed pro-rata to under-cap sectors.
    Unplaceable excess stays as implicit cash (weights may sum < 1).
    """
    sector_map = {t: s for t, s, _ in STRATESG_UNIVERSE}
    w = weights.copy().clip(lower=0)

    for _ in range(40):
        sec_total: dict[str, float] = {}
        for t in w.index:
            s = sector_map.get(t, "Other")
            sec_total[s] = sec_total.get(s, 0.0) + w[t]

        excess = 0.0
        for s, stot in sec_total.items():
            if stot > SECTOR_CAP + 1e-9:
                scale = SECTOR_CAP / stot
                for t in w.index:
                    if sector_map.get(t, "Other") == s:
                        excess += w[t] * (1.0 - scale)
                        w[t]   *= scale

        if excess < 1e-9:
            break

        sec_total = {}
        for t in w.index:
            s = sector_map.get(t, "Other")
            sec_total[s] = sec_total.get(s, 0.0) + w[t]

        eligible  = [
            t for t in w.index
            if sec_total.get(sector_map.get(t, "Other"), 0.0) < SECTOR_CAP - 1e-9
        ]
        if not eligible:
            break

        elg_sum = sum(w[t] for t in eligible)
        if elg_sum <= 0:
            break
        for t in eligible:
            w[t] += excess * (w[t] / elg_sum)

    return w


def _apply_min_weight_floor(weights: pd.Series, controversy_activated: set) -> pd.Series:
    """Lift non-penalised tickers below MIN_WEIGHT, funding from over-floor tickers."""
    w        = weights.copy()
    eligible = [t for t in w.index if t not in controversy_activated and w[t] > 1e-6]

    if not eligible:
        return w

    for _ in range(20):
        below     = [t for t in eligible if w[t] < MIN_WEIGHT - 1e-9]
        if not below:
            break
        deficit   = sum(MIN_WEIGHT - w[t] for t in below)
        above     = [t for t in eligible if w[t] > MIN_WEIGHT + 1e-9]
        above_sum = sum(w[t] for t in above)
        if above_sum <= 0:
            break
        for t in below:
            w[t] = MIN_WEIGHT
        for t in above:
            w[t] -= deficit * (w[t] / above_sum)

    return w


def run_stratesg(data: dict, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, dict]:
    """
    Simulate StratESG from start to end.
    Returns (daily NAV series, final ticker weights dict).
    """
    tickers = [t for t, _, _ in STRATESG_UNIVERSE if t in data]

    lookback_start = start - pd.Timedelta(days=LOOKBACK_DAYS)

    prices_full = (
        pd.DataFrame({t: data[t]["Close"] for t in tickers})
        .sort_index()
        .loc[lookback_start:end]
        .dropna(how="all")
    )
    prices = prices_full.loc[start:end]
    if prices.empty:
        raise ValueError("No price data in the investment window.")

    returns_full = prices_full.pct_change().fillna(0)

    def first_trading_day_on_or_after(cal_date):
        mask = prices.index >= cal_date
        return prices.index[mask][0] if mask.any() else None

    cal_rebal   = pd.date_range(start, end, freq=REBALANCE_FREQ, tz="UTC")
    rebal_dates = set(filter(None, (first_trading_day_on_or_after(d) for d in cal_rebal)))

    controversy_map: dict[str, pd.Timestamp] = {}
    for ticker, date_str, _, _ in STRATESG_CONTROVERSIES:
        if ticker not in tickers:
            continue
        reaction = first_trading_day_on_or_after(to_utc(date_str))
        if reaction is not None and start <= reaction <= end:
            controversy_map[ticker] = reaction

    controversy_activated: set = set()

    nav      = pd.Series(index=prices.index, dtype=float)
    holdings = pd.Series(0.0, index=tickers)
    cash     = 0.0

    current_weights = pd.Series(0.0, index=tickers)

    def _build_weights(as_of_date):
        window_start   = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS)
        window_returns = returns_full.loc[window_start:as_of_date]

        inv_vol_w  = _compute_inverse_vol_weights(window_returns).reindex(tickers).fillna(0)
        momentum_w = _compute_momentum_weights(
            window_returns.iloc[-MOMENTUM_WINDOW:]
        ).reindex(tickers).fillna(0)

        momentum_w = momentum_w.clip(lower=1e-6)
        inv_vol_w  = inv_vol_w.clip(lower=1e-6)
        blended = (momentum_w ** MOMENTUM_BLEND) * (inv_vol_w ** (1 - MOMENTUM_BLEND))
        total = blended.sum()
        if total > 0:
            blended = blended / total

        for ticker in controversy_activated:
            if ticker in blended.index:
                blended[ticker] *= CONTROVERSY_PENALTY

        # Redistribute freed weight to non-penalised tickers
        penalised_total     = sum(blended[t] for t in controversy_activated if t in blended.index)
        non_penalised       = [t for t in blended.index if t not in controversy_activated]
        non_penalised_total = blended[non_penalised].sum()
        if non_penalised_total > 0:
            scale = (1.0 - penalised_total) / non_penalised_total
            blended[non_penalised] *= scale

        blended = _apply_sector_cap(blended)
        blended = _apply_min_weight_floor(blended, controversy_activated)
        return blended

    # Re-apply _apply_sector_cap at execution time inside _rebalance so
    # that price drift between rebalances cannot cause a sector cap breach when
    # shares are actually purchased.    
    def _rebalance(cur_nav, px):
        nonlocal cash
        safe_weights = _apply_sector_cap(current_weights.copy())

        # Renormalise after sector cap so freed weight doesn't become cash
        total = safe_weights.sum()
        if total > 0:
            safe_weights = safe_weights / total

        h = pd.Series(0.0, index=tickers)
        for t in tickers:
            price = px.get(t, np.nan)
            wt    = safe_weights.get(t, 0.0)
            if not np.isnan(price) and price > 0 and wt > 0:
                h[t] = (wt * cur_nav) / price

        actual_invested = float((h * px).sum())
        cash = cur_nav - actual_invested
        return h

# ========== MAIN SIMULATION LOOP ==========
    for i, date in enumerate(prices.index):
        px = prices.loc[date]

        # (a) Activate any controversies for today FIRST, before building weights,
        #     so a controversy falling on a rebalance day gets folded into the single
        #     monthly rebalance — one set of trades, not two.
        triggered = [
            t for t, rd in controversy_map.items()
            if rd == date and t not in controversy_activated
        ]
        for t in triggered:
            print(f"\n  *** Controversy: {t} on {date.date()} — halving weight ***")
            controversy_activated.add(t)

        # (b) Regular monthly rebalance (weights now already reflect any new controversies)
        if date in rebal_dates:
            current_weights = _build_weights(date)
            cur_nav = 1.0 if i == 0 else float((holdings * px).sum() + cash)
            holdings = _rebalance(cur_nav, px)

            invested = current_weights.sum()
            print(f"\n  Rebalance on {date.date()}:")
            for t in sorted(tickers, key=lambda x: -current_weights[x]):
                if current_weights[t] > 1e-4:
                    marker = " (controversy)" if t in controversy_activated else ""
                    print(f"    {t}: {current_weights[t]:.4f}{marker}")
            if invested < 0.9999:
                print(f"    cash: {1.0 - invested:.4f}")

        # (c) Controversy mid-month rebalance — only fires when today is NOT already
        #     a scheduled rebalance day (avoids double rebalance)
        elif triggered:
            current_weights = _build_weights(date)
            cur_nav = float((holdings * px).sum() + cash)
            holdings = _rebalance(cur_nav, px)

            invested = current_weights.sum()
            print(f"  Updated weights after controversy:")
            for t in sorted(tickers, key=lambda x: -current_weights[x]):
                if current_weights[t] > 1e-4:
                    marker = " (controversy)" if t in controversy_activated else ""
                    print(f"    {t}: {current_weights[t]:.4f}{marker}")
            if invested < 0.9999:
                print(f"    cash: {1.0 - invested:.4f}")

        # (d) Day-0 initialisation when first day is not a rebalance date
        if i == 0 and date not in rebal_dates:
            current_weights = _build_weights(date)
            holdings = _rebalance(1.0, px)

        # (e) Mark-to-market NAV
        nav.iloc[i] = float((holdings * px).sum() + cash)

    final_px    = prices.iloc[-1]
    # Zero out micro-positions before final cash sweep to avoid fractional share noise
    min_final_weight = 0.005
    final_portfolio_value = float((holdings * final_px).sum() + cash)
    for t in tickers:
        if holdings[t] > 0 and (holdings[t] * final_px.get(t, 0)) / final_portfolio_value < min_final_weight:
            cash += holdings[t] * final_px.get(t, 0)
            holdings[t] = 0.0
    # After the main simulation loop, before computing final weights
    if cash > 0.01 * float((holdings * final_px).sum() + cash):
        holdings = _rebalance(float((holdings * final_px).sum() + cash), final_px)
    final_value = float((holdings * final_px).sum() + cash)
    ticker_weights = {
        t: float(holdings[t] * final_px.get(t, 0)) / final_value
        for t in tickers
        if holdings[t] > 1e-8
    }
    invested       = sum(ticker_weights.values())

    print(f"\nFinal weights on {prices.index[-1].date()}:")
    for t, w in sorted(ticker_weights.items(), key=lambda x: -x[1]):
        if w > 1e-4:
            marker = " (controversy)" if t in controversy_activated else ""
            print(f"  {t}: {w:.4f}{marker}")
    if invested < 0.9999:
        print(f"  cash: {1.0 - invested:.4f}")

    return nav, ticker_weights


# ---------------------------------------------------------------------------
# Data fetching — Alpaca (live / paper trading)
# ---------------------------------------------------------------------------
def fetch_historical_data(symbols, start_date, end_date=None,
                          timeframe=TimeFrame.Day, feed=None):
    """Fetch OHLCV data via Alpaca for stock symbols."""
    print(f"Fetching historical data for: {symbols}")

    start_time = to_utc(start_date)
    end_time   = to_utc(end_date) if end_date else None

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start_time,
        end=end_time,
        feed=feed,
    )
    bars = STOCK_CLIENT.get_stock_bars(request)

    data_dict = {}
    for symbol in symbols:
        if symbol in bars.data and bars.data[symbol]:
            b  = bars.data[symbol]
            df = pd.DataFrame(
                {
                    "Open":   [x.open   for x in b],
                    "High":   [x.high   for x in b],
                    "Low":    [x.low    for x in b],
                    "Close":  [x.close  for x in b],
                    "Volume": [x.volume for x in b],
                },
                index=pd.to_datetime([x.timestamp for x in b]),
            )
            data_dict[symbol] = df.sort_index()
            print(f"  {symbol}: {len(df)} bars")
        else:
            print(f"  {symbol}: no data returned")

    return data_dict


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _nearest_trading_date(df: pd.DataFrame, target: pd.Timestamp) -> pd.Timestamp | None:
    target_date = target.date()
    matches = df.index[df.index.normalize() == pd.Timestamp(target_date, tz="UTC")]
    if not matches.empty:
        return matches[0]
    for offset in range(1, 6):
        candidate = pd.Timestamp(target_date, tz="UTC") + pd.Timedelta(days=offset)
        matches   = df.index[df.index.normalize() == candidate]
        if not matches.empty:
            return matches[0]
    return None


def _build_event_map(data: dict) -> dict:
    event_map = {symbol: [] for symbol in data}
    for ticker, date_str, label, color in CASE_STUDY_EVENTS:
        if ticker not in data:
            continue
        event_ts = to_utc(date_str)
        matched  = _nearest_trading_date(data[ticker], event_ts)
        if matched is not None:
            event_map[ticker].append((matched, label, color))
        else:
            df = data[ticker]
            print(
                f"  Warning: no trading date found near {date_str} for {ticker} "
                f"(data range: {df.index[0].date()} to {df.index[-1].date()})"
            )
    return event_map


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------
LINE_COLORS   = {"NVDA": "#1f77b4", "AMZN": "#ff7f0e"}
EVENT_LABELS  = {
    "darkgreen": "BPTW / GPTW",
    "red":       "Union / protest",
    "darkred":   "Mass layoff",
}
INVEST_START  = pd.Timestamp("2025-05-01", tz="UTC")
INVEST_END    = pd.Timestamp("2026-03-31", tz="UTC")
INVEST_AMOUNT = 10_000   # USD


# ---------------------------------------------------------------------------
# Plot 1 — closing prices with event markers
# ---------------------------------------------------------------------------
def _annotate_events(ax, df: pd.DataFrame, events: list, y_min: float, y_max: float):
    color_proxies: dict[str, mlines.Line2D] = {}
    span = y_max - y_min

    for i, (event_ts, label, color) in enumerate(sorted(events, key=lambda e: e[0])):
        close_price = df.loc[event_ts, "Close"]

        ax.axvline(x=event_ts, color=color, linewidth=0.8, linestyle="--",
                   alpha=0.2, zorder=1)
        ax.scatter(event_ts, close_price, color=color, zorder=6, s=55,
                   edgecolors="white", linewidths=0.7)

        level  = i % 3
        y_frac = 0.95 - level * 0.14
        y_data = y_min + y_frac * span

        ax.annotate(
            label,
            xy=(event_ts, close_price),
            xytext=(event_ts, y_data),
            xycoords="data", textcoords="data",
            fontsize=7, color=color, va="top", ha="center",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.35, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color,
                      alpha=0.88, lw=0.7),
            clip_on=False, zorder=7,
        )

        if color not in color_proxies:
            color_proxies[color] = mlines.Line2D(
                [], [], color=color, marker="o", linestyle="None",
                markersize=5.5, label=EVENT_LABELS.get(color, label.split("\n")[0]),
            )

    return color_proxies


def plot_closing_prices(data: dict, event_map: dict,
                        output_path: str = "output/price_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    fig.suptitle(
        "Leader (NVDA) / Laggard (AMZN) - Stock Price with Key Events (Feb 2021 - Feb 2026)",
        fontsize=13, fontweight="bold", y=0.95,
    )

    for ax, symbol in zip(axes, list(data.keys())):
        df         = data[symbol]
        line_color = LINE_COLORS.get(symbol, "steelblue")
        events     = event_map.get(symbol, [])

        ax.plot(df.index, df["Close"], color=line_color, linewidth=1.8)

        y_min, y_max = 0, 275
        color_proxies = _annotate_events(ax, df, events, y_min, y_max)

        ax.set_title(f"{symbol}", fontsize=12, fontweight="bold", color=line_color, pad=6)
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Date")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(handles=list(color_proxies.values()),
                  fontsize=8, loc="lower right", framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 -- $10k invested, value over window
# ---------------------------------------------------------------------------
def plot_investment_growth(data: dict,
                           output_path: str = "output/investment_growth.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    for symbol, df in data.items():
        line_color = LINE_COLORS.get(symbol, "steelblue")
        subset     = df.loc[df.index >= INVEST_START]
        if subset.empty:
            print(f"  Warning: no data for {symbol} from {INVEST_START.date()}")
            continue

        growth    = (subset["Close"] / subset["Close"].iloc[0]) * INVEST_AMOUNT
        final_val = growth.iloc[-1]
        delta_pct = (final_val / INVEST_AMOUNT - 1) * 100
        sign      = "+" if delta_pct >= 0 else ""

        ax.plot(subset.index, growth, color=line_color, linewidth=2, label=symbol)
        ax.annotate(
            f"${final_val:,.0f}  ({sign}{delta_pct:.1f}%)",
            xy=(growth.index[-1], final_val),
            xytext=(8, 0), textcoords="offset points",
            fontsize=9, color=line_color, va="center",
            fontweight="bold", clip_on=False,
        )

    ax.axhline(INVEST_AMOUNT, color="grey", linewidth=0.9, linestyle=":", alpha=0.7)
    ax.set_title(f"Growth of ${INVEST_AMOUNT:,} Invested — Feb 2025 to Jan 2026",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 3 -- Glassdoor ratings over time
# ---------------------------------------------------------------------------
def plot_glassdoor(output_path: str = "output/glassdoor_ratings.png"):
    years     = sorted(GLASSDOOR_RATINGS.keys())
    nvda_vals = [GLASSDOOR_RATINGS[y][0] for y in years]
    amzn_vals = [GLASSDOOR_RATINGS[y][1] for y in years]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(years, nvda_vals, color=LINE_COLORS["NVDA"], linewidth=2,
            marker="o", markersize=6, label="NVDA")
    ax.plot(years, amzn_vals, color=LINE_COLORS["AMZN"], linewidth=2,
            marker="o", markersize=6, label="AMZN")
    ax.fill_between(years, nvda_vals, amzn_vals, alpha=0.08, color="#888888")

    for year, nv, amz in zip(years, nvda_vals, amzn_vals):
        ax.text(year, nv + 0.05, f"{nv}", ha="center", fontsize=7.5,
                color=LINE_COLORS["NVDA"], fontweight="bold")
        ax.text(year, amz - 0.07, f"{amz}", ha="center", fontsize=7.5,
                color=LINE_COLORS["AMZN"], fontweight="bold", va="top")

    ax.set_title("Glassdoor Ratings Over Time", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Rating (out of 5)")
    ax.set_xlabel("Year")
    ax.set_ylim(2.9, 5.1)
    ax.set_xticks(years)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 4 -- StratESG vs HAPI vs SPY
# ---------------------------------------------------------------------------
def plot_stratesg(data: dict, output_path: str = "output/stratesg_comparison.png"):
    start = INVEST_START
    end   = INVEST_END

    esg_tickers = [t for t, _, _ in STRATESG_UNIVERSE]
    missing = [t for t in esg_tickers if t not in data]
    if missing:
        lookback_start = start - pd.Timedelta(days=90)
        extra = fetch_asset_data(
            [(t, t) for t in missing], is_backtesting=True,
            start_date=lookback_start, end_date=end,
        )
        data.update(extra)

    esg_nav, _ = run_stratesg(data, start, end)
    esg_growth  = esg_nav * INVEST_AMOUNT

    benchmarks   = {"SPY": "SPY", "HAPI": "HAPI"}
    bmark_growth: dict[str, pd.Series] = {}
    for label, ticker in benchmarks.items():
        if ticker not in data:
            fetched = fetch_asset_data(
                [(ticker, ticker)], is_backtesting=True,
                start_date=start, end_date=end,
            )
            if ticker in fetched:
                data[ticker] = fetched[ticker]
        if ticker in data:
            subset = data[ticker]["Close"].loc[start:]
            if not subset.empty:
                bmark_growth[label] = (subset / subset.iloc[0]) * INVEST_AMOUNT

    palette = {"StratESG": "#2ecc71", "SPY": "#3498db", "HAPI": "#e67e22"}

    fig, ax = plt.subplots(figsize=(10, 5))
    end_annotations = []

    def _plot_series(series, label, color):
        final = series.dropna().iloc[-1]
        delta = (final / INVEST_AMOUNT - 1) * 100
        sign  = "+" if delta >= 0 else ""
        ax.plot(series.index, series, color=color, linewidth=2, label=label)
        end_annotations.append((final, label, color, f"${final:,.0f}  ({sign}{delta:.1f}%)"))

    _plot_series(esg_growth, "StratESG", palette["StratESG"])
    for label, series in bmark_growth.items():
        _plot_series(series, label, palette[label])

    end_annotations.sort(key=lambda x: x[0], reverse=True)
    last_x = esg_growth.dropna().index[-1]
    for rank, (final_val, label, color, text) in enumerate(end_annotations):
        ax.annotate(
            text,
            xy=(last_x, final_val),
            xytext=(10, rank * -6), textcoords="offset points",
            fontsize=8.5, color=color, va="center",
            fontweight="bold", clip_on=False,
        )

    CONTROVERSY_COLORS = [
        "#c0392b",  # strong red
        "#d35400",  # burnt orange
        "#f1c40f",  # yellow
        "#2ecc71",  # green
        "#3498db",  # blue
        "#5b2c6f",  # indigo
        "#8e44ad",  # violet
    ]

    controversy_in_range = [
        (ticker, date_str, reason)
        for ticker, date_str, reason, _ in STRATESG_CONTROVERSIES
        if start <= to_utc(date_str) <= end
    ]

    controversy_handles = []
    if controversy_in_range:
        for i, (ticker, date_str, reason) in enumerate(controversy_in_range):
            color    = CONTROVERSY_COLORS[i % len(CONTROVERSY_COLORS)]
            event_ts = to_utc(date_str)
            nearby   = esg_growth.index[esg_growth.index >= event_ts]
            if nearby.empty:
                continue
            plot_ts  = nearby[0]
            plot_val = esg_growth.loc[plot_ts]

            ax.axvline(plot_ts, color=color, linewidth=1.1,
                       linestyle="--", alpha=0.55, zorder=2)
            ax.scatter(plot_ts, plot_val, color=color, zorder=6,
                       s=60, edgecolors="white", linewidths=0.8)
            controversy_handles.append(mlines.Line2D(
                [], [], color=color, marker="o", linestyle="--",
                markersize=6, linewidth=1.1, label=f"{ticker}: {reason}",
            ))

    ax.axhline(INVEST_AMOUNT, color="#AAAAAA", linewidth=0.9, linestyle=":")
    ax.set_title("StratESG vs SPY — growth on $10k invested (May 1, 2025 to Mar 31, 2026)",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_xlabel("Date")

    strategy_legend = ax.legend(fontsize=9, loc="upper left",
                                framealpha=0.9, edgecolor="#CCCCCC")
    if controversy_handles:
        ax.add_artist(strategy_legend)
        ax.legend(handles=controversy_handles, fontsize=8,
                  loc="lower right", framealpha=0.9,
                  edgecolor="#CCCCCC", title="Controversies", title_fontsize=8.5)

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()

    # ── Performance metrics table ─────────────────────────────────────────────
    all_metrics = []

    # StratESG NAV starts at 1.0 by construction
    all_metrics.append(compute_metrics(esg_nav, "StratESG"))

    for label, series in bmark_growth.items():
        # bmark_growth is in dollar terms — normalise to 1.0
        bmark_nav = series / series.iloc[0]
        all_metrics.append(compute_metrics(bmark_nav, label))

    metrics_df = pd.DataFrame(all_metrics).set_index("Label")

    print("\n" + "=" * 72)
    print("  PERFORMANCE METRICS  (Apr 2025 – Mar 2026, RF = 4.5% p.a.)")
    print("=" * 72)
    print(metrics_df.to_string())
    print("=" * 72 + "\n")

    # Optional: also save to CSV for further analysis
    metrics_df.to_csv("output/performance_metrics.csv")
    print("Saved -> output/performance_metrics.csv")


def plot_stratesg_allocation_breakdown(
    data: dict,
    output_path: str = "output/stratesg_allocation_breakdown.png",
):
    """
    Two-part allocation chart:

    Top    — donut chart with one slice per sector, labeled with sector name + %.
    Bottom — grid of horizontal bar charts, one panel per sector.
             Each bar is a ticker ranked by weight, labeled left (ticker) and
             right (exact %). Controversy-penalised tickers are coloured in
             the controversy accent colour and marked with a dagger (†).

    Layout auto-scales: adding tickers or sectors requires no manual tuning.
    """
    import math
    import matplotlib.pyplot as plt
    import numpy as np

    # ── Run strategy ──────────────────────────────────────────────────────────
    start = INVEST_START
    end   = INVEST_END
    _, ticker_weights = run_stratesg(data, start, end)

    # ── Build sector → [(ticker, weight)] lookup ──────────────────────────────
    sector_map      = {t: s for t, s, _ in STRATESG_UNIVERSE}
    controversy_set = {t for t, _, _, _ in STRATESG_CONTROVERSIES}

    sectors_dict: dict[str, list[tuple[str, float]]] = {}
    for t, _, _ in STRATESG_UNIVERSE:
        if t not in ticker_weights:
            continue
        s = sector_map[t]
        sectors_dict.setdefault(s, []).append((t, ticker_weights[t]))
    for s in sectors_dict:
        sectors_dict[s].sort(key=lambda x: -x[1])

    # Only keep tickers with positive weight
    sectors_dict = {
        s: [(t, w) for t, w in pairs if w > 0]
        for s, pairs in sectors_dict.items()
        if any(w > 0 for _, w in pairs)
    }

    sector_totals  = {s: sum(w for _, w in pairs) for s, pairs in sectors_dict.items()}
    
    # Filter before sorting so sorted_sectors only contains sectors that will actually render
    sectors_dict = {
        s: [(t, w) for t, w in pairs if w >= 0.005]
        for s, pairs in sectors_dict.items()
    }
    sectors_dict = {s: pairs for s, pairs in sectors_dict.items() if pairs}

    # Recompute totals after filtering so donut sizes stay consistent with bar panels
    sector_totals = {s: sum(w for _, w in pairs) for s, pairs in sectors_dict.items()}

    sorted_sectors = sorted(sector_totals.items(), key=lambda x: -x[1])

    invested    = sum(ticker_weights.values())
    cash_weight = max(0.0, 1.0 - invested)

    # ── Design tokens ─────────────────────────────────────────────────────────
    BG           = "#FAFAF8"
    PANEL_BG     = "#FFFFFF"
    GRID_CLR     = "#EEEEEC"
    TEXT_DARK    = "#1A1A2E"
    TEXT_MID     = "#555566"
    CONTROV_CLR  = "#C0392B"
    CASH_CLR     = "#B8B8B8"

    SECTOR_PALETTE = [
        "#2E86AB", "#E67E22", "#27AE60", "#8E44AD", "#E74C3C",
        "#16A085", "#F39C12", "#2C3E50", "#D35400", "#1ABC9C",
        "#8E44AD", "#27AE60",
    ]
    sector_colors = {
        s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)]
        for i, (s, _) in enumerate(sorted_sectors)
    }

    # ── Layout constants (tune these to adjust density) ───────────────────────
    N_COLS       = 3
    BAR_HEIGHT   = 0.42   # bar thickness in axes-data units
    SLOT_IN      = 0.46   # inches of vertical space per ticker row
    HEADER_IN    = 0.40   # inches reserved for panel title
    PANEL_PAD_IN = 0.28   # vertical gap between panel rows
    DONUT_H_IN   = 4.6
    FIG_W_IN     = 15.0

    # Per-panel height scales with ticker count — uniform bar size regardless
    def panel_h_in(sector):
        return len(sectors_dict[sector]) * SLOT_IN + HEADER_IN

    n_rows = math.ceil(len(sorted_sectors) / N_COLS)

    # Each grid row is as tall as its tallest panel (+ gap)
    def row_max_h_in(row_idx):
        row_sectors = [s for i, (s, _) in enumerate(sorted_sectors) if i // N_COLS == row_idx]
        return max(panel_h_in(s) for s in row_sectors)

    row_heights_in = [row_max_h_in(r) + PANEL_PAD_IN for r in range(n_rows)]
    fig_h_in = DONUT_H_IN + sum(row_heights_in) + 0.4

    def to_frac(inches):
        """Convert inches to figure-height fraction."""
        return inches / fig_h_in

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(FIG_W_IN, fig_h_in), facecolor=BG)

    # ── Donut ─────────────────────────────────────────────────────────────────
    donut_frac = to_frac(DONUT_H_IN)
    ax_donut = fig.add_axes([0.22, 1.0 - donut_frac + 0.01, 0.56, donut_frac - 0.04])
    ax_donut.set_facecolor(BG)

    donut_labels = [s for s, _ in sorted_sectors]
    donut_sizes  = [v for _, v in sorted_sectors]
    donut_colors = [sector_colors[s] for s in donut_labels]
    if cash_weight > 1e-4:
        donut_labels.append("Cash")
        donut_sizes.append(cash_weight)
        donut_colors.append(CASH_CLR)

    wedges, _ = ax_donut.pie(
        donut_sizes,
        colors=donut_colors,
        startangle=90,
        wedgeprops=dict(width=0.50, edgecolor=BG, linewidth=2.2),
    )

    # Centre label
    date_label = end.strftime("%b %Y") if hasattr(end, "strftime") else str(end)
    ax_donut.text(0,  0.06, "StratESG", ha="center", va="center",
                  fontsize=11, fontweight="bold", color=TEXT_DARK)
    ax_donut.text(0, -0.18, date_label, ha="center", va="center",
                  fontsize=8, color=TEXT_MID)

    # Wedge leader-line labels
    #SHORTEN = {
    #    "Information": "Info", "Consumer": "Cons.", "Communication": "Comm."
    #}
    for wedge, label, size in zip(wedges, donut_labels, donut_sizes):
        angle_r = np.deg2rad((wedge.theta2 + wedge.theta1) / 2.0)
        x_mid,  y_mid  = 0.78 * np.cos(angle_r), 0.78 * np.sin(angle_r)
        x_text, y_text = 1.22 * np.cos(angle_r), 1.22 * np.sin(angle_r)
        short = label
        #for long, abbr in SHORTEN.items():
        #    short = short.replace(long, abbr)
        ax_donut.annotate(
            f"{short}\n{size:.1%}",
            xy=(x_mid, y_mid), xytext=(x_text, y_text),
            fontsize=7.5, ha="left" if x_text >= 0 else "right",
            va="center", color=TEXT_DARK,
            arrowprops=dict(arrowstyle="-", color="#CCCCCC", lw=0.9),
        )

    date_full = end.strftime("%B %-d, %Y") if hasattr(end, "strftime") else str(end)
    ax_donut.set_title(
        f"StratESG — Sector & Asset Allocation   |   {date_full}",
        fontsize=13, fontweight="bold", color=TEXT_DARK, pad=16,
    )

    # ── Bar panels ────────────────────────────────────────────────────────────
    L_MARGIN = 0.055
    R_MARGIN = 0.975
    T_START  = 1.0 - donut_frac - 0.015   # figure-fraction just below donut
    col_w    = (R_MARGIN - L_MARGIN) / N_COLS
    col_gap  = 0.025
    panel_w  = col_w - col_gap

    all_weights  = [w for pairs in sectors_dict.values() for _, w in pairs]
    global_xmax  = max(all_weights) * 1.40 if all_weights else 0.01

    for idx, (sector, sec_total) in enumerate(sorted_sectors):
        row = idx // N_COLS
        col = idx  % N_COLS

        tickers = sectors_dict[sector]
        # drop zero-weight tickers just in case (shouldn't be any, but just to be safe)
        tickers = [(t, w) for t, w in tickers if w > 0.0]
        n       = len(tickers)

        rows_above = sum(row_heights_in[:row])
        l = L_MARGIN + col * col_w + col_gap / 2
        # Top-align: pin the panel's top edge to the row's top edge (minus half the gap),
        # then extend downward by its own height — short panels flush with the row top,
        # eliminating the awkward gap above panels with fewer tickers.
        row_top = T_START - to_frac(rows_above) - to_frac(PANEL_PAD_IN / 2)
        h = to_frac(panel_h_in(sector))
        b = row_top - h

        ax = fig.add_axes([l, b, panel_w, h])
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)

        color      = sector_colors[sector]
        labels     = [t for t, _ in tickers]
        values     = [w for _, w in tickers]
        is_cont    = [t in controversy_set for t in labels]
        bar_colors = [CONTROV_CLR if c else color for c in is_cont]

        y_pos = list(range(n - 1, -1, -1))  # top-to-bottom order

        bars = ax.barh(y_pos, values, color=bar_colors,
                       height=BAR_HEIGHT, edgecolor="none")

        # Subtle vertical grid
        for xv in [0.02, 0.04, 0.06, 0.08]:
            if xv < global_xmax:
                ax.axvline(xv, color=GRID_CLR, lw=0.6, zorder=0)

        # Ticker labels (left y-axis)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(
            [f"{t} †" if c else t for t, c in zip(labels, is_cont)],
            fontsize=6.5, color=TEXT_DARK,
        )
        for ticklabel, cont in zip(ax.get_yticklabels(), is_cont):
            if cont:
                ticklabel.set_color(CONTROV_CLR)
                ticklabel.set_fontweight("semibold")

        # Percentage labels (right of bar)
        for bar, val in zip(bars, values):
            ax.text(
                val + global_xmax * 0.032,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}",
                va="center", ha="left", fontsize=7, color=TEXT_MID,
            )

        ax.set_xlim(0, global_xmax)
        ax.set_xticks([])
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_title(
            f"{sector}  ·  {sec_total:.1%}",
            fontsize=8.5, fontweight="bold",
            color=color, pad=5, loc="left",
        )

    # ── Controversy footnote ──────────────────────────────────────────────────
    if controversy_set & set(ticker_weights.keys()):
        fig.text(
            0.5, 0.003,
            "† Weight halved due to confirmed workforce controversy (>5%)",
            ha="center", fontsize=7.5, color=CONTROV_CLR, style="italic",
        )

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved -> {output_path}")
    plt.show()


def compute_metrics(nav: pd.Series, label: str, risk_free_rate: float = 0.045) -> dict:
    """
    Compute annualised performance metrics from a daily NAV series (starting at 1.0).
    risk_free_rate: annualised, e.g. 0.045 for 4.5% (approx 2025 US T-bill rate)
    """
    returns = nav.pct_change().dropna()
    n_days  = len(returns)
    ann     = 252  # trading days

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return     = nav.iloc[-1] / nav.iloc[0] - 1
    ann_return       = (1 + total_return) ** (ann / n_days) - 1

    # ── Risk metrics ─────────────────────────────────────────────────────────
    ann_vol          = returns.std() * np.sqrt(ann)
    daily_rf         = (1 + risk_free_rate) ** (1 / ann) - 1
    excess_returns   = returns - daily_rf
    sharpe           = (excess_returns.mean() / returns.std()) * np.sqrt(ann)

    # Sortino (downside deviation only)
    downside         = returns[returns < daily_rf] - daily_rf
    downside_std     = np.sqrt((downside ** 2).mean()) * np.sqrt(ann)
    sortino          = (ann_return - risk_free_rate) / downside_std if downside_std > 0 else np.nan

    # Max drawdown
    cumulative       = (1 + returns).cumprod()
    rolling_max      = cumulative.cummax()
    drawdown_series  = (cumulative - rolling_max) / rolling_max
    max_drawdown     = drawdown_series.min()

    # Calmar ratio (ann return / abs(max drawdown))
    calmar           = ann_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

    # Win rate
    win_rate         = (returns > 0).mean()

    return {
        "Label":           label,
        "Total Return":    f"{total_return:.2%}",
        "Ann. Return":     f"{ann_return:.2%}",
        "Ann. Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Sortino Ratio":   f"{sortino:.3f}",
        "Max Drawdown":    f"{max_drawdown:.2%}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Win Rate":        f"{win_rate:.2%}",
        "# Trading Days":  n_days,
    }

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    SYMBOL_MAPPING = [("NVDA", "NVDA"), ("AMZN", "AMZN")]
    START_DATE = pd.Timestamp("2021-02-01", tz="UTC")
    END_DATE   = pd.Timestamp("2026-02-01", tz="UTC")

    print("Fetching NVDA/AMZN data ...")
    data = fetch_asset_data(
        SYMBOL_MAPPING, is_backtesting=True,
        start_date=START_DATE, end_date=END_DATE,
    )
    for symbol, df in data.items():
        print(f"  {symbol}: {len(df)} rows  ({df.index[0].date()} to {df.index[-1].date()})")

    event_map = _build_event_map(data)

    plot_closing_prices(data, event_map)
    plot_investment_growth(data)
    plot_glassdoor()
    plot_stratesg(data)
    plot_stratesg_allocation_breakdown(data)