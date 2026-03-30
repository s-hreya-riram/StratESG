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
    ("NVDA", "2026-01-15", "BPTW #3",             "darkgreen"),
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
    ("NVDA", "InformationTechnology",  "AAA"),
    ("NOW",  "InformationTechnology",  "AAA"),
    ("EPAM", "InformationTechnology",  "AA"),
    ("GOOGL","CommunicationServices",  "AA"),
    ("MSI",  "InformationTechnology",  "AA"),
    ("BSX",  "HealthCare",             "AA"),
    ("GE",   "Industrials",            "AA"),
    ("PGR",  "Financials",             "AAA"),
    ("ISRG", "HealthCare",             "AAA"),
    ("MRK",  "HealthCare",             "AAA"),
    ("CRM",  "InformationTechnology",  "AAA"),
    ("LLY",  "HealthCare",             "AA"),
]

CONVICTION          = {"AAA": 1.0, "AA": 0.5}
SECTOR_CAP          = 0.33          # 33% max per sector (no individual asset cap)
CONTROVERSY_PENALTY = 0.50          # weight halved on controversy date
MIN_WEIGHT          = 0.02          # minimum weight floor per non-penalised ticker
REBALANCE_FREQ      = "MS"          # month-start rebalancing
LOOKBACK_DAYS       = 90            # explicit constant used everywhere
MOMENTUM_WINDOW     = 63            # ~3-month momentum lookback (trading days)
MOMENTUM_BLEND      = 0.50          # 50% inv-vol, 50% momentum

# Only controversies within the 2025-2026 investment window (>5% workforce confirmed):
#   CRM: ~5.5% cut (~4k of 73k), announced Sep 2 2025
#   MRK:  8.0% cut (~6k of 75k), announced Jul 31 2025
STRATESG_CONTROVERSIES = [
    # (ticker,  date,          reason,                              pct_affected)
    ("CRM",  "2025-09-02", "4k AI-driven support cuts (~5.5%)",   0.055),
    ("MRK",  "2025-07-31", "6k restructuring layoffs (~8%)",        0.080),
]


# ---------------------------------------------------------------------------
# StratESG engine
# ---------------------------------------------------------------------------

def _compute_inverse_vol_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Inverse-volatility weighting with conviction multipliers.

        weight_i = (1 / σ_i) × conviction_i,  normalised to sum to 1.

    Falls back to equal-weight when all volatilities are zero.
    """
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


# momentum signal helper
def _compute_momentum_weights(returns: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional momentum: rank tickers by cumulative return over the window,
    then score proportionally (rank / sum_of_ranks).  Negative-momentum tickers
    receive a score of zero so they don't drag the portfolio.
    """
    tickers = returns.columns.tolist()
    if returns.shape[0] < 2:
        return pd.Series(1.0 / len(tickers), index=tickers)

    cum_ret = (1 + returns).prod() - 1   # cumulative return over window

    # Apply conviction multipliers to momentum scores as well
    for ticker, _, rating in STRATESG_UNIVERSE:
        if ticker in cum_ret.index:
            cum_ret[ticker] *= CONVICTION[rating]

    # Zero out negative momentum
    scores = cum_ret.clip(lower=0)
    total = scores.sum()
    if total <= 0:
        # All negative — fall back to equal weight
        return pd.Series(1.0 / len(tickers), index=tickers)
    return scores / total


def _apply_sector_cap(weights: pd.Series) -> pd.Series:
    """
    Enforce SECTOR_CAP (35%) with no individual asset cap.

    When a sector total exceeds SECTOR_CAP, each asset within that sector is
    scaled down proportionally (preserving relative weights inside the sector)
    until the sector total equals SECTOR_CAP.  The freed excess is redistributed
    pro-rata to assets whose sector is still under the cap.

    If no eligible recipients exist (every sector is already at the cap),
    the excess is left uninvested — the returned weights sum to < 1 and the
    gap is treated as a cash holding by the caller.
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
                        w[t] *= scale

        if excess < 1e-9:
            break

        sec_total = {}
        for t in w.index:
            s = sector_map.get(t, "Other")
            sec_total[s] = sec_total.get(s, 0.0) + w[t]

        eligible = [
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


# minimum weight floor helper
def _apply_min_weight_floor(weights: pd.Series, controversy_activated: set) -> pd.Series:
    """
    Enforce MIN_WEIGHT floor on all tickers that haven't been controversy-penalised
    and that currently hold a non-zero weight.  Any ticker below the floor is lifted
    to MIN_WEIGHT; the excess cost is taken pro-rata from over-floor tickers.

    Controversy-penalised tickers are excluded so the penalty isn't accidentally
    neutralised by the floor.
    """
    w = weights.copy()
    eligible = [t for t in w.index if t not in controversy_activated and w[t] > 1e-6]

    if not eligible:
        return w

    for _ in range(20):
        below  = [t for t in eligible if w[t] < MIN_WEIGHT - 1e-9]
        if not below:
            break
        deficit = sum(MIN_WEIGHT - w[t] for t in below)
        above   = [t for t in eligible if w[t] > MIN_WEIGHT + 1e-9]
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
    Returns tuple of (daily NAV series, final ticker weights dict).

    Strategy rules (updated):
      - Monthly rebalancing on the first trading day of each month.
      - CHANGE 3: Weights = blend of inverse-volatility and 3-month momentum,
        each computed over a strict 90-day rolling window ending on the rebalance date.
      - CHANGE 4: 50/50 blend of inv-vol and momentum signals.
      - CHANGE 2: Minimum 2% floor applied to non-penalised tickers after sector cap.
      - CHANGE 1: Sector cap raised to 35%.
      - Controversy: affected ticker's weight is halved permanently (no recovery).
      - NAV is computed day-by-day from share holdings plus cash balance.
        Remaining cash after full allocation flows through naturally.
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

    controversy_activated = set()

    nav      = pd.Series(index=prices.index, dtype=float)
    holdings = pd.Series(0.0, index=tickers)
    cash     = 0.0

    current_weights = pd.Series(0.0, index=tickers)

    def _build_weights(as_of_date):
        # strict 90-day rolling window (not full history)
        window_start = as_of_date - pd.Timedelta(days=LOOKBACK_DAYS)
        window_returns = returns_full.loc[window_start:as_of_date]

        # blend inv-vol and momentum signals
        inv_vol_w  = _compute_inverse_vol_weights(window_returns).reindex(tickers).fillna(0)
        momentum_w = _compute_momentum_weights(
            window_returns.iloc[-MOMENTUM_WINDOW:]   # momentum over last 63 trading days
        ).reindex(tickers).fillna(0)

        blended = MOMENTUM_BLEND * momentum_w + (1 - MOMENTUM_BLEND) * inv_vol_w

        # Apply permanent controversy reduction
        for ticker in controversy_activated:
            if ticker in blended.index:
                blended[ticker] *= CONTROVERSY_PENALTY

        # Normalise before applying caps
        total = blended.sum()
        if total > 0:
            blended = blended / total

        # Sector cap then minimum weight floor
        blended = _apply_sector_cap(blended)
        # CHANGE 2: apply floor after sector cap
        blended = _apply_min_weight_floor(blended, controversy_activated)

        return blended

    def _rebalance(cur_nav, px):
        """
        Convert current_weights into share holdings.
        Any residual (from sector cap) stays as cash — no redistribution attempted,
        which avoids the over-investment bug from the original implementation.
        """
        nonlocal cash
        h = pd.Series(0.0, index=tickers)
        for t in tickers:
            price = px.get(t, np.nan)
            wt    = current_weights.get(t, 0.0)
            if not np.isnan(price) and price > 0 and wt > 0:
                h[t] = (wt * cur_nav) / price

        # Cash is simply whatever wasn't invested — no redistribution
        actual_invested = float((h * px).sum())
        cash = cur_nav - actual_invested
        return h

    # ========== MAIN SIMULATION LOOP ==========
    for i, date in enumerate(prices.index):
        px = prices.loc[date]

        # (a) Regular monthly rebalance
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

        # (b) Day-0 initialisation when the first day is not a rebalance date
        if i == 0 and date not in rebal_dates:
            current_weights = _build_weights(date)
            holdings = _rebalance(1.0, px)

        # (c) Controversy reaction — trigger rebalance on controversy date
        triggered = [
            t for t, rd in controversy_map.items()
            if rd == date and t not in controversy_activated
        ]
        if triggered:
            for t in triggered:
                print(f"\n  *** Controversy: {t} on {date.date()} — halving weight ***")
                controversy_activated.add(t)

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

        # (d) Mark-to-market NAV
        nav.iloc[i] = float((holdings * px).sum() + cash)

    ticker_weights = current_weights.to_dict()
    invested = sum(ticker_weights.values())

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
            b = bars.data[symbol]
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
        matches = df.index[df.index.normalize() == candidate]
        if not matches.empty:
            return matches[0]
    return None


def _build_event_map(data: dict) -> dict:
    """Group CASE_STUDY_EVENTS by symbol, resolving each date against the df index."""
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
    "darkgreen":  "BPTW / GPTW",
    "red":        "Union / protest",
    "darkred":    "Mass layoff",
}
INVEST_START  = pd.Timestamp("2025-02-01", tz="UTC")
INVEST_END    = pd.Timestamp("2026-02-01", tz="UTC")
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
            xycoords="data",
            textcoords="data",
            fontsize=7,
            color=color,
            va="top",
            ha="center",
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.35, lw=0.7),
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=color,
                      alpha=0.88, lw=0.7),
            clip_on=False,
            zorder=7,
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
    fig.suptitle("Leader (NVDA) / Laggard (AMZN) - Stock Price with Key Events (Feb 2021 - Feb 2026)",
                 fontsize=13, fontweight="bold", y=0.95)

    for ax, symbol in zip(axes, list(data.keys())):
        df         = data[symbol]
        line_color = LINE_COLORS.get(symbol, "steelblue")
        events     = event_map.get(symbol, [])

        ax.plot(df.index, df["Close"], color=line_color, linewidth=1.8)

        y_min, y_max = 0, 275
        color_proxies = _annotate_events(ax, df, events, y_min, y_max)

        ax.set_title(f"{symbol}", fontsize=12, fontweight="bold",
                     color=line_color, pad=6)
        ax.set_ylabel("Price (USD)")
        ax.set_xlabel("Date")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(handles=list(color_proxies.values()),
                  fontsize=8, loc="lower right", framealpha=0.9,
                  edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 2 -- $10k invested in February 2025, value today
# ---------------------------------------------------------------------------
def plot_investment_growth(data: dict,
                           output_path: str = "output/investment_growth.png"):
    fig, ax = plt.subplots(figsize=(10, 5))

    for symbol, df in data.items():
        line_color = LINE_COLORS.get(symbol, "steelblue")
        subset = df.loc[df.index >= INVEST_START]
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

    ax.set_title(f" Growth of ${INVEST_AMOUNT:,} Invested in February 2025 - January 2026",
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
# Plot 4 -- StratESG vs HAPI vs SPY ($10k invested Feb 2025)
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
    esg_growth = esg_nav * INVEST_AMOUNT

    benchmarks = {"SPY": "SPY", "HAPI": "HAPI"}
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

    palette = {
        "StratESG": "#2ecc71",
        "SPY":      "#3498db",
        "HAPI":     "#e67e22",
    }

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
        y_offset = rank * -4
        ax.annotate(
            text,
            xy=(last_x, final_val),
            xytext=(10, y_offset), textcoords="offset points",
            fontsize=8.5, color=color, va="center",
            fontweight="bold", clip_on=False,
        )

    # Distinct color per controversy event
    CONTROVERSY_COLORS = [
        "#c0392b",   # strong red    — first event
        "#d35400",   # burnt orange  — second event
        "#7d3c98",   # purple        — third (if ever added)
        "#1a5276",   # dark blue     — fourth
    ]

    controversy_in_range = [
        (ticker, date_str, reason)
        for ticker, date_str, reason, _ in STRATESG_CONTROVERSIES
        if start <= to_utc(date_str) <= end
    ]

    controversy_handles = []   # collected for the legend
    if controversy_in_range:
        for i, (ticker, date_str, reason) in enumerate(controversy_in_range):
            color    = CONTROVERSY_COLORS[i % len(CONTROVERSY_COLORS)]
            event_ts = to_utc(date_str)
            nearby   = esg_growth.index[esg_growth.index >= event_ts]
            if nearby.empty:
                continue

            plot_ts  = nearby[0]
            plot_val = esg_growth.loc[plot_ts]

            # Vertical dashed line spanning the full chart height
            ax.axvline(plot_ts, color=color, linewidth=1.1,
                       linestyle="--", alpha=0.55, zorder=2)

            # Dot on the StratESG line
            ax.scatter(plot_ts, plot_val, color=color, zorder=6,
                       s=60, edgecolors="white", linewidths=0.8)

            # Build legend handle: dot + label
            legend_label = f"{ticker}: {reason}"
            handle = mlines.Line2D(
                [], [], color=color,
                marker="o", linestyle="--",
                markersize=6, linewidth=1.1,
                label=legend_label
            )
            controversy_handles.append(handle)

    ax.axhline(INVEST_AMOUNT, color="#AAAAAA", linewidth=0.9, linestyle=":")

    ax.set_title("StratESG vs SPY vs HAPI- 1 year growth on $10k invested in Feb 2025",
                 fontsize=12, fontweight="bold", pad=8)
    ax.set_ylabel("Portfolio Value (USD)")
    ax.set_xlabel("Date")

    # Two-group legend: strategy lines (upper-left) + controversies (lower-left)
    strategy_legend = ax.legend(fontsize=9, loc="upper left",
                                framealpha=0.9, edgecolor="#CCCCCC")
    if controversy_handles:
        ax.add_artist(strategy_legend)   # keep first legend visible
        ax.legend(handles=controversy_handles, fontsize=8,
                  loc="lower right", framealpha=0.9,
                  edgecolor="#CCCCCC", title="Controversies",
                  title_fontsize=8.5)

    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Plot 5 - StratESG allocation breakdown (pie chart by sector with ticker breakdown)
# ---------------------------------------------------------------------------
def plot_stratesg_allocation_breakdown(data: dict, output_path: str = "output/stratesg_allocation_breakdown.png"):
    start = INVEST_START
    end   = INVEST_END
    _, ticker_weights = run_stratesg(data, start, end)

    print("Final ticker weights:")
    for ticker, weight in sorted(ticker_weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker}: {weight:.4f}")

    invested = sum(ticker_weights.values())
    cash_weight = 1.0 - invested
    if cash_weight > 1e-4:
        print(f"  cash:  {cash_weight:.4f}")

    sectors_dict = {}
    for ticker, _, _ in STRATESG_UNIVERSE:
        if ticker in ticker_weights:
            sector = next((s for t, s, _ in STRATESG_UNIVERSE if t == ticker), "Other")
            sectors_dict.setdefault(sector, {})[ticker] = ticker_weights[ticker]

    sector_totals = {s: sum(v.values()) for s, v in sectors_dict.items()}
    sorted_sectors = sorted(sector_totals.items(), key=lambda x: -x[1])

    sector_base_colors = plt.cm.Set2(np.linspace(0, 1, len(sorted_sectors) + (1 if cash_weight > 1e-4 else 0)))
    sector_to_color = {s: sector_base_colors[i] for i, (s, _) in enumerate(sorted_sectors)}

    labels, sizes, colors = [], [], []

    for sector, _ in sorted_sectors:
        for ticker, weight in sorted(sectors_dict[sector].items(), key=lambda x: -x[1]):
            labels.append(ticker)
            sizes.append(weight)
            colors.append(sector_to_color[sector])

    if cash_weight > 1e-4:
        labels.append("Cash")
        sizes.append(cash_weight)
        colors.append(sector_base_colors[len(sorted_sectors)])

    fig, ax = plt.subplots(figsize=(13, 8))

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"fontsize": 9, "weight": "bold"},
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )

    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(8)

    legend_elements = [
        mlines.Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=sector_to_color[sector], markersize=10,
                      label=f"{sector} ({sector_totals[sector]:.1%})")
        for sector, _ in sorted_sectors
    ]
    if cash_weight > 1e-4:
        legend_elements.append(
            mlines.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=sector_base_colors[len(sorted_sectors)],
                          markersize=10, label=f"Cash ({cash_weight:.1%})")
        )

    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 0.5),
              fontsize=10, framealpha=0.95, title="Sectors", title_fontsize=11)

    ax.set_title("StratESG Portfolio Allocation by Asset (Jan 2026)",
                 fontsize=13, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved -> {output_path}")
    plt.show()


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

    plot_closing_prices(data, event_map)          # -> output/price_comparison.png
    plot_investment_growth(data)                  # -> output/investment_growth.png
    plot_glassdoor()                              # -> output/glassdoor_ratings.png
    plot_stratesg(data)                           # -> output/stratesg_comparison.png
    plot_stratesg_allocation_breakdown(data)      # -> output/stratesg_allocation_breakdown.png