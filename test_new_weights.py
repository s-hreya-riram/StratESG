#!/usr/bin/env python3
"""Test the new risk-adjusted return weighting strategy."""

import pandas as pd
import numpy as np
from data_fetcher import (
    INVEST_START, INVEST_END, INVEST_AMOUNT, STRATESG_UNIVERSE,
    fetch_asset_data, run_stratesg
)

print("=" * 70)
print("TESTING NEW RISK-ADJUSTED RETURN WEIGHTING STRATEGY")
print("=" * 70)

# Fetch StratESG universe data
esg_tickers = [t for t, _, _ in STRATESG_UNIVERSE]
lookback_start = INVEST_START - pd.Timedelta(days=90)

print(f"\nFetching {len(esg_tickers)} StratESG tickers...")
data = fetch_asset_data(
    [(t, t) for t in esg_tickers],
    is_backtesting=True,
    start_date=lookback_start,
    end_date=INVEST_END,
)
print(f"✅ Fetched {len(data)} tickers")

# Run strategy
print(f"\n📊 Running StratESG simulation from {INVEST_START.date()} to {INVEST_END.date()}...")
nav, weights = run_stratesg(data, INVEST_START, INVEST_END)

# Calculate performance
final_value = nav.iloc[-1] * INVEST_AMOUNT
total_return = (final_value / INVEST_AMOUNT) - 1
days = len(nav)
annual_return = (final_value / INVEST_AMOUNT) ** (252 / days) - 1

daily_returns = nav.pct_change().dropna()
annual_vol = daily_returns.std() * np.sqrt(252)
risk_free_rate = 0.04
sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0

print(f"\n{'='*70}")
print("📈 RESULTS - StratESG (Risk-Adjusted Return Weights)")
print(f"{'='*70}")
print(f"Final Portfolio Value:     ${final_value:>12,.2f}")
print(f"Total Return:              {total_return:>12.2%}")
print(f"Annualized Return:         {annual_return:>12.2%}")
print(f"Annualized Volatility:     {annual_vol:>12.2%}")
print(f"Sharpe Ratio:              {sharpe:>12.3f}")

print(f"\n📊 Final Portfolio Weights:")
print(f"{'Ticker':<10} {'Weight':>10} {'Cumulative':>12}")
print("-" * 32)
cumulative = 0
for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
    cumulative += weight
    print(f"{ticker:<10} {weight:>9.2%}  {cumulative:>11.2%}")

print(f"\n✅ Test complete!")
