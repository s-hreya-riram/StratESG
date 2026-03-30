# =============================================================================
# config.py — StratESG Fund Configuration
# All fund parameters, asset universes, events, and weights live here.
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# -----------------------------------------------------------------------------
# Backtest windows
# Each window: (start_date, end_date, bptw_year)
# 1-month lag from list publication (Jan) → holdings from Feb
# -----------------------------------------------------------------------------
BACKTEST_WINDOWS = [
    ("2023-02-01", "2024-02-01", 2023),
    ("2024-02-01", "2025-02-01", 2024),
    ("2025-02-01", "2026-02-01", 2025),
]

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
BENCHMARKS = {
    "S&P 500": "SPY",
    "HAPI":    "HAPI",   # Harbor Human Capital Factor ETF (Oct 2022 inception)
}

# -----------------------------------------------------------------------------
# Conviction multipliers
# ─────────────────────────────────────────────────────────────────────────────
# Overlap verified directly from GPTW pages (no carry-forward needed):
#   2023 → greatplacetowork.com/best-workplaces/100-best/2023
#           NVDA #6, CRM #8 appear on both BPTW 2023 + GPTW 2023
#   2024 → greatplacetowork.com/best-workplaces/100-best/2024
#           NVDA #3, BOX #18 appear on both BPTW 2024 + GPTW 2024
#   2025 → uses GPTW 2024 (most recent at Feb 2025 start)
#           NVDA only overlap between BPTW 2025 + GPTW 2024
# -----------------------------------------------------------------------------
CONVICTION_HIGH = 1.0   # On both BPTW + GPTW same year
CONVICTION_LOW  = 0.5   # On BPTW only

# -----------------------------------------------------------------------------
# Portfolio construction parameters
# -----------------------------------------------------------------------------
VOL_WINDOW_DAYS   = 20       # Rolling window for volatility estimation
SECTOR_CAP        = 0.33     # Max allocation per GICS sector (pro-rata redistribution)
REBALANCE_FREQ    = "MS"     # Monthly rebalance (Month Start)
INITIAL_CAPITAL   = 100_000.0

# -----------------------------------------------------------------------------
# Red flag parameters
# Red flag mode: "strict" — size down 50% on event date,
#                only restore if company appears on next year's BPTW list
# -----------------------------------------------------------------------------
RED_FLAG_MODE    = "strict"
RED_FLAG_HAIRCUT = 0.50      # Retain this fraction on trigger (0.5 = size down 50%)

# -----------------------------------------------------------------------------
# Asset universes by BPTW year
# Format: ticker -> (gics_sector, conviction_source)
# conviction_source: "both" = BPTW + GPTW verified, "bptw" = BPTW only
# -----------------------------------------------------------------------------
UNIVERSES: Dict[int, Dict[str, Tuple[str, str]]] = {

    2023: {
        # GPTW 2023 verified overlap: NVDA (#6), CRM (#8)
        "NVDA": ("Information Technology",  "both"),
        "CRM":  ("Information Technology",  "both"),
        "BOX":  ("Information Technology",  "bptw"),
        "GOOGL":("Communication Services",  "bptw"),
        "NOW":  ("Information Technology",  "bptw"),
        "HUBS": ("Information Technology",  "bptw"),
        "ADBE": ("Information Technology",  "bptw"),
        "CRWD": ("Information Technology",  "bptw"),
        "MSFT": ("Information Technology",  "bptw"),
        "EXPI": ("Real Estate",             "bptw"),
        "MRVL": ("Information Technology",  "bptw"),
        "LULU": ("Consumer Discretionary",  "bptw"),
    },

    2024: {
        # GPTW 2024 verified overlap: NVDA (#3), BOX (#18)
        # VMware acquired by Broadcom Nov 2023 — data.py drops if no prices
        "NVDA": ("Information Technology",  "both"),
        "BOX":  ("Information Technology",  "both"),
        "NOW":  ("Information Technology",  "bptw"),
        "PCOR": ("Information Technology",  "bptw"),
        "VMW":  ("Information Technology",  "bptw"),
        "DAL":  ("Industrials",             "bptw"),
        "RJF":  ("Financials",              "bptw"),
        "ADBE": ("Information Technology",  "bptw"),
        "TOST": ("Information Technology",  "bptw"),
        "MSFT": ("Information Technology",  "bptw"),
        "ADSK": ("Information Technology",  "bptw"),
        "EXPI": ("Real Estate",             "bptw"),
        "LULU": ("Consumer Discretionary",  "bptw"),
        "LLY":  ("Health Care",             "bptw"),
        "CALX": ("Information Technology",  "bptw"),
    },

    2025: {
        # GPTW 2025 x BPTW 2025 overlap: NVDA only
        "NVDA": ("Information Technology",  "both"),
        "LLY":  ("Health Care",             "bptw"),
        "MSFT": ("Information Technology",  "bptw"),
        "EXPI": ("Real Estate",             "bptw"),
        "RLI":  ("Financials",              "bptw"),
        "ASML": ("Information Technology",  "bptw"),
        "BAH":  ("Industrials",             "bptw"),
        "ADSK": ("Information Technology",  "bptw"),
        "CRM":  ("Information Technology",  "bptw"),
        "ADBE": ("Information Technology",  "bptw"),
        "MSI":  ("Information Technology",  "bptw"),
        "PG":   ("Consumer Staples",        "bptw"),
        "MRK":  ("Health Care",             "bptw"),
        "EPAM": ("Information Technology",  "bptw"),
        "SAIL": ("Information Technology",  "bptw"),
        "SBGSY":("Industrials",             "bptw"),
    },
}

# -----------------------------------------------------------------------------
# Red flag events
# -----------------------------------------------------------------------------
@dataclass
class RedFlagEvent:
    ticker: str
    date:   str
    reason: str
    mode:   str   # "layoff" | "ceo_departure" | "protest"

RED_FLAGS: List[RedFlagEvent] = [
    RedFlagEvent("GOOGL", "2023-01-20", "12,000 layoffs (~6% workforce)",          "layoff"),
    RedFlagEvent("MSFT",  "2023-01-18", "10,000 layoffs (~5% workforce)",          "layoff"),
    RedFlagEvent("CRM",   "2023-01-04", "7,000 layoffs (~10% workforce)",          "layoff"),
    RedFlagEvent("LULU",  "2024-07-22", "CEO Calvin McDonald exits abruptly",      "ceo_departure"),
    RedFlagEvent("EXPI",  "2024-02-22", "Sexual misconduct allegations vs founder","protest"),
    RedFlagEvent("EPAM",  "2022-02-24", "Ukraine war — workforce disruption",      "protest"),
]

# -----------------------------------------------------------------------------
# NVIDIA vs Amazon case study events
# (ticker, date, label, color)
# -----------------------------------------------------------------------------
CASE_STUDY_EVENTS = [
    #("NVDA", "2018-01-15", "BPTW\nTop 25",               "#76b900"),
    #("NVDA", "2020-01-15", "BPTW\n#20",                  "#76b900"),
    #("NVDA", "2022-01-15", "BPTW #1\n+GPTW #5",          "gold"),
    ("NVDA", "2023-01-15", "BPTW #5\nBest Led #1",       "gold"),
    ("NVDA", "2024-01-15", "BPTW #2\nGPTW #3",           "gold"),
    #("AMZN", "2019-03-01", "OSHA: injury\nrate 2× avg",   "#cc4400"),
    #("AMZN", "2020-04-20", "COVID warehouse\nprotests",   "red"),
    #("AMZN", "2022-04-01", "ALU union\nvote victory",     "red"),
    ("AMZN", "2022-11-16", "10k layoffs",   "darkred"),
    ("AMZN", "2023-01-18", "18k layoffs",   "darkred"),
    ("AMZN", "2025-10-28", "14k layoffs",   "darkred"),
    ("AMZN", "2026-01-28", "16k layoffs",   "darkred"),
]

GLASSDOOR_RATINGS = {
    2016:(4.2,3.7), 2017:(4.3,3.6), 2018:(4.4,3.6),
    2019:(4.5,3.5), 2020:(4.5,3.4), 2021:(4.6,3.4),
    2022:(4.6,3.3), 2023:(4.5,3.2), 2024:(4.4,3.3), 2025:(4.4,3.4),
}

BPTW_APPEARANCES = {
    "NVDA": [2018,2020,2021,2022,2023,2024,2025,2026],
    "AMZN": [],
}

# -----------------------------------------------------------------------------
# Chart styling
# -----------------------------------------------------------------------------
COLORS = {
    "fund":    "#1F3864",
    "spy":     "#2196F3",
    "hapi":    "#FF9800",
    "nvda":    "#76b900",
    "amzn":    "#FF9900",
    "gold":    "#FFD700",
    "red":     "#D32F2F",
    "darkred": "#7B0000",
    "bg":      "#FAFAFA",
    "grid":    "#E0E0E0",
}

OUTPUT_DIR = "output"



#### EXTRA STUFF

import os

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient


load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")


ALPACA_CONFIG = {
    "API_KEY": ALPACA_API_KEY,
    "API_SECRET": ALPACA_API_SECRET,
    "PAPER": True,
}

STOCK_CLIENT = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
CRYPTO_CLIENT = CryptoHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)

TRADING_CLIENT = TradingClient(ALPACA_API_KEY, ALPACA_API_SECRET, paper=True)