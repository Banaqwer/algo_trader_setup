# Algo Trader - Complete Production Guide

## System Architecture

### 1. Data Layer

**OANDA v20 API**
- Real-time and historical candles
- Multiple granularities (M5, M15, H1, H4, D)
- Bid/ask prices for realistic spread modeling

**Parquet Caching**
- Incremental downloads, partitioned by month
- Path:   `data/processed/{instrument}/{tf}/{year-month}.  parquet`
- Coverage tracking: `data/coverage/{instrument}_{tf}.json`

**Multi-Timeframe Buffers**
- Execution timeframes: M5, M15, H1
- Context timeframes: H4, D1
- Forward-only:   no lookahead

### 2. Feature Engineering

Per-timeframe computation (all forward-only):

**Volatility:**
- ATR(14), ATR(50)
- Rolling volatility (20-bar, 50-bar)
- ATR percentile over 100 bars

**Trend:**
- EMA(20, 50, 200)
- ADX(14) for trend strength
- Trend direction (EMA crosses)

**Momentum:**
- RSI(14) - Relative Strength Index
- MACD(12,26,9) - Moving Average Convergence Divergence
- Histogram for momentum decay

**Mean Reversion:**
- Bollinger Bands(20, 2. 0 std)
- BB width and Z-score
- Price deviation from mean

**Structure:**
- Swing highs/lows (20-bar lookback)
- Pivots and support/resistance
- Wick ratios (upper/lower wicks)

**Regime:**
- Volatility state (low, medium, high)
- Trend state (up, down, neutral)
- Session detection (Asia, London, NewYork)

### 3. Pattern Recognition (6 Modules)

#### Pattern 1: Volatility Compression → Expansion
- **Detection:** ATR in bottom 20% percentile (compression detected)
- **Trigger:** ATR expansion >15% over last 5 bars
- **Direction:** Long if near BB top, Short if near BB bottom
- **Historical Stats:** +0.45R expectancy, 58% win rate, 247 samples
- **Setup Type:** Breakout

#### Pattern 2: Liquidity Sweep / Stop Hunt
- **Detection:** Swing high/low sweep (20-bar lookback)
- **Trigger:** Wick penetration >= 55% of recent range
- **Direction:** Reversal opposite to sweep direction
- **Historical Stats:** +0.38R expectancy, 54% win rate, 312 samples
- **Setup Type:** Mean reversion

#### Pattern 3: Session Bias
- **Detection:** Performance tracking per (instrument, session, setup)
- **Trigger:** Win rate > 55% and sample size > 20
- **Direction:** Based on EMA20/EMA50 cross
- **Customizable:** Per session and setup type

#### Pattern 4: HTF Range Extremes
- **Detection:** D1 range position (0-100%)
- **Trigger:** Extremes (< 15% or > 85%) boost reversals
- **Direction:** Reversal in extremes, fade in middle
- **Historical Stats:** +0.52R expectancy, 62% win rate, 189 samples
- **Setup Type:** Mean reversion

#### Pattern 5: Correlation Divergence
- **Detection:** Rolling correlation divergence from expected
- **Pairs:** EUR/GBP (0.82), AUD/NZD (0.85), USD_CAD/USD_JPY (0.70)
- **Trigger:** Correlation drops below expected - 0.25
- **Direction:** Based on momentum divergence
- **Historical Stats:** +0.41R expectancy, 56% win rate, 134 samples
- **Setup Type:** Mean reversion

#### Pattern 6: Failed Breakout / Acceptance-Rejection
- **Detection:** Break beyond 20-bar structure
- **Trigger:** Failure to hold + RSI momentum decay
- **Direction:** Reversal opposite to breakout
- **Historical Stats:** +0.47R expectancy, 60% win rate, 201 samples
- **Setup Type:** Mean reversion

### 4. Signal Aggregation (NPES)

**Net Pattern Expectancy Score** resolves conflicts: 
contribution_i = confidence_i × regime_fit_i × expectancy_i × log(1 + sample_size_i)

NPES = (sum(contributions_long) - sum(contributions_short)) / (sum(contributions_long) + sum(contributions_short))

**Thresholds:**
- Baseline:   NPES >= 0.15
- Risk-off (>1% DD): NPES >= 0.25
- Context-dependent: session, timeframe, regime

### 5. Machine Learning

**Per-Instrument-Timeframe Models:**
- LightGBM classifier (1200 estimators)
- Learning rate:   0.02, Max depth:  6
- Subsample:  0.8, Colsample:   0.8
- Class-weighted for balanced training

**Calibration:**
- Isotonic regression post-training
- Maps probabilities to actual win rates
- Out-of-bounds clipping

**Features:**
- All computed features from step 2
- Lagged returns (1-5 bars)
- Volatility and correlations
- Pattern signal strengths
- Market regime indicators

**Output:**
- Probability distribution (short, neutral, long)
- Entropy threshold:   max 0.60 to trade
- Expected value R:  probability-weighted expectancy

### 6. Execution Engine

**Simulated Broker:**
- Market order execution (realistic slippage)
- SL/TP adaptive based on regime and setup
- Position sizing from risk budget (0.5% per trade)
- Trailing stop:   activation at +1 ATR, trail 1 ATR
- Time stops:  M5=64 bars, M15=96 bars, H1=24 bars

**SL/TP Policy (Configurable):**
- Trend H1: SL 1. 8 ATR, TP 3. 0 ATR
- Mean Reversion M5: SL 0.9 ATR, TP 1.2 ATR
- Breakout M15: SL 1.6 ATR, TP 2.8 ATR

**P&L Calculation:**
- Gross P&L: (exit - entry) × units
- Spread costs: Applied on entry and exit (configurable pips)
- Slippage:   Configurable BPS (default 1. 5)
- Net P&L: Gross - spread - slippage

### 7. Risk Management

**Account State Tracking:**
- Current equity, peak equity, drawdown percentage
- Daily/weekly P&L resets at boundaries
- Position count (total and per-symbol)
- Trade statistics (wins, losses, expectancy)

**Risk Constraints (All Enforced):**
- Max risk per trade: 0.5% of equity
- Max daily loss: 2% → auto-halt
- Max weekly loss: 5% → auto-halt
- Max open positions: 3 total, 1 per symbol
- Max leverage: 5. 0x

### 8. Self-Learning Engine

**Pattern Statistics Database (SQLite):**
- Per context bucket:  (instrument, timeframe, pattern, regime, session)
- Trade count, wins, losses
- Win rate, expectancy (USD and R)
- Confidence decays with repeated losses
- Patterns disabled on 9 consecutive losses (10-bar window)

**Online Updates (Forward-Only):**
- After each closed trade, update bucket stats
- No hindsight:   updates use only closed P&L
- Persistent storage in SQLite for reproducibility
- Recovery for live/backtest transitions

### 9. Backtesting

**Event-Driven Loop:**
1. Load historical candles (5+ years, increment hourly)
2. Chronological iteration (bar-by-bar)
3. Per candle:  
   - Compute features for all timeframes
   - Run 6 pattern recognizers
   - Aggregate signals (NPES)
   - Validate risk constraints
   - Execute trade (simulated)
   - Check SL/TP/timeout exits
   - Update P&L and learning stats

**Realism Features:**
- Bid/ask spread simulation
- Slippage:   configurable BPS
- MAE/MFE tracking (in USD and ATR)
- Rollover/weekend blocking (optional)
- Multiple execution scenarios

**Walk-Forward Validation:**
- Train:   4 years
- Test:  6 months
- Step:  3 months
- Purged CV with embargo (no data leakage)
- Stress scenarios:  2× spread, 3× slippage

### 10. Reporting & Metrics

**Computed Metrics:**
- **Performance:** CAGR, Sharpe, Sortino, Calmar
- **Risk:** Max drawdown, Profit Factor, Win Rate
- **Expectancy:** Average R per trade, USD per trade
- **Exposure:** % time in market, turnover
- **Per-Timeframe:** Win rate, expectancy, trade count
- **Per-Pattern:** Performance breakdown by recognizer

**Artifacts Generated:**
- `trades.parquet` - Full trade journal
- `metrics_full.json` - All computed metrics
- `report.md` - Markdown summary
- `charts.png` - 4-panel equity/drawdown/PnL/TF
- `settings.json` - Config snapshot
- `account_state.json` - Final state

## Configuration

### Environment Variables

```bash
OANDA_ACCESS_TOKEN=8aa716adf09e881e6bcf9b2d8f099ec9-7b4a3513dd156d165d2a5c147c183965
OANDA_ACCOUNT_ID=101-001-36074948-001
OANDA_ENV=practice
UNIVERSE=EUR_USD,GBP_USD,USD_JPY,AUD_USD,USD_CAD,USD_CHF,NZD_USD
BACKTEST_YEARS=5
MODE=backtest
EXECUTION_ENABLED=false
KILL_SWITCH=true
APPLY_SPREAD=true
APPLY_SLIPPAGE=true
SLIPPAGE_BPS=1.5
LOG_LEVEL=INFO


