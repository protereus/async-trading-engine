# IG Live Trading — Risk Management Technical Reference

**Purpose:** This document was written as pre-implementation research before building the risk layer, and became the specification the risk-management modules are implemented and tested against. Every section maps a class of live-environment risk to concrete code requirements, constants, error codes, and validation criteria that the demo environment does not enforce.

**Scope:** UK retail spread betting account under FCA jurisdiction, accessed via IG REST API (transactions/history) and Lightstreamer (streaming prices/account events).

**Core principle:** Long-term survival depends far less on signal accuracy than on defensive engineering against API instability, structural execution costs, and rigid regulatory liquidation thresholds. The demo environment hides all of these.

**Reading this against the code:** The spec was drafted generically and sometimes prescribes thread-based mechanics (a "SessionManager thread", `tenacity` decorators, `threading.Event` flags). The engine is a single-loop `asyncio` application, so each of those translates to an asyncio construct: background threads → `asyncio` tasks, `tenacity` → the retry/backoff loop in `ig_http.py`, shared thread flags → plain state on the event loop. Section 8 maps every spec module to the file that implements it and the tests that pin it. Checklist items below are ticked where implemented; the one open item is marked.

---

## 0. Mandatory Implementation Checklist

A live-ready bot must satisfy every item below. Treat this as a pre-flight gate before flipping `BOT_ENV=live`. (`uv run pytest -m preflight` runs the deterministic pre-go-live validation subset.)

- [x] Pre-trade `marketStatus` query on every epic before opening a new tranche — `IGClient.require_tradeable`, gated in `rerank_runner.py`
- [x] Slippage-aware fill model with optional `guaranteed stop` parameter — `ig_margin.py` slippage/premium tables, `guaranteedStop` on the order payload
- [ ] Corporate action / ex-dividend calendar parser with technical-trigger suppression — **not implemented**; the exposure is pinned by the ex-div audit tests in `tests/test_take_profit.py`
- [x] REST rate limiter set to `allowanceAccountTrading - 2` with exponential backoff and jitter — `ig_http.py` per-bucket token buckets
- [x] Independent task for CST / X-SECURITY-TOKEN refresh on a < 60s rolling window — `ig_session.py` keepalive loop (45 s)
- [x] Lightstreamer heartbeat monitor with self-healing reconnect + re-subscribe — `ig_ls_connection.py`
- [x] Tick validation schema rejecting zero / null / >N-σ outliers — `TickValidator` in `ig_feed_handlers.py`
- [x] Real-time margin utilisation calculation with circuit breaker at 80% — `bot/risk/margin.py`
- [x] Tier-aware position sizing per asset class — `bot/risk/ig_margin.py`
- [x] Overnight funding model including Wednesday FX 3× and Friday equities 3× multipliers — `bot/risk/funding.py`
- [x] Spread-widening monitor with execution halt above 2σ from 30-day mean — `bot/risk/spread_monitor.py`
- [x] FSCS £120,000 cap enforced as a soft ceiling per broker instance — `bot/risk/fscs.py`

---

## 1. Execution Dynamics

### 1.1 Slippage

**Demo behaviour:** Orders fill instantaneously at the quoted price. The matching engine does not interact with a real order book. Size never causes rejection.

**Live behaviour:** Market orders sweep the actual order book. Stop-losses triggered during volatility (CB announcements, macro releases, geopolitical shocks) execute against thinning liquidity and fill at materially worse prices than the algorithm's calculated stop level. Backtested risk-reward ratios are systematically invalidated by negative slippage.

**Implementation requirements:**

- Treat the `level` parameter on stop orders as a **trigger**, not an execution price.
- Add a `slippage_buffer_pips` configuration value per asset class; expected fill = `stop_level ± slippage_buffer`.
- Where the strategy cannot tolerate gap risk, append `"guaranteedStop": true` to the order payload.
- When `guaranteedStop` is used, deduct the **guaranteed stop premium** from expected value calculations:

| Volatility Category | Example Instruments | Premium | Min Stop Distance |
|---|---|---|---|
| Low | FTSE 100, major US stocks | 0.30% of position value | 5.0% – 10.0% |
| Medium | Mid-cap equities, SETSmm shares | 0.70% | 7.5% |
| High | Small-cap equities, EM stocks | 1.00% | 12.5% |

- Recompute strategy net alpha with continuous premium drag. Scalping / high-frequency strategies often turn from marginally profitable in backtest to persistently loss-making once guaranteed-stop premiums are applied.

### 1.2 Ex-Dividend Adjustments

When an index constituent goes ex-dividend, the index price drops mathematically by the payout amount. The broker neutralises P&L via a cash credit (long) or debit (short) to the ledger.

**Risk:** The drop is **synthetic** and not driven by market sentiment. Algorithms using moving averages, momentum oscillators, or absolute-price breakout triggers will misread the adjustment as a bearish breakout and may open erroneous shorts.

**Implementation requirements:**

- Parse a corporate-action calendar feed (third-party or scraped) for every traded epic.
- Maintain a per-epic `ex_dividend_dates: List[date]` cache.
- On any ex-dividend date, **suspend technical triggers for the affected epic for at least one full session** (recommended: from market open until the next bar after the adjustment has propagated through indicator windows).
- Tag affected bars in the candle store with an `adjustment_flag` so indicators can optionally exclude or backfill them.

### 1.3 Market Restrictions

The live environment introduces dynamic restrictions absent from demo:

- `MARKET_CLOSED_WITH_EDITS` — closing-only state
- `EDITS_ONLY`
- `OFFLINE`
- `ON_AUCTION`
- `SUSPENDED`

Causes include insufficient liquidity, underlying clearing partner raising margin requirements, market-cap thresholds breached, or corporate actions.

**Risk:** Any grid / DCA / martingale / scale-in strategy that depends on opening successive tranches will be left holding an unbalanced, unhedged book if a `Closing only` state is encountered mid-execution.

**Implementation requirements:**

- Before every **opening** order, query `GET /markets/{epic}` and validate `marketStatus == "TRADEABLE"`.
- If non-tradeable, route to a `RestrictedMarketHandler` that:
  - Logs the restriction with reason code
  - Cancels any pending scale-in tranches for that epic
  - Triggers a position-reconciliation routine to confirm net exposure matches strategy intent
  - Optionally hedges or flattens the unbalanced portion via a correlated tradeable proxy

---

## 2. REST API Resilience

### 2.1 Rate Limiting

The REST API uses a token bucket. Limits are dynamic. The documented allowance is ~40 trade requests/minute but **enforcement diverges between demo and live**. Background session token refreshes (every ~60s on a separate thread) consume primary allowance tokens, causing 403s on what looks like an under-limit request rate.

**Error codes to classify and handle:**

```
error.public-api.exceeded-api-key-allowance
error.public-api.exceeded-account-allowance
error.public-api.exceeded-account-trading-allowance
error.public-api.exceeded-account-historical-data-allowance
```

**Implementation requirements:**

- Configure the internal rate limiter to `allowanceAccountTrading - 2 requests/minute` (the "magic number" buffer for session-refresh overhead).
- Maintain **separate buckets** for: trade requests, historical data requests, account/info queries.
- Replace any `time.sleep()` pacing with exponential backoff + jitter. Use `tenacity` in Python or equivalent.

**Reference backoff pattern:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type

@retry(
    retry=retry_if_exception_type(ApiExceededException),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    stop=stop_after_attempt(6),
    reraise=True,
)
def place_order(payload): ...
```

- Catch HTTP 403, 429, 500, 501, 503 explicitly. Do not retry 400 / 401 — those are payload or auth bugs.

### 2.2 Session Management

IG v3 session tokens expire after **60 seconds of inactivity**. Both `CST` and `X-SECURITY-TOKEN` headers must be present on every authenticated call. A missing or stale token returns `KeyError: CST` on parse, or 401/403 on execution.

**Implementation requirements:**

- Run a dedicated `SessionManager` thread (or async task) that:
  - Tracks `token_issued_at` timestamp
  - Refreshes via `POST /session/refresh-token` at the 45-second mark (15-second safety margin)
  - Exposes a thread-safe accessor returning the **current** token pair
- All transactional code must read tokens via the accessor immediately before request construction — never cache locally for more than one request.
- On any 401, force an immediate refresh and a single retry. On second failure, halt trading and alert.

### 2.3 Infrastructure Reliability

The public status page **lags actual API health**. Documented behaviour: API returns 500 / 501 / 503 while `status.ig.com` shows "All Systems Operational". Specific epics (e.g. `TSLA`) have returned zero-value or empty data arrays in live while demo remained fully functional.

**Implementation requirements:**

- Do not trust the status page as a gate. Implement local health metrics.
- Add a **tick validation schema** to every ingestion path. Reject and discard ticks where:
  - `bid == 0` or `ask == 0` or `null`
  - Non-numeric values
  - Tick price differs from previous tick by > N standard deviations (configurable, default N = 6) of the trailing 100-tick window
- On corrupted-tick detection: suspend trading on the affected epic, flag for manual review, and continue polling for recovery.
- For historical data backfills, chunk requests into 12-hour windows and stitch — bypasses undocumented historical-data throttling.

---

## 3. Lightstreamer Streaming Resilience

### 3.1 Silent Thread Death

The Lightstreamer Python SDK runs background listener threads in `MERGE` mode (continuous prices) and `DISTINCT` mode (discrete trade confirmations). When the IG server force-drops the connection — which happens reliably after ~2 hours, and sometimes daily — these threads **die without raising an exception** and without notifying the main thread.

Consequence: the algorithm believes it is connected, waits for ticks that never arrive, fails to track P&L, fails to trail stops, fails to execute exits. Unhedged exposure drifts.

### 3.2 Self-Healing Architecture

**Implementation requirements — non-negotiable:**

- Active heartbeat monitor: a main-thread loop that compares the UNIX timestamp of the last received tick against the system clock.
- Threshold: if no tick received across **any subscribed channel** for `HEARTBEAT_TIMEOUT_SEC` (default 10s), declare the connection dead.
- Pass heartbeat flags between listener threads and main thread via a thread-safe shared object (e.g. `threading.Event` or a Redis key) — not via stdout, which `nohup` and similar tools break.

**Recovery protocol (must execute in order):**

1. Tear down the broken Lightstreamer connection. Free sockets and listener threads.
2. Re-authenticate via REST `POST /session` to obtain **fresh** CST + X-SECURITY-TOKEN. The previous tokens are invalidated by the drop.
3. Re-establish the Lightstreamer session with the new cryptographic headers.
4. Re-subscribe to every `MERGE` and `DISTINCT` channel. **Server-side subscriptions are wiped on disconnect** — they do not survive reconnect.
5. Resume only after the heartbeat monitor confirms at least one valid tick on each subscription.

**Additional error to handle:** `Cause: 2 - Requested Adapter Set not available` — server-side misconfiguration. Pause execution, exponential backoff, retry after 30–120s.

**Deployment note:** `nohup python bot.py &` frequently breaks Lightstreamer background threading due to stdout buffering. Use a process supervisor (systemd, supervisord, or a wrapper shell script that redirects stdout/stderr to a log file and forces unbuffered Python via `python -u`).

---

## 4. Leverage, Margin, and Forced Liquidation

### 4.1 Retail Margin Rates

FCA-regulated retail accounts operate under mandatory margin floors. Position sizing logic must dynamically read the margin rate from this table:

| Asset Class | Instrument | Retail Margin Rate | Max Leverage |
|---|---|---|---|
| Forex | Major pairs (EUR/USD, GBP/USD) | 3.33% | 1:30 |
| Forex | Minor pairs (AUD/USD, EUR/JPY) | 5.00% | 1:20 |
| Indices | Major (FTSE 100, US 500) | 5.00% | 1:20 |
| Commodities | Spot Gold | 5.00% | 1:20 |
| Commodities | Spot Silver, Copper, US Crude, Brent | 10.00% | 1:10 |
| Equities | Major shares & ETFs | 20.00% | 1:5 |
| Crypto | BTC, ETH (where permitted) | 50.00% | 1:2 |

**Worked example:** £10,000 notional EUR/USD consumes £333 of free margin. The same £10,000 notional on an equity ETF consumes £2,000 — a 6× difference. A cross-asset portfolio must size positions per asset, not per notional.

### 4.2 Tiered Margining

Margin rates **increase** as position size crosses tier thresholds. A linear scaling algorithm that pushes from Tier 1 into Tier 2 will see the marginal cost per point of exposure double or triple. The full order may be rejected for insufficient margin.

**Implementation requirements:**

- Maintain an internal `ExposureMap` keyed by epic and asset class, tracking current notional.
- Before any new order, calculate **post-trade exposure** and look up the resulting tier.
- If the trade would push exposure into a higher tier, either:
  - Re-size the order to fit within the current tier ceiling, or
  - Pre-deposit the additional margin required at the new tier, or
  - Reject the trade and log a tier-breach event
- Source IG's current tier matrices from the official PDFs (e.g. "TIERED SPREAD BETTING MARGINS: INDICES") and cache locally. Refresh on a weekly schedule.

### 4.3 The 50% Margin Close-Out Rule

**This is the single most important live constraint.** If account equity falls below **50% of the total margin required** to hold open positions, the broker's automated systems will close one or more positions as soon as market conditions allow. There is no warning, no grace period, no appeal.

- Demo accounts do **not** enforce this. A mean-reversion strategy that averages down through 80–90% unrealised drawdowns will appear spectacular in backtest and will be liquidated at maximum drawdown in live.
- English Court of Appeal precedent and Financial Ombudsman Service decisions confirm: the broker has **no duty** to protect the customer from this mechanism. The close-out is legally fortified.

**Implementation requirements:**

- Compute margin utilisation ratio on every tick that updates open-position P&L:

```
margin_utilisation = (equity - unrealised_pnl_loss) / total_margin_required
```

Where `equity` = account balance + unrealised P&L, and `total_margin_required` = sum of margin held against all open positions.

- Equivalent form for monitoring distance from liquidation:

```
liquidation_buffer = (equity / total_margin_required) - 0.50
```

- Hard-coded circuit breakers (configurable, defaults shown):
  - At `equity / total_margin_required` = **0.80** → halt all new position entries
  - At **0.65** → begin defensive position-reduction (close worst-performing positions first)
  - At **0.55** → emergency flatten of all non-hedge positions
- Recompute on every account-status push from Lightstreamer **and** on every tick of an instrument with an open position. Never rely on the broker to warn.

---

## 5. Overnight Funding

Positions held past **22:00 UK time** incur overnight funding. Backtests rarely model this accurately. Live drag is asymmetric, asset-dependent, and amplified by weekend multipliers.

### 5.1 Forex (Tom-Next Rate)

Three-step calculation per position per night:

```
V = (price_in_points × 1.5%) / 360

swap_rate_long  = tom_next_rate + V
swap_rate_short = tom_next_rate - V

overnight_cost = bet_size × swap_rate
```

- The 1.5% is the IG administration markup (annualised).
- Divisor is 360 for FX.
- **Wednesday anomaly:** spot FX settles T+2, so a position held through 22:00 Wednesday incurs **three days** of funding to cover weekend settlement. Daily carry effectively spikes 300% on Wednesday night.

**Implementation requirements:**

- Hold a `tom_next_rate` cache per pair, updated daily.
- Funding model must multiply by 3 for Wednesday-night holdings.
- Subtract cumulative expected funding from the strategy's projected alpha during pre-trade EV checks.

### 5.2 Equities, Indices, ETFs

```
overnight_charge = position_value × (benchmark_rate ± admin_fee) / divisor
```

- `benchmark_rate`: SONIA (GBP), ESTR (EUR), SOFR (USD), etc.
- `admin_fee`: 2.5% – 3.4% annualised
- `divisor`: **365** for GBP / SGD / ZAR; **360** for other currencies
- Sign: long positions are **charged** the benchmark + admin; short positions **receive** the benchmark minus admin (often net negative when rates are low)
- **Friday anomaly:** equities positions held through 22:00 Friday incur 3× funding for the weekend (vs. Wednesday for FX)

**Short-selling additional cost — borrow fee:**

- Variable annualised charge based on interbank borrow rate + 0.5% admin
- Highly volatile for small-caps; can spike to double-digit annualised rates with zero notice
- Pull from the API per epic before opening any short; reject if borrow > configured ceiling

### 5.3 Commodities and Bonds (Daily Funded Bets)

DFBs have no expiry. Pricing is synthesised from the two most liquid futures contracts (front-month `P2`, back-month `P3`).

```
basis_adjustment = synthetic_curve_movement (cash-neutral, offsets running P&L)
ig_charge        = position_value × admin_fee / divisor
```

- Admin fee: 3.4% for commodities, 3.0% for sovereign bonds
- The basis adjustment is cash-neutral; the IG charge is **absolute capital destruction**
- For medium-term trend-followers holding for weeks/months, subtract the annualised admin fee directly from the strategy's projected CAGR

### 5.4 Implementation: Funding Module

A standalone `OvernightFundingCalculator` module must:

- Accept an open position and the current UK datetime
- Return expected funding cost for the upcoming 22:00 rollover
- Apply weekday multipliers correctly (Wed × 3 for FX, Fri × 3 for equities/indices/commodities)
- Feed projected funding into the position's running EV; if expected funding over the planned hold period exceeds projected gross alpha, the position should not be opened

---

## 6. Spread and Hidden Costs

### 6.1 Dynamic Spread Widening

Spreads are integrated into the bid-ask, not a line-item fee. Every position opens at an immediate loss equal to the spread. Spreads widen with volatility and illiquidity. UK non-FTSE 350 small-caps were recently widened from 0.35% to 0.50% on DFBs.

**Implementation requirements:**

- Maintain a 30-day rolling mean and standard deviation of the spread per epic.
- Before any order: query the current spread via the API.
- If `current_spread > mean + 2σ`, **halt execution** on that epic until spread normalises.
- For scalping/HF strategies: log cumulative spread paid and compare against gross alpha. If spread cost ≥ gross alpha, kill the strategy.

### 6.2 ProRealTime Subscription (if applicable)

If the bot is later routed via PRT: £30/month subscription, refunded if ≥ 4 qualifying trades execute. IG reserves the right to maintain the fee for low-nominal-value trades. Not relevant if API-only.

---

## 7. Legal and Systemic Boundaries

### 7.1 No Broker Duty of Care

- English Court of Appeal: contractual margin close-out provisions exist for **the broker's** protection, not the customer's.
- FOS decisions: brokers are not required to warn before close-out, not required to honour appeals on platform-instability losses, not required to "protect the customer from himself".
- **The algorithm's internal risk code is the only line of defence.** Build accordingly.

### 7.2 FSCS £120,000 Cap

- Client funds are held in segregated nominee accounts at a Tier-1 custodian.
- In the event of broker or custodian insolvency, losses are shared proportionally; FSCS covers up to **£120,000 per person per institution**.
- **Implication for scaling:** if compounded capital exceeds £120,000, the surplus is uninsured at this broker. Plan for capital splits across independent brokerage infrastructures before crossing this threshold.

---

## 8. Module-to-Risk Mapping

Map each risk to a single owning module so concerns stay separated. The first column is the module as originally specified; the second and third columns are where it landed in the codebase and the tests that pin its behaviour.

| Spec module | Implementation | Tests |
|---|---|---|
| `SessionManager` — CST / X-SECURITY-TOKEN lifecycle, refresh + keepalive | `src/bot/execution/ig_session.py` (async tasks, not threads) | `tests/test_ig_client.py` |
| `RateLimiter` — token buckets per endpoint class, exponential backoff | `src/bot/execution/ig_http.py` | `tests/test_ig_client.py` |
| `MarketStateGuard` — pre-trade `marketStatus` checks, restricted-market handling | `IGClient.require_tradeable` + entry gate in `rerank_runner.py`; corporate-action calendar **not implemented** | `tests/test_ig_client.py`, ex-div audit in `tests/test_take_profit.py` |
| `StreamSupervisor` — LS heartbeat, reconnect, re-subscribe, tick validation | `src/bot/data/ig_ls_connection.py`, `TickValidator` in `ig_feed_handlers.py` | `tests/test_ig_feed.py` |
| `MarginEngine` — utilisation ratio, tier lookups, circuit breakers | `src/bot/risk/margin.py`, `src/bot/risk/ig_margin.py` | `tests/test_margin.py`, `tests/test_ig_margin.py` |
| `FundingCalculator` — tom-next, benchmarks, DFB charges, weekday multipliers | `src/bot/risk/funding.py` | `tests/test_funding.py` |
| `SpreadMonitor` — rolling mean/σ per epic, halt on widening | `src/bot/risk/spread_monitor.py` | `tests/test_spread_monitor.py` |
| `CostLedger` — guaranteed-stop premiums, funding accrual | premium + slippage tables in `src/bot/risk/ig_margin.py`, funding preview in `ig_convert.py`; statement reconciliation **not implemented** | `tests/test_ig_margin.py` |
| `OrderRouter` — slippage buffers, guaranteed-stop attachment | `src/bot/execution/ig_client.py`, `ig_convert.py` | `tests/test_ig_client.py` |
| `CircuitBreaker` — halts, equity/margin thresholds, risk-event logging | `src/bot/risk/risk_manager.py` (drawdown tiers, loss windows, halt state) | `tests/test_risk_manager.py` |

---

## 9. Validation Criteria Before Going Live

Before the `BOT_ENV=live` flag flips. Items with an automated equivalent in the suite are noted; the deterministic subset runs as `uv run pytest -m preflight`.

1. **Chaos test the streaming layer** — kill the Lightstreamer connection at random intervals via `iptables` rules. The bot must reconnect, re-subscribe, and lose zero positions. *(Mock-level equivalent: forced-disconnect reconnect/re-subscribe tests in `tests/test_ig_feed.py`; the host-level `iptables` run stays manual.)*
2. **Force 403/500 responses** — proxy that injects errors into the REST path. Bot must back off and recover. *(Automated: retry/backoff and rate-limit-error tests in `tests/test_ig_client.py`.)*
3. **Synthetic ex-dividend test** — inject a sudden 2% price drop on a held index. Bot must not interpret as a signal. *(Audited: `tests/test_take_profit.py` pins the current — still vulnerable — behaviour until suppression lands.)*
4. **Margin walk-down** — simulate a steady drawdown approaching 80% utilisation. Verify each circuit breaker fires at the correct threshold. *(Automated: threshold-transition tests in `tests/test_margin.py`.)*
5. **Wednesday/Friday rollover test** — verify funding calculations multiply by 3 on the correct nights for FX vs equities respectively. *(Automated: `tests/test_funding.py`.)*
6. **Restricted market test** — attempt to scale into an epic flagged `MARKET_CLOSED_WITH_EDITS`. Bot must reject the scale-in and trigger reconciliation. *(Automated: market-status gate tests in `tests/test_ig_client.py`.)*
7. **Token expiry test** — idle the bot for > 60 seconds, then issue an order. Refresh must have happened automatically. *(Automated: keepalive/refresh tests in `tests/test_ig_client.py`.)*

Only after all seven pass deterministically should live capital be deployed, and even then, start with the smallest permitted bet size and scale exposure over a minimum of two weeks while observing live behaviour against expected behaviour.

---

## 10. Summary Architectural Mandates

1. **Microstructure adaptation** — slippage buffers, market-status pre-checks, ex-dividend suppression, partial-fill handling.
2. **REST resilience** — `allowance - 2` rate buffer, independent token-refresh thread, exponential backoff with jitter, full error-code taxonomy.
3. **Streaming stability** — active heartbeat, self-healing reconnect, tick validation, full re-subscription on every recovery.
4. **Capital preservation** — real-time margin utilisation, tier-aware sizing, circuit breakers well above the 50% liquidation floor.
5. **Cost internalisation** — funding (with weekday multipliers), guaranteed-stop premiums, dynamic spreads — all subtracted from projected alpha *before* a trade is taken.

Success in live trading is a function of defensive engineering, not signal accuracy. Treat every section above as load-bearing.
