"""Tests for the read-only dashboard snapshot.

Specifically guards the 2026-06-01 patch that:

* Switched the dashboard's open-positions display from ``ig_quote_scale``
  to the ``ig_display_price`` divisor.  The IG-native metals (XAU/XAG) carry a
  divisor of 1.0, so they render at the IG spot level directly.
* Added a ``stop_display`` + ``stop_pct`` column derived from
  ``take_profit_state`` so the operator can see the IG-side stop level
  per position without leaving the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from webgui.data import DashboardData


@pytest.fixture()
def fake_environment(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal candles.db + bot_state.json pair the dashboard can read.

    XAU/USD (IG-native, display divisor 1.0): entry 8726.2 / latest 8855
    so the test sees the new display divisor in action.  A TP state with
    ``entry_stop_pct=0.05`` confirms the stop column populates.
    """
    db_path = tmp_path / "candles.db"
    state_path = tmp_path / "bot_state.json"

    conn = sqlite3.connect(db_path)
    # Minimal schema — only the columns DashboardData touches.
    conn.executescript(
        """
        CREATE TABLE candles (
            symbol TEXT, timestamp INTEGER, open REAL, high REAL,
            low REAL, close REAL, volume REAL,
            PRIMARY KEY (symbol, timestamp)
        );
        CREATE TABLE signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            horizon_bars INTEGER NOT NULL,
            mean_return REAL, direction_confidence REAL,
            uncertainty REAL, entry_price REAL,
            realized_return_at_horizon REAL, gap_spanned INTEGER DEFAULT 0
        );
        CREATE TABLE asset_correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            computed_at INTEGER, symbol_a TEXT, symbol_b TEXT, correlation REAL
        );
        CREATE TABLE macro_features (
            series_id TEXT, observation_date INTEGER, value REAL, fetched_at INTEGER,
            PRIMARY KEY (series_id, observation_date)
        );
        CREATE TABLE sentiment_scores (
            scored_at INTEGER, asset TEXT, sentiment REAL,
            confidence REAL, agreement REAL, sources TEXT, escalated INTEGER,
            PRIMARY KEY (scored_at, asset)
        );
        """
    )
    # One XAU/USD candle at the IG spot level.
    conn.execute(
        "INSERT INTO candles VALUES ('XAU/USD', 1780318800000, 8800.0, 8860.0, 8790.0, 8855.0, 0.0)"
    )
    # One XAG/USD candle at the post-restart 14:07 price.
    conn.execute(
        "INSERT INTO candles VALUES "
        "('XAG/USD', 1780318800000, 10320.0, 10330.0, 10310.0, 10325.0, 0.0)"
    )
    conn.commit()
    conn.close()

    state = {
        "positions": {
            "XAU/USD": {
                "symbol": "CS.D.USCGC.TODAY.IP",
                "side": "buy",
                "entry_price": 8726.2,
                "quantity": 0.06,
                # Heartbeat-refreshed live IG spot level; the dashboard
                # prefers this over the (stale) candle close.
                "current_price": 8855.0,
                "unrealised_pnl": 0.0,
                "realised_pnl": 0.0,
                "opened_at": 1780085284000,
                "updated_at": 1780085284000,
            },
            "XAG/USD": {
                "symbol": "CS.D.USCSI.TODAY.IP",
                "side": "buy",
                "entry_price": 10321.6,
                "quantity": 1.08,
                "current_price": 10325.0,  # heartbeat-refreshed live IG level
                "unrealised_pnl": 0.0,
                "realised_pnl": 0.0,
                "opened_at": 1780322825000,
                "updated_at": 1780322825000,
            },
        },
        "take_profit_state": {
            "XAU/USD": {
                "symbol": "XAU/USD",
                "entry_price": 8726.2,
                "entry_stop_pct": 0.05,
                "current_trailing_stop": None,
                "peak_price": 8726.2,
                "opened_at_ms": 1780085284000,
                "signal_decay_strikes": 0,
                "topk_miss_strikes": 0,
                "breakeven_armed": False,
                "trail_armed": False,
                "predicted_mfe_pct": 0.05,
                "predicted_mae_pct": 0.005,
                "predicted_peak_bar": 60,
                "bar_interval_ms": 3600000,
            },
            "XAG/USD": {
                "symbol": "XAG/USD",
                "entry_price": 10321.6,
                "entry_stop_pct": 0.0177,
                "current_trailing_stop": None,
                "peak_price": 10321.6,
                "opened_at_ms": 1780322825000,
                "signal_decay_strikes": 0,
                "topk_miss_strikes": 0,
                "breakeven_armed": False,
                "trail_armed": False,
                "predicted_mfe_pct": 0.01,
                "predicted_mae_pct": 0.001,
                "predicted_peak_bar": 40,
                "bar_interval_ms": 3600000,
            },
        },
        "equity": 20800.0,
        "cash": 20800.0,
        "open_pnl": 0.0,
        "peak_equity": 21000.0,
        "pnl_24h": -25.0,
        "last_heartbeat": 1780322825000,
        "bot_started_at": 1780318800000,
        "risk": {"trading_halted": False, "consecutive_losses": 0},
    }
    state_path.write_text(json.dumps(state))
    return db_path, state_path


class TestPositionDisplay:
    def test_metal_renders_ig_spot_level_directly(
        self, fake_environment: tuple[Path, Path]
    ) -> None:
        """IG-native metals (XAU/USD) carry a display divisor of 1.0, so the
        dashboard shows the IG spot level directly (no ETF-style conversion)."""
        db_path, state_path = fake_environment
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xau = next(p for p in snap["positions"] if p["symbol"] == "XAU/USD")
        # Entry: 8726.2 / 1.0 = 8726.2
        assert xau["entry_display"] == pytest.approx(8726.2)
        # Current: live state IG level 8855.0 / 1.0 = 8855.0
        assert xau["current_display"] == pytest.approx(8855.0)

    def test_current_price_prefers_live_state_over_stale_candle(
        self, fake_environment: tuple[Path, Path]
    ) -> None:
        """The dashboard must use the heartbeat-refreshed live IG level in state,
        not the last closed hourly candle (8855.0 for XAU/USD) — that staleness is
        exactly what made per-position P&L disagree with the live aggregate."""
        db_path, state_path = fake_environment
        # Rewrite XAU/USD's live price to something distinct from both entry and the
        # candle so the source is unambiguous.
        state = json.loads(state_path.read_text())
        state["positions"]["XAU/USD"]["current_price"] = 8900.0
        state_path.write_text(json.dumps(state))

        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xau = next(p for p in snap["positions"] if p["symbol"] == "XAU/USD")
        assert xau["current_display"] == pytest.approx(8900.0)  # from state, not 8855.0
        # P&L% derives from the live level: (8900 - 8726.2) / 8726.2.
        assert xau["unrealised_pnl_pct"] == pytest.approx((8900.0 - 8726.2) / 8726.2 * 100)

    def test_current_price_falls_back_to_candle_when_no_live_price(
        self, fake_environment: tuple[Path, Path]
    ) -> None:
        """If no live price has been written yet (current_price == 0), the
        dashboard falls back to the last closed candle so the column isn't blank."""
        db_path, state_path = fake_environment
        state = json.loads(state_path.read_text())
        state["positions"]["XAU/USD"]["current_price"] = 0.0
        state_path.write_text(json.dumps(state))

        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xau = next(p for p in snap["positions"] if p["symbol"] == "XAU/USD")
        # Candle close 8855.0 → /1.0 = 8855.0.
        assert xau["current_display"] == pytest.approx(8855.0)

    def test_xag_display_matches_ig_level(self, fake_environment: tuple[Path, Path]) -> None:
        """XAG/USD carries an explicit ``_DISPLAY_DIVISOR`` entry of 1.0 that
        matches ``ig_quote_scale("XAG/USD") == 1.0``, so the dashboard shows
        the IG spot level whether or not the override is consulted.  Guards
        a real bug class — a symbol silently dropped from a display table
        rendering a wrong number."""
        db_path, state_path = fake_environment
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xag = next(p for p in snap["positions"] if p["symbol"] == "XAG/USD")
        assert xag["entry_display"] == pytest.approx(10321.6)
        assert xag["current_display"] == pytest.approx(10325.0)


class TestPositionStopColumn:
    def test_initial_stop_derived_from_entry_stop_pct(
        self, fake_environment: tuple[Path, Path]
    ) -> None:
        """No trailing stop armed yet → ``stop_display`` is the initial stop
        derived from ``entry_price × (1 − entry_stop_pct)``, displayed in
        the same units as ``entry_display``."""
        db_path, state_path = fake_environment
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xag = next(p for p in snap["positions"] if p["symbol"] == "XAG/USD")
        # Initial stop level = 10321.6 × (1 - 0.0177) = 10138.91
        assert xag["stop_pct"] == pytest.approx(0.0177)
        assert xag["stop_display"] == pytest.approx(10138.908, abs=0.01)

    def test_metal_stop_uses_display_divisor(self, fake_environment: tuple[Path, Path]) -> None:
        """XAU/USD stop level must flow through the same display divisor so the
        operator can compare entry/current/stop visually."""
        db_path, state_path = fake_environment
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xau = next(p for p in snap["positions"] if p["symbol"] == "XAU/USD")
        # Initial stop level (IG) = 8726.2 × (1 - 0.05) = 8289.89
        # Display = 8289.89 / 1.0 = 8289.89 (metal divisor 1.0)
        assert xau["stop_pct"] == pytest.approx(0.05)
        assert xau["stop_display"] == pytest.approx(8289.89, abs=0.01)

    def test_trailing_stop_overrides_initial_when_armed(self, tmp_path: Path) -> None:
        """Once ``current_trailing_stop`` is set, the dashboard reports the
        trailing stop and derives ``stop_pct`` from it — the operator sees
        the *active* exit, not the never-relevant initial."""
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE candles (symbol TEXT, timestamp INTEGER, open REAL,
                high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (symbol, timestamp));
            CREATE TABLE signal_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
                scored_at INTEGER NOT NULL, symbol TEXT NOT NULL,
                horizon_bars INTEGER NOT NULL,
                mean_return REAL, direction_confidence REAL,
                uncertainty REAL, entry_price REAL,
                realized_return_at_horizon REAL, gap_spanned INTEGER DEFAULT 0);
            CREATE TABLE asset_correlations (id INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at INTEGER, symbol_a TEXT, symbol_b TEXT,
                correlation REAL);
            CREATE TABLE macro_features (series_id TEXT,
                observation_date INTEGER, value REAL, fetched_at INTEGER,
                PRIMARY KEY (series_id, observation_date));
            CREATE TABLE sentiment_scores (scored_at INTEGER, asset TEXT,
                sentiment REAL, confidence REAL, agreement REAL,
                sources TEXT, escalated INTEGER, PRIMARY KEY (scored_at, asset));
            """
        )
        conn.commit()
        conn.close()
        state = {
            "positions": {
                "XAG/USD": {
                    "symbol": "CS.D.USCSI.TODAY.IP",
                    "side": "buy",
                    "entry_price": 10321.6,
                    "quantity": 1.08,
                    "current_price": 10500.0,
                    "unrealised_pnl": 0.0,
                    "realised_pnl": 0.0,
                    "opened_at": 1780322825000,
                    "updated_at": 1780322825000,
                }
            },
            "take_profit_state": {
                "XAG/USD": {
                    "symbol": "XAG/USD",
                    "entry_price": 10321.6,
                    "entry_stop_pct": 0.0177,
                    "current_trailing_stop": 10400.0,  # trail armed at +0.76 %
                    "breakeven_armed": True,
                    "trail_armed": True,
                }
            },
            "equity": 20800.0,
            "cash": 20800.0,
            "open_pnl": 0.0,
            "peak_equity": 21000.0,
            "pnl_24h": 0.0,
            "last_heartbeat": 1780322825000,
            "bot_started_at": 1780318800000,
            "risk": {"trading_halted": False, "consecutive_losses": 0},
        }
        state_path.write_text(json.dumps(state))
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        xag = next(p for p in snap["positions"] if p["symbol"] == "XAG/USD")
        # Trailing stop overrides → 10400.0 is reported as the active stop.
        assert xag["stop_display"] == pytest.approx(10400.0)
        # stop_pct derived from (entry − trail) / entry = (10321.6 − 10400.0) / 10321.6
        # which is negative (stop is above entry = profitable trail).  The
        # dashboard reports the signed value so the column can carry the
        # right sign visually.
        assert xag["stop_pct"] == pytest.approx((10321.6 - 10400.0) / 10321.6)


# ---------------------------------------------------------------------------
# Prediction-vs-realised visualisations (borrowed from the Kronos web UI):
# predicted_close_path chart, calibration scatter, resolved-signal error table.
# ---------------------------------------------------------------------------

_FULL_SIGNAL_HISTORY_SCHEMA = """
CREATE TABLE candles (
    symbol TEXT, timestamp INTEGER, open REAL, high REAL,
    low REAL, close REAL, volume REAL, PRIMARY KEY (symbol, timestamp)
);
CREATE TABLE signal_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_at INTEGER NOT NULL, symbol TEXT NOT NULL, horizon_bars INTEGER NOT NULL,
    mean_return REAL, direction_confidence REAL, uncertainty REAL,
    predicted_mfe_pct REAL, predicted_mae_pct REAL, predicted_volatility REAL,
    monotonicity REAL, entry_price REAL NOT NULL,
    realized_return_at_horizon REAL, realized_max_high_pct REAL, realized_min_low_pct REAL,
    gap_spanned INTEGER DEFAULT 0, predicted_close_path BLOB
);
CREATE TABLE asset_correlations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at INTEGER, symbol_a TEXT, symbol_b TEXT, correlation REAL
);
CREATE TABLE macro_features (
    series_id TEXT, observation_date INTEGER, value REAL, fetched_at INTEGER,
    PRIMARY KEY (series_id, observation_date)
);
CREATE TABLE sentiment_scores (
    scored_at INTEGER, asset TEXT, sentiment REAL, confidence REAL,
    agreement REAL, sources TEXT, escalated INTEGER, PRIMARY KEY (scored_at, asset)
);
"""

_MINIMAL_STATE = {
    "positions": {},
    "equity": 20000.0,
    "cash": 20000.0,
    "open_pnl": 0.0,
    "peak_equity": 20000.0,
    "pnl_24h": 0.0,
    "last_heartbeat": 1780322825000,
    "bot_started_at": 1780318800000,
    "risk": {"trading_halted": False, "consecutive_losses": 0},
}


def _path_blob(closes: list[float]) -> bytes:
    """Encode a predicted close path the way stores it (LE float32)."""
    import array

    arr = array.array("f", closes)
    if sys.byteorder == "big":  # pragma: no cover - CI is little-endian
        arr.byteswap()
    return arr.tobytes()


@pytest.fixture()
def resolved_env(tmp_path: Path) -> tuple[Path, Path]:
    """A DB with three resolved signal_history rows, the newest carrying a path."""
    db_path = tmp_path / "candles.db"
    state_path = tmp_path / "bot_state.json"
    conn = sqlite3.connect(db_path)
    conn.executescript(_FULL_SIGNAL_HISTORY_SCHEMA)
    # entry 100 → path rises to 103 (a +3 % predicted close at horizon).
    path = _path_blob([100.0, 101.0, 102.5, 103.0])
    conn.executemany(
        "INSERT INTO signal_history (scored_at, symbol, horizon_bars, mean_return, "
        "direction_confidence, uncertainty, entry_price, realized_return_at_horizon, "
        "realized_max_high_pct, realized_min_low_pct, gap_spanned, predicted_close_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # newest — predicted +3 %, realised +2 %, direction hit, has path
            (3000, "EUR/USD", 120, 0.03, 0.8, 1.0, 100.0, 0.02, 0.04, 0.01, 0, path),
            # middle — predicted +1 %, realised −0.5 %, direction MISS, no path
            (2000, "GBP/USD", 120, 0.01, 0.7, 1.5, 1.2, -0.005, 0.006, 0.012, 0, None),
            # oldest — predicted −2 %, realised −3 %, direction hit, gap-spanned
            (1000, "XAU/USD", 120, -0.02, 0.75, 2.0, 2000.0, -0.03, 0.005, 0.04, 1, None),
        ],
    )
    conn.commit()
    conn.close()
    state_path.write_text(json.dumps(_MINIMAL_STATE))
    return db_path, state_path


class TestPredictionPathChart:
    def test_uses_newest_resolved_row_with_a_path(self, resolved_env: tuple[Path, Path]) -> None:
        db_path, state_path = resolved_env
        pp = DashboardData(db_path=db_path, state_path=state_path).prediction_vs_realized_path()
        assert pp is not None
        assert pp["symbol"] == "EUR/USD"
        # Last path point: (103 − 100) / 100 = +3 %.
        assert pp["pred_return"] == pytest.approx(3.0)
        assert pp["realized_return"] == pytest.approx(2.0)
        assert pp["error"] == pytest.approx(-1.0)  # realised − predicted
        # Four close points → four "x,y" pairs in the polyline.
        assert len(pp["pred_points"].split()) == 4

    def test_band_spans_realised_high_low_envelope(self, resolved_env: tuple[Path, Path]) -> None:
        db_path, state_path = resolved_env
        pp = DashboardData(db_path=db_path, state_path=state_path).prediction_vs_realized_path()
        assert pp is not None
        assert pp["band"]["hi"] == pytest.approx(4.0)  # realized_max_high_pct 0.04
        assert pp["band"]["lo"] == pytest.approx(-1.0)  # −realized_min_low_pct 0.01
        # Higher % maps to a smaller SVG y, so the high edge sits above the low.
        assert pp["band"]["y_hi"] < pp["band"]["y_lo"]

    def test_none_when_no_resolved_path(self, tmp_path: Path) -> None:
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(_FULL_SIGNAL_HISTORY_SCHEMA)
        # Unresolved row only (realized_return_at_horizon NULL).
        conn.execute(
            "INSERT INTO signal_history (scored_at, symbol, horizon_bars, mean_return, "
            "entry_price) VALUES (4000, 'EUR/USD', 120, 0.03, 100.0)"
        )
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))
        assert (
            DashboardData(db_path=db_path, state_path=state_path).prediction_vs_realized_path()
            is None
        )


class TestCalibrationScatter:
    def test_points_and_quintiles(self, resolved_env: tuple[Path, Path]) -> None:
        db_path, state_path = resolved_env
        cal = DashboardData(db_path=db_path, state_path=state_path).calibration_scatter()
        assert cal is not None
        # Three non-gap-spanned resolved rows? XAU/USD is gap_spanned → excluded.
        assert cal["sample_size"] == 2
        assert len(cal["points"]) == 2
        # axis_max ≥ largest |value|: realised −0.5 %, +2 %, predicted +1 %, +3 %.
        assert cal["axis_max"] == pytest.approx(3.0)
        # Quintile polyline present (≤5 points; here 2 buckets are non-empty).
        assert cal["quintile_points"]

    def test_point_direction_colour_flag(self, resolved_env: tuple[Path, Path]) -> None:
        db_path, state_path = resolved_env
        cal = DashboardData(db_path=db_path, state_path=state_path).calibration_scatter()
        assert cal is not None
        # EUR/USD pred +3 % / realised +2 % → both positive → correct.
        # GBP/USD pred +1 % / realised −0.5 % → sign mismatch → incorrect.
        flags = sorted(p["correct"] for p in cal["points"])
        assert flags == [False, True]


class TestResolvedSignalsTable:
    def test_orders_newest_first_with_error_and_hit(self, resolved_env: tuple[Path, Path]) -> None:
        db_path, state_path = resolved_env
        rows = DashboardData(db_path=db_path, state_path=state_path).resolved_signals()
        assert [r["symbol"] for r in rows] == ["EUR/USD", "GBP/USD", "XAU/USD"]
        eur = rows[0]
        assert eur["error"] == pytest.approx(0.02 - 0.03)
        assert eur["correct"] is True
        gbp = rows[1]
        assert gbp["correct"] is False  # +1 % predicted, −0.5 % realised
        xau = rows[2]
        assert xau["gap_spanned"] == 1  # flagged, but still listed

    def test_snapshot_exposes_all_three_keys(self, resolved_env: tuple[Path, Path]) -> None:
        db_path, state_path = resolved_env
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        assert snap["prediction_path"] is not None
        assert snap["calibration"] is not None
        assert len(snap["resolved"]) == 3


class TestNoResolvedRowsGuard:
    def test_minimal_schema_with_no_signals_returns_empty(self, tmp_path: Path) -> None:
        """The cheap MAX(scored_at) guard must short-circuit before the
        column-heavy SELECTs touch predicted_* columns a minimal DB lacks."""
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        # Deliberately minimal: NO predicted_close_path / predicted_* columns.
        conn.executescript(
            """
            CREATE TABLE candles (symbol TEXT, timestamp INTEGER, open REAL, high REAL,
                low REAL, close REAL, volume REAL, PRIMARY KEY (symbol, timestamp));
            CREATE TABLE signal_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
                scored_at INTEGER NOT NULL, symbol TEXT NOT NULL, horizon_bars INTEGER NOT NULL,
                mean_return REAL, direction_confidence REAL, uncertainty REAL, entry_price REAL,
                realized_return_at_horizon REAL, gap_spanned INTEGER DEFAULT 0);
            CREATE TABLE asset_correlations (id INTEGER PRIMARY KEY AUTOINCREMENT,
                computed_at INTEGER, symbol_a TEXT, symbol_b TEXT, correlation REAL);
            CREATE TABLE macro_features (series_id TEXT, observation_date INTEGER,
                value REAL, fetched_at INTEGER, PRIMARY KEY (series_id, observation_date));
            CREATE TABLE sentiment_scores (scored_at INTEGER, asset TEXT, sentiment REAL,
                confidence REAL, agreement REAL, sources TEXT, escalated INTEGER,
                PRIMARY KEY (scored_at, asset));
            """
        )
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))
        data = DashboardData(db_path=db_path, state_path=state_path)
        assert data.prediction_vs_realized_path() is None
        assert data.calibration_scatter() is None
        assert data.resolved_signals() == []


_MINIMAL_SCHEMA = """
    CREATE TABLE candles (symbol TEXT, timestamp INTEGER, open REAL, high REAL,
        low REAL, close REAL, volume REAL, PRIMARY KEY (symbol, timestamp));
    CREATE TABLE signal_history (id INTEGER PRIMARY KEY AUTOINCREMENT,
        scored_at INTEGER NOT NULL, symbol TEXT NOT NULL, horizon_bars INTEGER NOT NULL,
        mean_return REAL, direction_confidence REAL, uncertainty REAL, entry_price REAL,
        realized_return_at_horizon REAL, gap_spanned INTEGER DEFAULT 0);
    CREATE TABLE asset_correlations (id INTEGER PRIMARY KEY AUTOINCREMENT,
        computed_at INTEGER, symbol_a TEXT, symbol_b TEXT, correlation REAL);
    CREATE TABLE macro_features (series_id TEXT, observation_date INTEGER,
        value REAL, fetched_at INTEGER, PRIMARY KEY (series_id, observation_date));
    CREATE TABLE sentiment_scores (scored_at INTEGER, asset TEXT, sentiment REAL,
        confidence REAL, agreement REAL, sources TEXT, escalated INTEGER,
        PRIMARY KEY (scored_at, asset));
"""


class TestRiskUtilization:
    def test_computes_live_risk_against_caps(self, fake_environment: tuple[Path, Path]) -> None:
        """Total risk-on sums quantity x distance-to-effective-stop per position
        (mirrors RiskBudgetLedger.live_risk_gbp) against RiskConfig's caps."""
        db_path, state_path = fake_environment
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        ru = snap["risk_utilization"]
        assert ru["open_positions"] == 2
        assert ru["max_open_positions"] == 8
        assert ru["open_pct_of_cap"] == pytest.approx(25.0)
        # XAU: 0.06 x (8726.2 - 8289.89) = 26.1786
        # XAG: 1.08 x (10321.6 - 10138.908) = 197.30736
        assert ru["total_risk_gbp"] == pytest.approx(223.48596, abs=0.01)
        assert ru["max_total_risk_pct"] == pytest.approx(0.05)
        assert ru["total_risk_pct"] == pytest.approx(223.48596 / 20800.0, abs=1e-6)
        assert ru["risk_pct_of_cap"] == pytest.approx(ru["total_risk_pct"] * 100 / 0.05, abs=1e-3)

    def test_no_open_positions_is_zero_not_error(self, tmp_path: Path) -> None:
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(_MINIMAL_SCHEMA)
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        ru = snap["risk_utilization"]
        assert ru["open_positions"] == 0
        assert ru["total_risk_gbp"] == 0.0
        assert ru["total_risk_pct"] == 0.0
        assert ru["open_pct_of_cap"] == 0.0


class TestFeedFreshness:
    def test_groups_by_ig_native_metals_vs_eodhd(self, tmp_path: Path) -> None:
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(_MINIMAL_SCHEMA)
        now_ms = int(time.time() * 1000)
        # EUR/USD (EODHD, FX) fresh; XAU/USD (IG-native metal) stale (> 3h old).
        conn.execute(
            "INSERT INTO candles VALUES ('EUR/USD', ?, 1.08, 1.09, 1.07, 1.085, 0.0)",
            (now_ms - 5 * 60_000,),
        )
        conn.execute(
            "INSERT INTO candles VALUES ('XAU/USD', ?, 2400, 2410, 2390, 2405, 0.0)",
            (now_ms - 4 * 3_600_000,),
        )
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))

        data = DashboardData(db_path=db_path, state_path=state_path)
        groups = {g["label"]: g for g in data.feed_freshness()}
        eodhd = groups["EODHD (FX + US shares)"]
        assert eodhd["age_s"] == pytest.approx(300, abs=5)
        assert eodhd["stale"] is False

        metals = groups["IG-native metals (XAU/XAG)"]
        assert metals["age_s"] == pytest.approx(4 * 3_600, abs=5)
        assert metals["stale"] is True

    def test_no_candles_for_a_source_reports_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(_MINIMAL_SCHEMA)
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))

        data = DashboardData(db_path=db_path, state_path=state_path)
        for g in data.feed_freshness():
            assert g["latest_ms"] is None
            assert g["age_s"] is None
            assert g["stale"] is False

    def test_twelvedata_failover_relabels_the_non_metal_group(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(_MINIMAL_SCHEMA)
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))

        monkeypatch.setenv("CANDLE_EXCHANGE", "twelvedata")
        data = DashboardData(db_path=db_path, state_path=state_path)
        labels = {g["label"] for g in data.feed_freshness()}
        assert "TwelveData (failover active)" in labels
        assert "EODHD (FX + US shares)" not in labels


class TestNextRerankCountdown:
    def test_next_rerank_at_persists_after_in_progress_clears(self, tmp_path: Path) -> None:
        """RerankStatusWriter merges fields into one cached payload, so
        next_rerank_at set at the end of a rerank survives in_progress
        flipping back to false — the always-on header countdown depends on
        reading the file without rerank_status()'s in_progress gate."""
        db_path = tmp_path / "candles.db"
        state_path = tmp_path / "bot_state.json"
        conn = sqlite3.connect(db_path)
        conn.executescript(_MINIMAL_SCHEMA)
        conn.commit()
        conn.close()
        state_path.write_text(json.dumps(_MINIMAL_STATE))

        next_rerank_at_s = time.time() + 1800  # 30 min from now
        rerank_status_path = state_path.parent / "rerank_status.json"
        rerank_status_path.write_text(
            json.dumps({"in_progress": False, "phase": "idle", "next_rerank_at": next_rerank_at_s})
        )

        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        # Progress banner stays hidden between reranks...
        assert snap["rerank_progress"] is None
        # ...but the countdown field is still populated.
        assert snap["service"]["next_rerank_at_ms"] == pytest.approx(
            next_rerank_at_s * 1000, abs=1000
        )

    def test_missing_rerank_status_file_yields_none(
        self, fake_environment: tuple[Path, Path]
    ) -> None:
        db_path, state_path = fake_environment
        snap = DashboardData(db_path=db_path, state_path=state_path).snapshot()
        assert snap["service"]["next_rerank_at_ms"] is None
