"""Read-only FastAPI dashboard for the trading bot.

Reads ``bot_state.json`` (atomic, last-write-wins from the bot's StateManager)
and ``candles.db`` (SQLite WAL, read-only ``mode=ro`` connection) — never
mutates either.  Runs as a separate systemd service on 127.0.0.1:8080 behind
SSH port forwarding.
"""
