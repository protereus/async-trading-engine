"""Offline analysis modules — not loaded by the running bot.

Each module here is a pure-Python analysis that reads from ``signal_history``
or related tables and produces a report.  Designed to be re-runnable from
``scripts/`` CLI wrappers or imported into a notebook.
"""
