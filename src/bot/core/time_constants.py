"""Shared millisecond-duration constants.

Extracted from the several modules that each independently redeclared these
(``_HOUR_MS`` alone appeared identically in 7+ files) — a single source
means a typo in one no longer silently diverges from the rest.
"""

from __future__ import annotations

HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
WEEK_MS = 7 * DAY_MS
MONTH_MS = 30 * DAY_MS
