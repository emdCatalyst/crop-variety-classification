"""Application-wide rate limiter (slowapi).

The limiter is keyed by client IP. Routes pick limits via the @limiter.limit
decorator; the global handler is registered in app/main.py so a 429 returns
JSON instead of HTML.
"""
from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=[])
