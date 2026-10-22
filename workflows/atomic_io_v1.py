"""Atomic project JSON writes with bounded Windows sharing-error recovery."""
import time

from workflows import reservoir_v2 as native


def write(path, value):
    for attempt in range(10):
        try: return native.write(path, value)
        except OSError as exc:
            if getattr(exc, "winerror", None) not in (5, 32) or attempt == 9:
                raise
            time.sleep(.1)
