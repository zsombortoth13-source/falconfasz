"""Tiny standalone module holding the run_captured_live() lock registry.

This has to live in a properly `import`-ed module, NOT as a top-level global in
turtle_ui.py itself -- Streamlit re-execs the main script's module namespace fresh
on every rerun (confirmed empirically: id(_run_captured_live_locks) differed between
two successive button clicks in the same process, even guarded by
`if "x" not in globals()`), so a plain global there can never actually persist across
reruns. A separately imported module is cached in sys.modules and reused across
reruns within the same process, same as td/opt/wf already rely on for their own
snapshot/restore state.
"""
import threading

_locks = {}


def get_lock(run_key: str) -> threading.Lock:
    return _locks.setdefault(run_key, threading.Lock())
