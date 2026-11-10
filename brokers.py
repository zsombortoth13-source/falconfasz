"""Broker profiles: every broker-specific number the backtests use, in one JSON file per broker.

A profile describes what a broker charges and requires; compile_profile() turns it, plus the
choices made in the Fees tab (contract size, margin calls, edits), into the settings the engine
reads (turtle_dual_system's USE_PER_CONTRACT_FEES, FEE_SCHEDULE, ASSET_COSTS, SIZE_STEP,
MARGIN_SPEC, ...). Adding a broker means adding a JSON file to the brokers/ folder -- or editing the
numbers in the Fees tab and saving them as a new profile. No code change.

Profile layout (see brokers/ibkr.json and brokers/tmgm.json):
    name, kind ("futures" or "cfd"), currency, notes
    defaults: slippage_pct, spread_pct, commission_pct, holding_pct   (% per side / % per year)
              futures: margin_maintenance_ratio    cfd: stop_out_pct, leverage
    instruments: {asset: ...}
        futures: {"micro": CONTRACT, "standard": CONTRACT}   (micro optional)
            CONTRACT = code, multiplier, commission, exchange, regulatory (all $ per contract per
                       side), margin_pct (initial margin, % of contract value), verified
                       optional per-asset spread_pct / slippage_pct, optional fee_per_side override,
                       optional maintenance_pct (the broker's maintenance margin, % of contract value;
                       without it margin_maintenance_ratio times the initial margin), optional
                       margin_snapshot (the dollar margins and price the percentages came from)
        cfd: code, lot_step (units of the asset), leverage, spread_pct, commission_pct (per side),
             holding_pct (swap or admin fee, % per year), verified; optional slippage_pct
             (leverage 1 describes a fully paid spot instrument); optional maintenance_pct (the broker's
             maintenance margin as % of position value; without it the stop-out % times the initial margin)
        either kind, all optional and informational: currency (of the fees, when not the profile's),
             source (where the numbers come from), estimated_fields (numbers in an otherwise
             verified instrument that the source does not give)

Every number carries a `verified` flag. False means ESTIMATE -- plausible, not confirmed against the
broker's own pages for your account; the Fees tab flags those. Fees are kept in the currency of the
contract's price series, because the engine treats all prices as one currency.
"""
import copy
import json
import os
import re

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brokers")

SIZING_FRACTIONAL = "fractional"
SIZING_MICRO = "micro"
SIZING_MINI = "mini"
SIZING_CFD = "cfd"


def list_profiles() -> dict:
    """{profile name: file path} for every JSON profile in the brokers/ folder."""
    found = {}
    if os.path.isdir(PROFILE_DIR):
        for file in sorted(os.listdir(PROFILE_DIR)):
            if file.endswith(".json"):
                path = os.path.join(PROFILE_DIR, file)
                try:
                    found[load(path)["name"]] = path
                except (OSError, ValueError, KeyError):
                    continue
    return found


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(profile: dict, name: str) -> str:
    """Writes the profile under a new name and returns the file path."""
    profile = dict(profile, name=name)
    os.makedirs(PROFILE_DIR, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "profile"
    path = os.path.join(PROFILE_DIR, f"{slug}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    return path


def sizing_options(profile: dict) -> list:
    """Sizing modes that make sense for this kind of broker, first = default."""
    if profile["kind"] == "futures":
        return [SIZING_MICRO, SIZING_MINI, SIZING_FRACTIONAL]
    return [SIZING_CFD, SIZING_FRACTIONAL]


def _contract(profile: dict, asset: str, sizing: str):
    """The futures contract spec used for `asset` under this sizing mode, or None if the profile
    has no instrument for it. Mini and fractional use the standard / micro contract respectively."""
    options = profile["instruments"].get(asset)
    if not options:
        return None
    if sizing in (SIZING_MICRO, SIZING_FRACTIONAL):
        return options.get("micro") or options.get("standard")
    return options.get("standard") or options.get("micro")


def fee_per_side(spec: dict) -> float:
    if spec.get("fee_per_side") is not None:
        return float(spec["fee_per_side"])
    return float(spec["commission"]) + float(spec["exchange"]) + float(spec.get("regulatory", 0.0))


def _default(profile: dict, key: str, fallback: float = 0.0) -> float:
    return float(profile.get("defaults", {}).get(key, fallback))


# ----------------------------------------------------------------------------
# editable table
# ----------------------------------------------------------------------------
def table_rows(profile: dict, assets, sizing: str) -> list:
    """One row per asset the profile covers, with the numbers that matter, for the Fees tab editor."""
    rows = []
    for asset in assets:
        if profile["kind"] == "futures":
            spec = _contract(profile, asset, sizing)
            if spec is None:
                continue
            rows.append({
                "Asset": asset, "Contract": spec["code"], "Multiplier": float(spec["multiplier"]),
                "Fee per side ($)": round(fee_per_side(spec), 4),
                "Initial margin (%)": float(spec["margin_pct"]),
                "Spread (%)": float(spec.get("spread_pct", _default(profile, "spread_pct"))),
                "Slippage (%)": float(spec.get("slippage_pct", _default(profile, "slippage_pct"))),
                "Verified": bool(spec.get("verified", False)),
            })
        else:
            spec = profile["instruments"].get(asset)
            if spec is None:
                continue
            rows.append({
                "Asset": asset, "Symbol": spec["code"], "Lot step (units)": float(spec["lot_step"]),
                "Spread (%)": float(spec.get("spread_pct", _default(profile, "spread_pct"))),
                "Slippage (%)": float(spec.get("slippage_pct", _default(profile, "slippage_pct"))),
                "Commission (% per side)": float(spec.get("commission_pct", _default(profile, "commission_pct"))),
                "Swap / holding (% per year)": float(spec.get("holding_pct", _default(profile, "holding_pct"))),
                "Leverage (x)": float(spec.get("leverage", _default(profile, "leverage", 20.0))),
                "Verified": bool(spec.get("verified", False)),
            })
    return rows


_FUTURES_COLUMNS = {"Multiplier": "multiplier", "Fee per side ($)": "fee_per_side", "Initial margin (%)": "margin_pct",
                    "Spread (%)": "spread_pct", "Slippage (%)": "slippage_pct"}
_CFD_COLUMNS = {"Lot step (units)": "lot_step", "Spread (%)": "spread_pct", "Slippage (%)": "slippage_pct",
                "Commission (% per side)": "commission_pct", "Swap / holding (% per year)": "holding_pct",
                "Leverage (x)": "leverage"}


def apply_edits(profile: dict, assets, sizing: str, edited_rows: dict) -> dict:
    """A copy of the profile with the edits made in the table (st.data_editor's edited_rows, keyed by
    row position) written back. An edited number is no longer a verified one."""
    if not edited_rows:
        return profile
    profile = copy.deepcopy(profile)
    rows = table_rows(profile, assets, sizing)
    columns = _FUTURES_COLUMNS if profile["kind"] == "futures" else _CFD_COLUMNS
    for idx, changes in edited_rows.items():
        if int(idx) >= len(rows):
            continue
        asset = rows[int(idx)]["Asset"]
        spec = (_contract(profile, asset, sizing) if profile["kind"] == "futures"
                else profile["instruments"][asset])
        for column, value in changes.items():
            if column in columns and value is not None:
                old_initial = spec.get("margin_pct")
                spec[columns[column]] = float(value)
                spec["verified"] = False
                if columns[column] == "leverage":
                    spec.pop("maintenance_pct", None)         # a stated maintenance rate no longer matches
                elif columns[column] == "margin_pct" and spec.get("maintenance_pct") is not None and old_initial:
                    spec["maintenance_pct"] = float(spec["maintenance_pct"]) * float(value) / float(old_initial)
    return profile


# ----------------------------------------------------------------------------
# profile -> engine settings
# ----------------------------------------------------------------------------
def compile_profile(profile: dict, assets, sizing: str, margin_calls: bool = False,
                    margin_stress: float = 1.0, stop_out_pct: float = None) -> dict:
    """Engine settings for this profile and choices. Returns {"engine": {global name: value},
    "notes": [...], "estimates": [...]}; the caller sets each engine entry on turtle_dual_system."""
    engine = {
        "SLIPPAGE_PCT": _default(profile, "slippage_pct"), "SPREAD_PCT": _default(profile, "spread_pct"),
        "COMMISSION_PCT": _default(profile, "commission_pct"), "HOLDING_COST_PCT_PER_YEAR": _default(profile, "holding_pct"),
        "USE_PER_CONTRACT_FEES": False, "FEE_SCHEDULE": {}, "ASSET_COSTS": {}, "SIZE_STEP": {},
        "USE_MARGIN_CALLS": bool(margin_calls), "MARGIN_SPEC": {},
    }
    notes, estimates, missing, partial, unlevered = [], [], [], [], []
    futures = profile["kind"] == "futures"
    if futures:
        engine["USE_PER_CONTRACT_FEES"] = True
        ratio = _default(profile, "margin_maintenance_ratio", 0.9)
    else:
        stop_out = (stop_out_pct if stop_out_pct is not None else _default(profile, "stop_out_pct", 50.0)) / 100.0

    for asset in assets:
        if futures:
            spec = _contract(profile, asset, sizing)
            if spec is None:
                missing.append(asset)
                continue
            engine["FEE_SCHEDULE"][asset] = {"multiplier": float(spec["multiplier"]),
                                             "fee_per_contract": fee_per_side(spec), "code": spec["code"]}
            if sizing in (SIZING_MICRO, SIZING_MINI):
                engine["SIZE_STEP"][asset] = float(spec["multiplier"])
            if sizing == SIZING_MICRO and not profile["instruments"][asset].get("micro"):
                notes.append(f"{asset}: no micro contract, standard {spec['code']} used")
            initial = min(1.0, float(spec["margin_pct"]) / 100.0 * margin_stress)
            if spec.get("maintenance_pct") is not None:       # the broker's own maintenance rate for this contract
                maint = min(initial, float(spec["maintenance_pct"]) / 100.0 * margin_stress)
            else:
                maint = initial * ratio
            costs = {k: spec[k] for k in ("spread_pct", "slippage_pct") if k in spec}
        else:
            spec = profile["instruments"].get(asset)
            if spec is None:
                missing.append(asset)
                continue
            if sizing == SIZING_CFD:
                engine["SIZE_STEP"][asset] = float(spec["lot_step"])
            initial = min(1.0, margin_stress / max(float(spec.get("leverage", _default(profile, "leverage", 20.0))), 1.0))
            if spec.get("maintenance_pct") is not None:       # the broker states the maintenance rate itself
                maint = min(initial, float(spec["maintenance_pct"]) / 100.0 * margin_stress)
            else:
                maint = initial * stop_out
            costs = {k: spec[k] for k in ("spread_pct", "slippage_pct", "commission_pct", "holding_pct") if k in spec}
        if costs:
            engine["ASSET_COSTS"][asset] = {k: float(v) for k, v in costs.items()}
        engine["MARGIN_SPEC"][asset] = (initial, maint)
        if not spec.get("verified", False):
            estimates.append(asset)
        elif spec.get("estimated_fields"):
            partial.append(f"{asset} ({', '.join(spec['estimated_fields'])})")
        if not futures and initial >= 1.0:
            unlevered.append(asset)

    if unlevered:
        notes.append(f"{', '.join(unlevered)}: fully paid, no leverage. Keep total exposure at or below 100% "
                     "(switch margin calls on to enforce it).")
    if missing:
        notes.append("Not in this profile, so traded with the flat default costs and no whole-size "
                     "rounding: " + ", ".join(missing))
    if estimates:
        notes.append(f"{len(estimates)} instrument(s) use ESTIMATED numbers, not confirmed against the "
                     f"broker: {', '.join(estimates)}")
    if partial:
        notes.append("Fees are from the broker's document, but these numbers in the same instrument are "
                     f"estimates: {'; '.join(partial)}")
    return {"engine": engine, "notes": notes, "estimates": estimates}
