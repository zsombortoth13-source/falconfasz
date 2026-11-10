"""Position sizing and margin settings for the Turtle backtests.

Three ways to size and fund positions (chosen in the Fees tab):
    fractional  the engine's original behavior: any size, no margin (unless you switch margin calls on)
    micro       whole micro futures contracts (MES, MNQ, MGC, ...); an asset with no micro contract
                uses its standard contract
    mini        whole standard / e-mini futures contracts (ES, NQ, GC, ...)
    cfd         fractional lots in steps of `cfd_step` units, funded by leverage

Margin calls use a daily-close check in turtle_dual_system._enforce_margin. Rates here are the
fraction of a position's CURRENT notional that must be held as margin.

DATA PROVENANCE -- READ THIS
    The futures margin rates below are ESTIMATES of typical overnight exchange margin as a share of
    contract value, NOT figures taken from your broker. Real margins are set in dollars per contract,
    differ by broker, and are raised sharply in volatile markets -- exactly when a margin call bites.
    Edit them in the Fees tab, and use the stress multiplier to test higher margins.
    CFD leverage presets follow the retail caps (ESMA / ASIC) and a generic offshore 1:100 figure; your
    broker and account type decide the real number.
"""
import ibkr_fees

SIZING_FRACTIONAL = "Fractional (any size, the engine's original)"
SIZING_MICRO = "Micro futures (whole contracts)"
SIZING_MINI = "Mini / standard futures (whole contracts)"
SIZING_CFD = "CFD (fractional lots)"
SIZING_MODES = [SIZING_FRACTIONAL, SIZING_MICRO, SIZING_MINI, SIZING_CFD]

# initial margin as a fraction of contract notional -- ESTIMATES, see the module docstring
FUTURES_INITIAL_MARGIN = {
    "SP500": 0.085, "Nasdaq": 0.075, "Russell2000": 0.07, "Dow": 0.075, "Gold": 0.05, "Silver": 0.09,
    "Copper": 0.065, "CrudeOil": 0.10, "Platinum": 0.08, "Palladium": 0.10, "Corn": 0.055, "Cocoa": 0.08,
    "Bitcoin": 0.35, "Ger40": 0.05, "Nikkei225": 0.05,
}
MAINTENANCE_RATIO = 0.9   # exchange maintenance margin is typically about 90% of initial for speculators
DEFAULT_FUTURES_RATE = 0.10

CFD_CLASS = {
    "SP500": "index", "Nasdaq": "index", "Dow": "index", "Ger40": "index", "Nikkei225": "index",
    "Russell2000": "minor index", "Gold": "gold", "Silver": "commodity", "Platinum": "commodity",
    "Palladium": "commodity", "Copper": "commodity", "CrudeOil": "commodity", "Corn": "commodity",
    "Cocoa": "commodity", "Bitcoin": "crypto",
}
CFD_LEVERAGE_PRESETS = {
    "Retail caps (ESMA / ASIC)": {"index": 20, "minor index": 10, "gold": 20, "commodity": 10, "crypto": 2},
    "Offshore / professional (1:100)": {"index": 100, "minor index": 100, "gold": 100, "commodity": 100,
                                         "crypto": 20},
    "Custom (one leverage for everything)": None,
}
CFD_DEFAULT_STEP = 0.01   # smallest CFD size, in units of the underlying (a 0.01 lot of most CFDs)
CFD_STOP_OUT = 0.5        # broker closes positions when equity falls to 50% of the margin in use


def contract_spec(asset: str, mode: str):
    """The ibkr_fees.Spec used for whole-contract sizing in `mode`, or None if the asset has no
    futures mapping."""
    options = ibkr_fees.FEE_TABLE.get(asset)
    if not options:
        return None
    if mode == SIZING_MICRO:
        return options.get("micro") or options["standard"]
    return options["standard"]


def size_steps(mode: str, assets, cfd_step: float = CFD_DEFAULT_STEP) -> tuple:
    """({asset: smallest tradable size in engine shares}, notes) for the chosen sizing mode."""
    steps, notes = {}, []
    if mode in (SIZING_MICRO, SIZING_MINI):
        fallback, unmapped = [], []
        for name in assets:
            spec = contract_spec(name, mode)
            if spec is None:
                unmapped.append(name)
                continue
            steps[name] = float(spec.multiplier)
            if mode == SIZING_MICRO and "micro" not in ibkr_fees.FEE_TABLE[name]:
                fallback.append(f"{name} ({spec.code})")
        if fallback:
            notes.append("No micro contract, standard contract used: " + ", ".join(fallback) + ".")
        if unmapped:
            notes.append("No futures contract known for " + ", ".join(unmapped) + " -- sized fractionally.")
    elif mode == SIZING_CFD:
        steps = {name: float(cfd_step) for name in assets}
    return steps, notes


def futures_margin_rows(mode: str, assets) -> list:
    """Rows for the editable margin table: asset, contract, initial margin as % of contract value."""
    rows = []
    for name in assets:
        spec = contract_spec(name, mode if mode in (SIZING_MICRO, SIZING_MINI) else SIZING_MINI)
        rows.append({"Asset": name, "Contract": spec.code if spec else "-",
                     "Initial margin (% of value)": round(100 * FUTURES_INITIAL_MARGIN.get(name, DEFAULT_FUTURES_RATE), 2)})
    return rows


def futures_margin(assets, overrides_pct: dict = None, stress: float = 1.0) -> dict:
    """{asset: (initial rate, maintenance rate)} from the estimate table, optional per-asset edits
    (percent) and a stress multiplier."""
    overrides_pct = overrides_pct or {}
    spec = {}
    for name in assets:
        initial = overrides_pct.get(name, 100 * FUTURES_INITIAL_MARGIN.get(name, DEFAULT_FUTURES_RATE)) / 100.0
        initial = min(1.0, initial * stress)
        spec[name] = (initial, initial * MAINTENANCE_RATIO)
    return spec


def cfd_margin(assets, preset: str, custom_leverage: float = 30.0, stop_out: float = CFD_STOP_OUT,
               stress: float = 1.0) -> dict:
    """{asset: (initial rate, stop-out rate)} where the initial rate is 1 / leverage and the broker
    closes positions once equity falls to `stop_out` of the margin in use."""
    leverage_by_class = CFD_LEVERAGE_PRESETS.get(preset)
    spec = {}
    for name in assets:
        leverage = custom_leverage if leverage_by_class is None else leverage_by_class[CFD_CLASS.get(name, "commodity")]
        initial = min(1.0, stress / max(leverage, 1.0))
        spec[name] = (initial, initial * stop_out)
    return spec


def cfd_leverage_rows(preset: str, assets, custom_leverage: float = 30.0) -> list:
    leverage_by_class = CFD_LEVERAGE_PRESETS.get(preset)
    return [{"Asset": name, "Class": CFD_CLASS.get(name, "commodity"),
             "Max leverage": custom_leverage if leverage_by_class is None
             else leverage_by_class[CFD_CLASS.get(name, "commodity")]} for name in assets]
