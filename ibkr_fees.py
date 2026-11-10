"""Interactive Brokers futures fee model for the Turtle backtests.

What it does
    apply(td, ...) fills turtle_dual_system.FEE_SCHEDULE with a per-contract all-in fee
    (IBKR commission + exchange fee + regulatory fee) and a contract multiplier for each asset,
    and turns on td.USE_PER_CONTRACT_FEES. The engine then charges
        fee = (shares / multiplier) * fee_per_contract       on every entry, add and exit fill
    instead of the flat COMMISSION_PCT. Assets without a schedule entry keep the % commission.

What it does NOT model (all of these matter for real trading)
    - Whole contracts. The engine sizes in fractional units, so a $10k account "trades" 0.03 of
      an ES contract. Real accounts must round to whole contracts, which changes both sizing
      and fee drag. A small account may be unable to take some signals at all.
    - Roll costs. Yahoo's continuous futures series don't charge the fees of rolling each
      expiry (two fills per roll).
    - Market data subscriptions, margin interest, currency conversion and tax.
    - Slippage and spread: still the engine's SLIPPAGE_PCT / SPREAD_PCT percentages.

DATA PROVENANCE -- READ THIS
    IBKR blocks automated fetching of its fee pages, so only part of the schedule could be
    verified (from IBKR's own published pages as summarized by search, not from your account):
        verified = True    commission and exchange fee found on IBKR pages
        verified = False   ESTIMATE -- plausible figure, NOT confirmed; check it against
                           https://www.interactivebrokers.com/en/pricing/commissions-futures.php
                           and the per-exchange fee pages, then edit the table below.
    Fees are the lowest-volume tier (fixed pricing / first tiered bracket), non-member exchange
    rates, in the asset's own currency (Ger40 in EUR, everything else USD). apply() prints which
    products are estimates every time it runs.
"""
from collections import namedtuple

Spec = namedtuple("Spec", "code multiplier commission exchange regulatory verified")


def _s(code, mult, comm, exch, reg=0.02, verified=False):
    return Spec(code, mult, comm, exch, reg, verified)


# asset name (td.TICKERS key) -> {"micro": Spec or absent, "standard": Spec}
# multiplier = dollars (or euros) of P&L per contract per 1.0 move in the price series
FEE_TABLE = {
    "SP500": {"micro": _s("MES", 5, 0.25, 0.35, verified=True), "standard": _s("ES", 50, 0.85, 1.35, reg=0.0, verified=True)},
    "Nasdaq": {"micro": _s("MNQ", 2, 0.25, 0.35, verified=True), "standard": _s("NQ", 20, 0.85, 1.38)},
    "Russell2000": {"micro": _s("M2K", 5, 0.25, 0.35, verified=True), "standard": _s("RTY", 50, 0.85, 1.38)},
    "Dow": {"micro": _s("MYM", 0.5, 0.25, 0.35, verified=True), "standard": _s("YM", 5, 0.85, 1.38)},
    "Gold": {"micro": _s("MGC", 10, 0.25, 0.60), "standard": _s("GC", 100, 0.85, 1.60)},
    "Silver": {"micro": _s("SIL", 1000, 0.25, 0.60), "standard": _s("SI", 5000, 0.85, 1.60)},
    "Copper": {"micro": _s("MHG", 2500, 0.25, 0.60), "standard": _s("HG", 25000, 0.85, 1.60)},
    "CrudeOil": {"micro": _s("MCL", 100, 0.25, 0.50), "standard": _s("CL", 1000, 0.85, 1.60)},
    "Platinum": {"standard": _s("PL", 50, 0.85, 1.60)},
    "Palladium": {"standard": _s("PA", 100, 0.85, 1.60)},
    # Yahoo quotes corn in cents/bushel; a 5,000 bu contract moves $50 per 1.0 (one cent)
    "Corn": {"standard": _s("ZC", 50, 0.85, 1.50)},
    "Cocoa": {"standard": _s("CC", 10, 0.85, 2.00)},
    "Bitcoin": {"micro": _s("MBT", 0.1, 0.25, 1.25), "standard": _s("BTC", 5, 0.85, 7.50)},
    # fees in EUR for the Eurex DAX contracts (asset price is in EUR)
    "Ger40": {"micro": _s("FDXM", 5, 0.90, 0.45, reg=0.0), "standard": _s("FDAX", 25, 2.00, 0.0, reg=0.0, verified=True)},
    # CME's USD Nikkei contract, $5 per index point
    "Nikkei225": {"standard": _s("NKD", 5, 0.85, 2.00)},
}


def fee_per_contract(spec: Spec) -> float:
    return spec.commission + spec.exchange + spec.regulatory


_COST_NAMES = ("SLIPPAGE_PCT", "SPREAD_PCT", "COMMISSION_PCT", "HOLDING_COST_PCT_PER_YEAR")
ENGINE_DEFAULT_COSTS = {}


def remember_engine_defaults(td) -> dict:
    """Records the engine file's own cost settings the first time it is called (before anything
    has changed them), so the UI can always switch back to them. This module is imported once per
    process, so the record survives Streamlit re-running the app script."""
    if not ENGINE_DEFAULT_COSTS:
        ENGINE_DEFAULT_COSTS.update({name: getattr(td, name) for name in _COST_NAMES})
    return ENGINE_DEFAULT_COSTS


def table_rows(size: str, assets) -> list:
    """One row per asset that has an IBKR mapping, for the editable fee table in the UI."""
    rows = []
    for name in assets:
        options = FEE_TABLE.get(name)
        if not options:
            continue
        spec = options.get(size) or options["standard"]
        rows.append({"Asset": name, "Contract": spec.code, "Multiplier": spec.multiplier,
                     "Fee per side ($)": round(fee_per_contract(spec), 4), "Verified": spec.verified})
    return rows


def apply(td, size: str = "micro", assets=None, overrides: dict = None,
          zero_holding_cost: bool = True) -> tuple:
    """Turns on per-contract IBKR fees in turtle_dual_system `td`. Returns (snapshot, report_lines);
    pass the snapshot to restore() afterwards.

    size: "micro" uses the micro contract where IBKR has one (falls back to the standard contract
        otherwise); "standard" always uses the full-size contract.
    assets: asset names in the run (default td.TICKERS) -- only these get a schedule.
    overrides: {asset: {"fee_per_contract": x, "multiplier": y}} to replace table values.
    zero_holding_cost: futures carry no financing charge, so the engine's HOLDING_COST_PCT_PER_YEAR
        (a leveraged-CFD style cost) is set to 0. Pass False to keep it."""
    if size not in ("micro", "standard"):
        raise ValueError("size must be 'micro' or 'standard'")
    overrides = overrides or {}
    names = list(assets) if assets is not None else list(td.TICKERS)
    snapshot = {"USE_PER_CONTRACT_FEES": td.USE_PER_CONTRACT_FEES, "FEE_SCHEDULE": td.FEE_SCHEDULE,
                "HOLDING_COST_PCT_PER_YEAR": td.HOLDING_COST_PCT_PER_YEAR}

    schedule, lines, estimates, unmapped = {}, [], [], []
    for name in names:
        options = FEE_TABLE.get(name)
        if not options:
            unmapped.append(name)
            continue
        spec = options.get(size) or options["standard"]
        entry = {"multiplier": float(spec.multiplier), "fee_per_contract": fee_per_contract(spec), "code": spec.code}
        entry.update({k: v for k, v in overrides.get(name, {}).items() if k in entry})
        schedule[name] = entry
        flag = "" if spec.verified else "  (ESTIMATE)"
        lines.append(f"  {name:<12} {entry['code']:<5} x{entry['multiplier']:g}  "
                     f"${entry['fee_per_contract']:.2f}/contract per side{flag}")
        if not spec.verified:
            estimates.append(f"{name} ({spec.code})")

    td.FEE_SCHEDULE = schedule
    td.USE_PER_CONTRACT_FEES = True
    if zero_holding_cost:
        td.HOLDING_COST_PCT_PER_YEAR = 0.0

    header = [f"Interactive Brokers futures fees ON ({size} contracts where available; per contract, "
              f"per side, commission + exchange + regulatory):"]
    footer = []
    if zero_holding_cost:
        footer.append("  Holding cost set to 0 (futures have no financing charge).")
    if estimates:
        footer.append(f"  WARNING: {len(estimates)} product(s) use ESTIMATED fees, not verified against IBKR: "
                      f"{', '.join(estimates)}. Edit FEE_TABLE in ibkr_fees.py.")
    if unmapped:
        footer.append(f"  No IBKR mapping for {', '.join(unmapped)} -- these keep the % commission.")
    footer.append("  Not modelled: whole-contract rounding, roll costs, data fees.")
    return snapshot, header + lines + footer


def restore(td, snapshot: dict):
    for name, value in snapshot.items():
        setattr(td, name, value)
