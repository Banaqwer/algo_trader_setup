from data_loader import load_eurusd_timeframes
from strategies.pdh_acceptance_h4trend_v1 import StrategyV1

def _json_dumps(value):
    if value is None: return "null"
    if value is True: return "true"
    if value is False: return "false"
    if isinstance(value, (int, float)):
        if value != value: return "null"
        if value == float("inf"): return "1e9999"
        if value == float("-inf"): return "-1e9999"
        return str(value)
    if isinstance(value, str):
        escaped = (value.replace("\\", "\\\\").replace("\"", "\\\"")
            .replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t"))
        return f"\"{escaped}\""
    if isinstance(value, list): return "[" + ",".join(_json_dumps(item) for item in value) + "]"
    if isinstance(value, dict):
        items = [f"{_json_dumps(str(key))}:{_json_dumps(value[key])}" for key in sorted(value.keys(), key=lambda k: str(k))]
        return "{" + ",".join(items) + "}"
    return _json_dumps(str(value))

def _csv_escape(text):
    text = "" if text is None else str(text)
    return '"' + text.replace('"', '""') + '"' if any(ch in text for ch in [",", "\n", "\r", "\""]) else text

def _write_trades(trades, path):
    if trades is None: open(path, "w", encoding="utf-8").close(); return
    if hasattr(trades, "to_csv"): trades.to_csv(path, index=False); return
    if isinstance(trades, dict): trades = [trades]
    if isinstance(trades, list) and trades and isinstance(trades[0], dict):
        headers = sorted({key for row in trades for key in row.keys()})
        lines = [",".join(_csv_escape(header) for header in headers)]
        lines += [",".join(_csv_escape(row.get(header)) for header in headers) for row in trades]
        with open(path, "w", encoding="utf-8") as handle: handle.write("\n".join(lines))
        return
    with open(path, "w", encoding="utf-8") as handle: handle.write(str(trades))

def _pick(summary, keys):
    if not isinstance(summary, dict): return None
    lower = {str(key).lower(): key for key in summary}
    for key in keys:
        if key in summary: return summary[key]
        if key.lower() in lower: return summary[lower[key.lower()]]
    return None

def _metrics(trades, summary):
    total_trades = _pick(summary, ["total_trades", "trades", "num_trades"])
    winrate = _pick(summary, ["winrate", "win_rate", "win%", "win_pct"])
    expectancy = _pick(summary, ["expectancy_R", "expectancy_r", "expectancy"])
    profit_factor = _pick(summary, ["profit_factor", "profitfactor"])
    max_drawdown = _pick(summary, ["max_drawdown_R", "max_drawdown_r", "max_drawdown"])
    r_values = []
    if hasattr(trades, "iterrows"):
        for _, row in trades.iterrows():
            for key in ["r", "R", "pnl_R", "result_R", "profit_R", "risk_multiple"]:
                if hasattr(row, "get") and key in row: r_values.append(float(row[key])); break
    elif isinstance(trades, list):
        for row in trades:
            if isinstance(row, dict):
                for key in ["r", "R", "pnl_R", "result_R", "profit_R", "risk_multiple"]:
                    if key in row: r_values.append(float(row[key])); break
    if total_trades is None and trades is not None:
        try: total_trades = len(trades)
        except TypeError: total_trades = None
    if r_values:
        wins = [value for value in r_values if value > 0]
        losses = [value for value in r_values if value < 0]
        total = len(r_values)
        if winrate is None: winrate = len(wins) / total
        if expectancy is None: expectancy = sum(r_values) / total
        if profit_factor is None:
            loss_sum = -sum(losses); profit_factor = sum(wins) / loss_sum if loss_sum else float("inf")
        if max_drawdown is None:
            peak = 0.0; cumulative = 0.0; max_dd = 0.0
            for value in r_values:
                cumulative += value; peak = max(peak, cumulative); max_dd = max(max_dd, peak - cumulative)
            max_drawdown = max_dd
    return {"total_trades": total_trades, "winrate": winrate, "expectancy_R": expectancy, "profit_factor": profit_factor, "max_drawdown_R": max_drawdown}

def main():
    data = load_eurusd_timeframes("EUR_USD_M5.csv")
    try: strategy = StrategyV1(data)
    except TypeError: strategy = StrategyV1()
    if hasattr(strategy, "run"): result = strategy.run(data)
    elif hasattr(strategy, "simulate"): result = strategy.simulate(data)
    elif hasattr(strategy, "backtest"): result = strategy.backtest(data)
    else: raise RuntimeError("StrategyV1 lacks a runnable method.")
    trades = None; summary = None
    if isinstance(result, tuple) and len(result) == 2: trades, summary = result
    elif isinstance(result, dict):
        trades = result.get("trades") or result.get("trades_df"); summary = result.get("summary") or result.get("stats") or result
    else: trades = getattr(result, "trades", None); summary = getattr(result, "summary", None)
    metrics = _metrics(trades, summary)
    _write_trades(trades, "backtests/outputs/trades_v1.csv")
    summary_payload = dict(summary) if isinstance(summary, dict) else {}; summary_payload.update(metrics)
    with open("backtests/outputs/summary_v1.json", "w", encoding="utf-8") as handle: handle.write(_json_dumps(summary_payload))
    print(f"total trades: {metrics['total_trades']}")
    print(f"winrate: {metrics['winrate']}")
    print(f"expectancy_R: {metrics['expectancy_R']}")
    print(f"profit factor: {metrics['profit_factor']}")
    print(f"max drawdown in R: {metrics['max_drawdown_R']}")

if __name__ == "__main__":
    main()
