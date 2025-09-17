from typing import Dict, Any

def tool_calc(args: Dict[str, Any]) -> Dict[str, Any]:
    kind = args.get("kind")
    if kind == "apy_from_apr":
        apr = float(args["apr"]) ; n = int(args.get("n", 12))
        apy = (1 + apr / n) ** n - 1
        return {"apy": apy}
    if kind == "loan_payment":
        P = float(args["principal"]) ; r = float(args["rate"]) / 12 ; N = int(args["months"]) 
        if r == 0: pmt = P / N
        else:      pmt = (P * r) / (1 - (1 + r) ** (-N))
        return {"payment": pmt}
    if kind == "cagr":
        start = float(args["start"]) ; end = float(args["end"]) ; years = float(args["years"]) or 1
        cagr = (end / (start + 1e-9)) ** (1 / max(years, 1e-9)) - 1
        return {"cagr": cagr}
    return {"error": "unknown calc kind"}