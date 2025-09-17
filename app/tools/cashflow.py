from typing import Dict, Any, List

def tool_cashflow(args: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = args.get("rows", [])
    month = args.get("month")  # YYYY-MM
    by_cat: Dict[str, float] = {}
    for r in rows:
        if month and not str(r.get("date", "")).startswith(month):
            continue
        cat = r.get("category", "uncategorized")
        by_cat[cat] = by_cat.get(cat, 0.0) + float(r.get("amount", 0.0))
    total = sum(by_cat.values())
    return {"total": total, "by_category": by_cat}