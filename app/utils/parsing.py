def extract_text(filename: str, raw: bytes) -> str:
    name = (filename or "").lower()
    if name.endswith((".csv", ".tsv")):
        s = raw.decode(errors="ignore")
        return s.replace(",", " ").replace("	", " ")
    if name.endswith((".txt", ".md")):
        return raw.decode(errors="ignore")
    if name.endswith(".pdf"):
        return "(PDF extraction stub) " + (raw[:2048].decode(errors="ignore"))
    try:
        return raw.decode()
    except Exception:
        return ""