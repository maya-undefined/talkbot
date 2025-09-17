POLICY_PACKS = {
    "default": {
        "pii_rules": {"mask_account": True},
        "advice_boundaries": {
            "no_tax_legal": True,
            "disclosure_text": "This information is for educational purposes and is not financial advice.",
        },
        "prompt_injection": {
            "strip_markers": ["SYSTEM:", "DEVELOPER:", "Ignore previous", "Act as"],
        },
    }
}
