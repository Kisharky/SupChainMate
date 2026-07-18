"""
modules/doc_intel.py
Document intelligence — scan a freight invoice / BOL and reconcile it
against the shipment board.

Extraction:
  - PDF -> text via pypdf; TXT read directly
  - Groq LLM structures the fields when a key is configured; a regex
    extractor covers the offline case. LLM output is merged over the regex
    base and validated — amounts and matches always come from the data.

Reconciliation checks:
  1. Carrier known on the shipment board?
  2. Shipment references resolve to real shipments?
  3. Invoice total vs the recorded freight costs of those shipments
     (or vs the carrier's audited rate band when no per-shipment cost)
  4. Already-billed warning when the matched shipments carry a recorded cost

Verdicts: OK TO PAY / REVIEW - RATE MISMATCH / REVIEW - UNKNOWN SHIPMENTS /
REVIEW - UNKNOWN CARRIER / INSUFFICIENT DATA.
"""

from __future__ import annotations

import io
import json
import re
from typing import Optional

import numpy as np
import pandas as pd

import config
from modules import groq_ai

RATE_TOLERANCE = config.RATE_TOLERANCE  # deviation that triggers a rate mismatch


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text(file_bytes: bytes, filename: str) -> tuple[Optional[str], str]:
    """Pull raw text out of an uploaded PDF/TXT. Returns (text, message)."""
    name = filename.lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return None, "PDF support needs the pypdf package (pip install pypdf)."
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if not text.strip():
                return None, ("No selectable text in this PDF — it looks scanned. "
                              "Export a text PDF or paste the contents as .txt.")
            return text, "ok"
        except Exception as e:
            return None, f"Could not read PDF: {e}"
    try:
        return file_bytes.decode("utf-8", errors="replace"), "ok"
    except Exception as e:
        return None, f"Could not read file: {e}"


# ── Field extraction (regex base, LLM overlay) ─────────────────────────────────

_INV_NO = re.compile(r"invoice\s*(?:no\.?|number|#)?\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9\-/]{2,20})", re.I)
_TOTAL = re.compile(r"(?:total(?:\s+due)?|amount\s+due|balance(?:\s+due)?|grand\s+total)\s*[:\-]?\s*(?:USD|AUD|\$)?\s*([\d,]+(?:\.\d{1,2})?)", re.I)
_MONEY = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
_DATE = re.compile(r"(?:date|issued|dated)\s*[:\-]?\s*([0-9]{1,4}[/\-.][0-9]{1,2}[/\-.][0-9]{1,4})", re.I)
_REF_TOKEN = re.compile(r"\b([A-Z0-9]{6,16})\b")


def _regex_extract(text: str, known_carriers: list[str]) -> dict:
    fields: dict = {"invoice_number": None, "invoice_date": None, "carrier": None,
                    "total_amount": None, "shipment_refs": []}

    m = _INV_NO.search(text)
    if m:
        fields["invoice_number"] = m.group(1)
    m = _DATE.search(text)
    if m:
        try:
            fields["invoice_date"] = str(pd.to_datetime(m.group(1), dayfirst=False).date())
        except (ValueError, TypeError):
            pass

    totals = [float(t.replace(",", "")) for t in _TOTAL.findall(text)]
    if totals:
        fields["total_amount"] = max(totals)
    else:
        monies = [float(t.replace(",", "")) for t in _MONEY.findall(text)]
        if monies:
            fields["total_amount"] = max(monies)

    low = text.lower()
    for c in known_carriers:
        if c and str(c).lower() in low:
            fields["carrier"] = str(c)
            break

    fields["shipment_refs"] = list(dict.fromkeys(_REF_TOKEN.findall(text.upper())))[:50]
    return fields


_LLM_PROMPT = """Extract fields from this freight invoice / bill of lading text.
Respond ONLY with JSON, no explanation:
{
  "invoice_number": "string or null",
  "invoice_date": "YYYY-MM-DD or null",
  "carrier": "carrier company name or null",
  "total_amount": number or null,
  "shipment_refs": ["order/shipment/tracking references found"]
}"""


def extract_fields(text: str, known_carriers: list[str]) -> tuple[dict, str]:
    """Structured fields from document text. Returns (fields, engine)."""
    base = _regex_extract(text, known_carriers)
    if not groq_ai.is_available():
        return base, "offline"

    raw = groq_ai._call(
        messages=[{"role": "system", "content": _LLM_PROMPT},
                  {"role": "user", "content": text[:6000]}],
        model=groq_ai.MODEL, max_tokens=400, temperature=0.1,
    )
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        llm = json.loads(raw[start:end]) if start >= 0 else {}
    except (json.JSONDecodeError, ValueError):
        return base, "offline"

    merged = dict(base)
    for key in ("invoice_number", "invoice_date", "carrier"):
        if llm.get(key):
            merged[key] = str(llm[key])
    if isinstance(llm.get("total_amount"), (int, float)) and llm["total_amount"] > 0:
        merged["total_amount"] = float(llm["total_amount"])
    if isinstance(llm.get("shipment_refs"), list):
        # union — regex refs feed the matcher too
        merged["shipment_refs"] = list(dict.fromkeys(
            [str(r).upper() for r in llm["shipment_refs"]] + base["shipment_refs"]))[:50]
    return merged, "groq"


# ── Reconciliation ─────────────────────────────────────────────────────────────

def reconcile(fields: dict, shipments: pd.DataFrame,
              by_carrier: Optional[pd.DataFrame] = None) -> dict:
    """
    Match extracted invoice fields against the shipment board.
    Returns {verdict, findings, matched (DataFrame), expected_total}.
    """
    findings: list[str] = []
    problems = 0

    known_ids = set(shipments["shipment_id"].astype(str)) if "shipment_id" in shipments.columns else set()
    refs = [r for r in (fields.get("shipment_refs") or [])]
    matched_ids = [r for r in refs if r in known_ids]
    matched = shipments[shipments["shipment_id"].isin(matched_ids)] if matched_ids else shipments.iloc[0:0]

    # 1. Carrier
    carrier = fields.get("carrier")
    known_carriers = (set(shipments["carrier"].dropna().astype(str))
                      if "carrier" in shipments.columns else set())
    if carrier and known_carriers and carrier not in known_carriers:
        findings.append(f"Carrier '{carrier}' is not on your shipment board "
                        f"(known: {', '.join(sorted(known_carriers))}).")
        problems += 1
    elif carrier:
        findings.append(f"Carrier {carrier} recognised.")

    # 2. Shipment refs
    if matched_ids:
        findings.append(f"{len(matched_ids)} shipment reference(s) matched the board.")
        if carrier and "carrier" in matched.columns:
            wrong = matched[matched["carrier"].astype(str) != str(carrier)]
            if len(wrong):
                findings.append(f"⚠ {len(wrong)} matched shipment(s) belong to a different "
                                f"carrier than the invoice claims.")
                problems += 1
    elif refs:
        findings.append("No invoice reference matched a shipment on the board.")
        problems += 1

    # 3. Amount vs expectation
    total = fields.get("total_amount")
    expected = None
    if total is not None:
        if len(matched) and "freight_cost" in matched.columns and matched["freight_cost"].notna().any():
            expected = float(matched["freight_cost"].sum())
            diff_pct = (total - expected) / expected * 100 if expected > 0 else 0
            if abs(diff_pct) <= RATE_TOLERANCE * 100:
                findings.append(f"Invoice total ${total:,.2f} matches the recorded cost "
                                f"${expected:,.2f} ({diff_pct:+.1f}%).")
            else:
                findings.append(f"⚠ Invoice total ${total:,.2f} vs recorded ${expected:,.2f} "
                                f"({diff_pct:+.1f}%) — outside the {RATE_TOLERANCE*100:.0f}% tolerance.")
                problems += 1
            findings.append("Note: these shipments already carry recorded charges — "
                            "confirm this invoice isn't a re-bill before paying.")
        elif carrier and by_carrier is not None:
            row = by_carrier[by_carrier["Carrier"].astype(str) == str(carrier)]
            if len(row):
                med = float(row.iloc[0]["Median_Cost"])
                n = max(len(matched_ids), 1)
                per = total / n
                expected = med * n
                if med > 0 and per > med * (1 + RATE_TOLERANCE * 3):
                    findings.append(f"⚠ ${per:,.2f} per shipment vs {carrier}'s median "
                                    f"${med:,.2f} — well above the audited rate band.")
                    problems += 1
                else:
                    findings.append(f"${per:,.2f} per shipment is within {carrier}'s "
                                    f"audited range (median ${med:,.2f}).")
    else:
        findings.append("No invoice total found in the document.")

    # Verdict
    if total is None and not matched_ids:
        verdict = "INSUFFICIENT DATA"
    elif carrier and known_carriers and carrier not in known_carriers:
        verdict = "REVIEW — UNKNOWN CARRIER"
    elif refs and not matched_ids:
        verdict = "REVIEW — UNKNOWN SHIPMENTS"
    elif problems:
        verdict = "REVIEW — RATE MISMATCH"
    else:
        verdict = "OK TO PAY"

    return {"verdict": verdict, "findings": findings,
            "matched": matched, "expected_total": expected}


# ── Sample invoice (demo / testing) ────────────────────────────────────────────

def sample_invoice(shipments: pd.DataFrame, inflate: bool = True,
                   seed: int = 11) -> Optional[str]:
    """
    Build a realistic invoice text from real board shipments (one line
    optionally inflated) so the scanner can be demoed without a real document.
    """
    pool = shipments.dropna(subset=["freight_cost"]) if "freight_cost" in shipments.columns else shipments
    if not len(pool):
        return None
    rng = np.random.default_rng(seed)
    rows = pool.sample(min(3, len(pool)), random_state=seed)
    carrier = rows.iloc[0].get("carrier", "Acme Freight")
    lines, total = [], 0.0
    for i, (_, r) in enumerate(rows.iterrows()):
        amt = float(r.get("freight_cost", 25.0) or 25.0)
        if inflate and i == 0:
            amt = round(amt * 2.6, 2)  # simulated billing error
        total += amt
        lines.append(f"  Shipment {r['shipment_id']}   freight services   ${amt:,.2f}")
    inv_no = f"INV-{rng.integers(10000, 99999)}"
    return (
        f"{carrier}\nTAX INVOICE\n\n"
        f"Invoice Number: {inv_no}\nDate: 2018-08-15\n"
        f"Bill To: Your Company Pty Ltd\n\n"
        + "\n".join(lines)
        + f"\n\n  TOTAL DUE: ${total:,.2f}\n\nPayment terms: 14 days\n"
    )
