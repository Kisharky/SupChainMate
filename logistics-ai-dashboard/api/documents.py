"""
api/documents.py — Invoice & Document Intelligence.

Extracts and audits back-office documents (invoices, bills of lading, POs,
goods receipts) and runs a **three-way match** (PO ↔ Invoice ↔ Receipt) to catch
over-billing, quantity mismatches, and missing references before payment —
the AP-automation pattern that anchors freight-broker back offices.

This is the surface for the AI Router's already-declared OCR + RAG capabilities:
extraction/confidence and match results are representative here (labelled,
deterministic), and this module is the clean seam where a real document parser
plugs in. No AI engine or business logic is modified.
"""

from __future__ import annotations

import hashlib
from typing import Any

from api.services import _safe

_TYPES = {"invoice": "Invoice", "bol": "Bill of Lading", "po": "Purchase Order", "grn": "Goods Receipt"}

# Representative document queue. match_status is derived, not stored.
_DOCS = [
    ("invoice", "Northwind Freight", "PO-55210", 18420.00),
    ("invoice", "Continental Supply Co", "PO-55188", 40560.00),
    ("bol", "Apex Carriers LLC", "PO-55201", 12750.00),
    ("invoice", "Cedar Logistics", "PO-55177", 6240.00),
    ("invoice", "GlobalMart Wholesale", "PO-55164", 88120.00),
    ("grn", "Pioneer Transit", "PO-55155", 15300.00),
    ("invoice", "Blue Ridge Haulage", "PO-55149", 9310.00),
    ("po", "Vendor #4471", "PO-55212", 31240.00),
]


def _seed(s: str) -> int:
    return int(hashlib.sha1(s.encode()).hexdigest(), 16)


def _doc(i: int, kind: str, vendor: str, po: str, amount: float) -> dict[str, Any]:
    h = _seed(f"{kind}{vendor}{po}{i}")
    conf = min(99, 88 + h % 12)
    # Deterministic match outcome: mostly matched, some partial/exception.
    outcome = ("matched", "matched", "matched", "partial", "exception", "matched")[h % 6]
    discrepancies = 0 if outcome == "matched" else (1 if outcome == "partial" else 2)
    return {
        "id": f"DOC-{7100 + i}",
        "type": kind,
        "type_label": _TYPES[kind],
        "vendor": vendor,
        "po_number": po,
        "amount": amount,
        "extraction_confidence": conf,
        "match_status": outcome,
        "discrepancy_count": discrepancies,
        "status": "auto_approved" if outcome == "matched" else "needs_review",
        "hours_ago": 1 + (h % 60),
    }


def _queue() -> list[dict[str, Any]]:
    docs = [_doc(i, *d) for i, d in enumerate(_DOCS)]
    order = {"exception": 0, "partial": 1, "matched": 2}
    docs.sort(key=lambda d: (order[d["match_status"]], d["hours_ago"]))
    return docs


def _summary(docs: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [d for d in docs if d["match_status"] == "matched"]
    exceptions = [d for d in docs if d["match_status"] != "matched"]
    return {
        "documents_processed": len(docs),
        "straight_through_pct": round(100 * len(matched) / len(docs)) if docs else 0,
        "three_way_matched": len(matched),
        "exceptions": len(exceptions),
        "avg_confidence": round(sum(d["extraction_confidence"] for d in docs) / len(docs)) if docs else 0,
        "value_in_flight": round(sum(d["amount"] for d in docs)),
    }


def overview() -> dict[str, Any]:
    """Document queue + processing summary. Representative + labelled."""
    def build() -> dict[str, Any]:
        docs = _queue()
        return {"summary": _summary(docs), "queue": docs, "source": "representative"}
    return _safe(build, {"summary": {}, "queue": [], "source": "fallback"})


# ---- Per-document detail: extracted fields + three-way match breakdown --------

_LINE_ITEMS = [
    ("SKU-7781", "Pallet freight — SP lane", 12, 24.90),
    ("SKU-3355", "Expedited handling", 4, 88.00),
    ("SKU-9020", "Fuel surcharge", 1, 310.00),
]


def detail(doc_id: str) -> dict[str, Any]:
    """Extracted fields and a three-way match (PO ↔ Invoice ↔ Receipt) for one
    document. Discrepancies are deterministic from the id so the demo is stable."""
    def build() -> dict[str, Any]:
        match = next((d for d in _queue() if d["id"] == doc_id), None)
        if match is None:
            return {"ok": False, "error": "unknown document", "doc_id": doc_id}
        h = _seed(doc_id)
        # Which line (if any) carries a discrepancy, driven by the match status.
        bad_line = -1 if match["match_status"] == "matched" else h % len(_LINE_ITEMS)
        lines: list[dict[str, Any]] = []
        for idx, (sku, desc, qty, price) in enumerate(_LINE_ITEMS):
            po_amt = round(qty * price, 2)
            inv_qty = qty
            inv_price = price
            if idx == bad_line:
                if match["match_status"] == "partial":
                    inv_price = round(price * 1.08, 2)      # price variance
                else:
                    inv_qty = qty + 2                        # quantity mismatch
            inv_amt = round(inv_qty * inv_price, 2)
            rec_qty = qty                                    # receipt = what was received
            ok = (inv_qty == qty and abs(inv_price - price) < 0.01)
            lines.append({
                "sku": sku, "description": desc,
                "po_qty": qty, "po_price": price, "po_amount": po_amt,
                "invoice_qty": inv_qty, "invoice_price": inv_price, "invoice_amount": inv_amt,
                "receipt_qty": rec_qty,
                "status": "matched" if ok else "mismatch",
            })
        discrepancies = []
        for ln in lines:
            if ln["status"] == "mismatch":
                if ln["invoice_qty"] != ln["po_qty"]:
                    discrepancies.append(
                        f"{ln['sku']}: invoiced {ln['invoice_qty']} vs PO {ln['po_qty']} units")
                if abs(ln["invoice_price"] - ln["po_price"]) >= 0.01:
                    discrepancies.append(
                        f"{ln['sku']}: unit price ${ln['invoice_price']} vs PO ${ln['po_price']}")
        action = ("Auto-approve — all lines reconcile." if not discrepancies
                  else "Hold for AP review — resolve the flagged lines before payment.")
        return {
            "ok": True, "doc_id": doc_id, "type_label": match["type_label"],
            "vendor": match["vendor"], "po_number": match["po_number"],
            "extraction_confidence": match["extraction_confidence"],
            "match_status": match["match_status"],
            "fields": {
                "Vendor": match["vendor"], "PO Number": match["po_number"],
                "Invoice Total": f"${match['amount']:,.2f}",
                "Currency": "USD", "Terms": "Net 30", "Due": "2026-08-20",
            },
            "lines": lines,
            "discrepancies": discrepancies,
            "recommended_action": action,
            "source": "representative",
        }
    return _safe(build, {"ok": False, "doc_id": doc_id, "error": "unavailable"})
