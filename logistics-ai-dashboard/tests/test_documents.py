"""Tests for Invoice & Document Intelligence (extraction + three-way match)."""
from api import documents


def test_overview_shape_and_summary():
    o = documents.overview()
    assert o["source"] == "representative"
    q = o["queue"]
    assert q
    s = o["summary"]
    assert s["documents_processed"] == len(q)
    assert s["three_way_matched"] == sum(1 for d in q if d["match_status"] == "matched")
    assert s["exceptions"] == sum(1 for d in q if d["match_status"] != "matched")
    assert s["value_in_flight"] == round(sum(d["amount"] for d in q))


def test_queue_sorted_exceptions_first():
    rank = {"exception": 0, "partial": 1, "matched": 2}
    ranks = [rank[d["match_status"]] for d in documents.overview()["queue"]]
    assert ranks == sorted(ranks)


def test_matched_document_has_no_discrepancies():
    q = documents.overview()["queue"]
    matched = next(d for d in q if d["match_status"] == "matched")
    det = documents.detail(matched["id"])
    assert det["ok"] and det["discrepancies"] == []
    assert all(ln["status"] == "matched" for ln in det["lines"])


def test_exception_document_flags_discrepancy():
    q = documents.overview()["queue"]
    exc = next((d for d in q if d["match_status"] != "matched"), None)
    assert exc is not None
    det = documents.detail(exc["id"])
    assert det["ok"] and det["discrepancies"]
    assert any(ln["status"] == "mismatch" for ln in det["lines"])


def test_unknown_document():
    assert documents.detail("DOC-0000")["ok"] is False
