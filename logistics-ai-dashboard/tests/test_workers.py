"""Tests for the AI Digital Workers cockpit (representative, over real capabilities)."""
from api import workers


def test_roster_derives_from_planner_capabilities():
    from planner import PLANNER
    cockpit = workers.cockpit()
    assert cockpit["source"] == "representative"
    # One worker per registered Planner capability — discovered, not hard-coded.
    assert len(cockpit["workers"]) == len(PLANNER.capabilities())
    cap_names = {c["name"] for c in PLANNER.capabilities()}
    assert {w["id"] for w in cockpit["workers"]} == cap_names


def test_worker_fields_and_bounds():
    for w in workers.cockpit()["workers"]:
        assert {"name", "skill", "domain", "status", "zero_touch_pct", "tasks_today"} <= w.keys()
        assert 55 <= w["zero_touch_pct"] <= 97
        assert w["status"] in ("active", "idle")


def test_summary_aggregates_queue_states():
    c = workers.cockpit()
    s = c["summary"]
    assert s["total_workers"] == len(c["workers"])
    assert s["tasks_automated_today"] == sum(w["tasks_today"] for w in c["workers"])
    assert s["awaiting_approval"] == sum(1 for t in c["queue"] if t["state"] == "awaiting_approval")
    assert s["escalated"] == sum(1 for t in c["queue"] if t["state"] == "escalated")


def test_queue_is_deterministic():
    a = workers.cockpit()["queue"]
    b = workers.cockpit()["queue"]
    assert [t["id"] for t in a] == [t["id"] for t in b]
    assert all(t["state"] in ("auto_completed", "awaiting_approval", "escalated", "running") for t in a)
