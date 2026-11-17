"""Tests for the /v1/anonymize id-masking endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.anonymize import (
    anonymize_batches,
    anonymize_initial,
    build_mapping,
    collect_ids,
)
from app.main import app
from app.models import Batch, InitialData

from conftest import (
    BUGGY_INCREMENTAL_DELETE_MISSING,
    CORRECT_FULL,
    CORRECT_INCREMENTAL,
    op_delete,
    op_insert,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def _payload(incremental: str) -> dict:
    return {
        "initial": {
            "orders": [
                {"order_id": 1, "amount": 1000},
                {"order_id": 2, "amount": 500},
                {"order_id": 3, "amount": 300},
            ],
            "refunds": [{"refund_id": 10, "order_id": 1, "amount": 200}],
        },
        "batches": [
            {"batch_id": 11, "changes": [op_insert("refunds", refund_id=20, order_id=1, amount=50)]},
            {"batch_id": 22, "changes": [op_insert("refunds", refund_id=30, order_id=2, amount=70)]},
            {"batch_id": 33, "changes": [op_insert("refunds", refund_id=21, order_id=3, amount=90)]},
            {"batch_id": 44, "changes": [op_delete("refunds", refund_id=21)]},
        ],
        "sql": {"full_select": CORRECT_FULL, "incremental": incremental},
    }


# Original business ids used by _payload: orders 1,2,3; refunds 10,20,30,21.
ORIGINAL_IDS = {1, 2, 3, 10, 20, 21, 30}


def _walk(node, path=()):
    """Yield (path, key, value) for every dict entry / list item."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield path, key, value
            yield from _walk(value, path + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, path + (str(index),))


# ---------------------------------------------------------------------------
# mapping unit tests
# ---------------------------------------------------------------------------

def test_build_mapping_is_injective_positive_and_non_overlapping():
    ids = {5, 1, 9, 100}
    mapping = build_mapping(ids)
    assert set(mapping) == ids
    new_ids = set(mapping.values())
    assert len(new_ids) == len(ids)  # injective
    assert all(v >= 1 for v in new_ids)  # positive
    assert not (new_ids & ids)  # non-overlapping with the original ids


def test_build_mapping_handles_huge_ids_without_overflow():
    ids = {2**63 - 1, 2**63 - 2}
    mapping = build_mapping(ids)
    assert all(1 <= v < 2**63 for v in mapping.values())
    assert not (set(mapping.values()) & ids)


def test_build_mapping_deterministic():
    ids = {7, 3, 3, 42}
    assert build_mapping(ids) == build_mapping(set(ids))


def test_anonymize_keeps_amounts_batches_and_order():
    initial = InitialData.model_validate(_payload(BUGGY_INCREMENTAL_DELETE_MISSING)["initial"])
    batches = [Batch.model_validate(b) for b in _payload(BUGGY_INCREMENTAL_DELETE_MISSING)["batches"]]
    mapping = build_mapping(collect_ids(initial, batches))

    anon_initial = anonymize_initial(initial, mapping)
    anon_batches = anonymize_batches(batches, mapping)

    # Amounts untouched, in the same row order.
    assert [r.amount for r in anon_initial.orders] == [1000, 500, 300]
    assert [r.amount for r in anon_initial.refunds] == [200]
    # Batch ids and change order preserved.
    assert [b.batch_id for b in anon_batches] == [11, 22, 33, 44]
    assert [c.op for b in anon_batches for c in b.changes] == ["insert"] * 3 + ["delete"]
    # FK consistency: refund attribution follows the order mapping.
    assert anon_initial.refunds[0].order_id == mapping[1]
    # Refund insert in batch 11 points at the same anonymized order.
    assert anon_batches[0].changes[0].after["order_id"] == mapping[1]
    assert anon_batches[0].changes[0].after["amount"] == 50
    # Delete target is the anonymized refund id.
    assert anon_batches[3].changes[0].before == {"refund_id": mapping[21]}


# ---------------------------------------------------------------------------
# shareable path
# ---------------------------------------------------------------------------

def test_anonymize_shareable_replaces_all_ids():
    client = TestClient(app)
    response = client.post("/v1/anonymize", json=_payload(BUGGY_INCREMENTAL_DELETE_MISSING))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "shareable"

    sample = body["anonymized_input"]
    assert set(sample) == {"initial", "batches"}  # no SQL delivered

    # Deterministic smallest-free mapping: 1->4, 2->5, 3->6, 10->7,
    # 20->8, 21->9, 30->11.
    assert sample["initial"]["orders"] == [
        {"order_id": 4, "amount": 1000},
        {"order_id": 5, "amount": 500},
        {"order_id": 6, "amount": 300},
    ]
    assert sample["initial"]["refunds"] == [
        {"refund_id": 7, "order_id": 4, "amount": 200}
    ]
    assert [b["batch_id"] for b in sample["batches"]] == [11, 22, 33, 44]
    assert sample["batches"][0]["changes"][0]["after"] == {
        "refund_id": 8, "order_id": 4, "amount": 50,
    }
    assert sample["batches"][3]["changes"][0]["before"] == {"refund_id": 9}

    # Witness uses the new ids and matches the original witness shape.
    assert body["first_difference_batch_id"] == 44
    assert body["difference"] == {"missing": [[6, 300]], "extra": [[6, 210]]}


def test_anonymize_response_contains_no_original_ids_sql_or_mapping():
    client = TestClient(app)
    payload = _payload(BUGGY_INCREMENTAL_DELETE_MISSING)
    body = client.post("/v1/anonymize", json=payload).json()
    assert body["status"] == "shareable"

    text = json.dumps(body)
    # No original SQL text and no sql/mapping fields anywhere.
    assert payload["sql"]["incremental"] not in text
    assert payload["sql"]["full_select"] not in text
    for _, key, _ in _walk(body):
        assert key not in ("sql", "full_select", "incremental", "mapping")

    # No original business id appears in any id field or witness row.
    seen_new_ids: set[int] = set()
    for path, key, value in _walk(body):
        if key in ("order_id", "refund_id"):
            assert value not in ORIGINAL_IDS, f"原号泄露于 {path}"
            assert value >= 1
            seen_new_ids.add(value)
        if key == "missing" or key == "extra":
            for row in value:
                assert row[0] not in ORIGINAL_IDS
    assert seen_new_ids and not (seen_new_ids & ORIGINAL_IDS)


def test_anonymized_sample_replays_to_same_witness_with_original_sql():
    client = TestClient(app)
    payload = _payload(BUGGY_INCREMENTAL_DELETE_MISSING)
    body = client.post("/v1/anonymize", json=payload).json()
    assert body["status"] == "shareable"

    # The external maintainer replays the anonymized sample with their own SQL.
    replay = client.post(
        "/v1/validate",
        json={**body["anonymized_input"], "sql": payload["sql"]},
    ).json()
    assert replay["outcome"] == "different"
    assert replay["first_difference_batch_id"] == body["first_difference_batch_id"]
    assert replay["difference"] == body["difference"]


def test_anonymize_is_deterministic():
    client = TestClient(app)
    payload = _payload(BUGGY_INCREMENTAL_DELETE_MISSING)
    first = client.post("/v1/anonymize", json=payload).json()
    second = client.post("/v1/anonymize", json=payload).json()
    assert first == second


def test_anonymize_example_leak_refund_delete(client: TestClient):
    payload = json.loads((EXAMPLES / "leak_refund_delete.json").read_text())
    response = client.post("/v1/anonymize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "shareable"
    assert body["first_difference_batch_id"] == 103
    # 1001->1, 1002->2, 1003->3, 9001->4, 9002->5, 9003->6.
    assert body["difference"] == {"missing": [[1, 3800]], "extra": [[1, 3300]]}
    sample_ids = {
        row["order_id"] for row in body["anonymized_input"]["initial"]["orders"]
    }
    assert sample_ids == {1, 2}
    assert not (sample_ids & {1001, 1002, 1003})


# ---------------------------------------------------------------------------
# refusal paths: report not_shareable, deliver nothing
# ---------------------------------------------------------------------------

def _assert_not_shareable_delivers_nothing(body: dict) -> None:
    assert body["status"] == "not_shareable"
    assert body["anonymized_input"] is None
    assert body["first_difference_batch_id"] is None
    assert body["difference"] is None
    assert body["notes"]


def test_anonymize_no_difference_original_refused(client: TestClient):
    response = client.post("/v1/anonymize", json=_payload(CORRECT_INCREMENTAL))
    assert response.status_code == 200
    _assert_not_shareable_delivers_nothing(response.json())


def test_anonymize_undeterminable_original_refused(client: TestClient):
    payload = _payload(BUGGY_INCREMENTAL_DELETE_MISSING)
    payload["sql"]["full_select"] = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
        "WHERE x < 100000000) SELECT count(*), 0 FROM c"
    )
    response = client.post("/v1/anonymize", json=payload)
    assert response.status_code == 200
    _assert_not_shareable_delivers_nothing(response.json())


# Buggy base (refund deletes ignored) plus a compensation guarded by a
# hard-coded original order id: fires only in the original sample, so the
# anonymized witness cannot correspond to the original one.
WITNESS_MISMATCH_INCREMENTAL = BUGGY_INCREMENTAL_DELETE_MISSING + """
UPDATE result SET net_amount = net_amount + 50
WHERE order_id = 3
  AND EXISTS (SELECT 1 FROM batch_changes WHERE tbl = 'refunds' AND op = 'delete');
"""


def test_anonymize_witness_mismatch_refused(client: TestClient):
    # Sanity: the original sample does diverge (partial +50 compensation).
    original = client.post("/v1/validate", json=_payload(WITNESS_MISMATCH_INCREMENTAL)).json()
    assert original["outcome"] == "different"
    assert original["first_difference_batch_id"] == 44
    assert original["difference"] == {"missing": [[3, 300]], "extra": [[3, 260]]}

    response = client.post("/v1/anonymize", json=_payload(WITNESS_MISMATCH_INCREMENTAL))
    assert response.status_code == 200
    _assert_not_shareable_delivers_nothing(response.json())


# Extra statement keyed on a *new* id value: a no-op for the original sample
# (no order 4) but corrupting every batch after anonymization (order 1 -> 4),
# so the first-difference batch no longer matches.
FIRST_BATCH_MISMATCH_INCREMENTAL = BUGGY_INCREMENTAL_DELETE_MISSING + """
UPDATE result SET net_amount = net_amount - 1 WHERE order_id = 4;
"""


def test_anonymize_first_difference_batch_mismatch_refused(client: TestClient):
    original = client.post("/v1/validate", json=_payload(FIRST_BATCH_MISMATCH_INCREMENTAL)).json()
    assert original["outcome"] == "different"
    assert original["first_difference_batch_id"] == 44

    response = client.post("/v1/anonymize", json=_payload(FIRST_BATCH_MISMATCH_INCREMENTAL))
    assert response.status_code == 200
    _assert_not_shareable_delivers_nothing(response.json())


def test_anonymize_rejects_illegal_input(client: TestClient):
    payload = _payload(BUGGY_INCREMENTAL_DELETE_MISSING)
    payload["batches"][3]["changes"] = [
        {"op": "delete", "table": "refunds", "before": {"refund_id": 999}}
    ]
    response = client.post("/v1/anonymize", json=payload)
    assert response.status_code == 422
    assert "不存在" in response.json()["detail"]


def test_anonymize_rejects_unsafe_sql(client: TestClient):
    response = client.post("/v1/anonymize", json=_payload("DROP TABLE result"))
    assert response.status_code == 422


def test_anonymize_rejects_duplicate_batch_ids(client: TestClient):
    payload = _payload(BUGGY_INCREMENTAL_DELETE_MISSING)
    payload["batches"][1]["batch_id"] = payload["batches"][0]["batch_id"]
    assert client.post("/v1/anonymize", json=payload).status_code == 422


def test_anonymize_advertised_in_openapi(client: TestClient):
    schema = client.get("/openapi.json").json()
    assert "/v1/anonymize" in schema["paths"]
    # Original endpoints untouched.
    assert "/v1/minimize" in schema["paths"]
    assert "/v1/validate" in schema["paths"]
