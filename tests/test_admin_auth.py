"""Tests for F14 — Admin authentication.

Verifies that all three admin endpoints reject requests without a valid
X-Admin-Token header and accept requests with the correct token.
"""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import create_engine

from app import db as db_module
from app.config import settings
from app.db import init_db

_TOKEN = "test-secret"
_WRONG = "wrong-token"


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "admin_token", _TOKEN)

    # File-based temp DB so all connections share the same tables.
    db_file = tmp_path / "test.db"
    test_engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
    )

    # Patch engine in db module and every router that imports it directly.
    import app.api.analytics as analytics_module
    import app.api.orders as orders_module
    import app.api.tickets as tickets_module

    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(orders_module, "engine", test_engine)
    monkeypatch.setattr(tickets_module, "engine", test_engine)
    monkeypatch.setattr(analytics_module, "engine", test_engine)

    init_db()

    from app.main import app

    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# GET /tickets
# ---------------------------------------------------------------------------


def test_tickets_no_token(client):
    assert client.get("/tickets").status_code == 401


def test_tickets_wrong_token(client):
    assert client.get("/tickets", headers={"X-Admin-Token": _WRONG}).status_code == 401


def test_tickets_correct_token(client):
    assert client.get("/tickets", headers={"X-Admin-Token": _TOKEN}).status_code == 200


# ---------------------------------------------------------------------------
# GET /analytics/summary
# ---------------------------------------------------------------------------

_ANALYTICS_URL = "/analytics/summary?from=2024-01-01&to=2024-01-31"


def test_analytics_no_token(client):
    assert client.get(_ANALYTICS_URL).status_code == 401


def test_analytics_wrong_token(client):
    assert (
        client.get(_ANALYTICS_URL, headers={"X-Admin-Token": _WRONG}).status_code
        == 401
    )


def test_analytics_correct_token(client):
    assert (
        client.get(_ANALYTICS_URL, headers={"X-Admin-Token": _TOKEN}).status_code
        == 200
    )


# ---------------------------------------------------------------------------
# PATCH /refunds/{refund_id}
# ---------------------------------------------------------------------------

_REFUND_URL = "/refunds/999"
_REFUND_BODY = {"status": "approved"}


def test_refund_no_token(client):
    assert client.patch(_REFUND_URL, json=_REFUND_BODY).status_code == 401


def test_refund_wrong_token(client):
    assert (
        client.patch(
            _REFUND_URL, json=_REFUND_BODY, headers={"X-Admin-Token": _WRONG}
        ).status_code
        == 401
    )


def test_refund_correct_token(client):
    # Auth passes; refund #999 doesn't exist → 404, not 401.
    assert (
        client.patch(
            _REFUND_URL, json=_REFUND_BODY, headers={"X-Admin-Token": _TOKEN}
        ).status_code
        == 404
    )
