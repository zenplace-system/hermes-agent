"""A cached agent can outlive a Codex OAuth token rotated by another process."""

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

from agent.agent_runtime_helpers import recover_with_credential_pool
from agent.credential_pool import load_pool


def test_codex_401_reloads_new_token_for_cached_session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.agent_runtime_helpers._ra",
        lambda: SimpleNamespace(logger=logging.getLogger(__name__)),
    )
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    auth_file = home / "auth.json"

    def write_token(token):
        auth_file.write_text(json.dumps({
            "version": 1,
            "credential_pool": {"openai-codex": [{
                "id": "account-a",
                "label": "account-a",
                "auth_type": "oauth",
                "priority": 0,
                "source": "manual:device_code",
                "access_token": token,
                "refresh_token": "unused-refresh-token",
            }]},
        }), encoding="utf-8")

    write_token("old-access-token")
    cached_pool = load_pool("openai-codex")
    assert [entry.id for entry in cached_pool.entries()] == ["account-a"]
    write_token("new-access-token")

    swap = Mock(return_value=True)
    agent = SimpleNamespace(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-6.1-sol",
        api_key="old-access-token",
        _credential_pool=cached_pool,
        _credential_pool_entry_id=None,
        _is_entitlement_failure=lambda *_: False,
        _swap_credential=swap,
    )

    recovered, _ = recover_with_credential_pool(
        agent,
        status_code=401,
        has_retried_429=False,
        error_context={"reason": "token_invalidated"},
    )

    assert recovered is True
    swapped = swap.call_args.args[0]
    assert swapped.id == "account-a"
    assert swapped.runtime_api_key == "new-access-token"
    assert agent._credential_pool is not cached_pool
    assert json.loads(auth_file.read_text())["credential_pool"]["openai-codex"][0]["access_token"] == "new-access-token"


def test_codex_401_does_not_guess_among_multiple_accounts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.agent_runtime_helpers._ra",
        lambda: SimpleNamespace(logger=logging.getLogger(__name__)),
    )
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    auth_file = home / "auth.json"

    def write_pool(first_token):
        auth_file.write_text(json.dumps({
            "version": 1,
            "credential_pool": {"openai-codex": [
                {"id": "account-a", "label": "a", "auth_type": "oauth", "priority": 0,
                 "source": "manual:device_code", "access_token": first_token,
                 "refresh_token": "unused-a"},
                {"id": "account-b", "label": "b", "auth_type": "oauth", "priority": 1,
                 "source": "manual:device_code", "access_token": "account-b-token",
                 "refresh_token": "unused-b"},
            ]},
        }), encoding="utf-8")

    write_pool("old-a-token")
    cached_pool = load_pool("openai-codex")
    write_pool("new-a-token")
    swap = Mock(return_value=True)
    agent = SimpleNamespace(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-6.1-sol",
        api_key="unmatched-wrapper-token",
        _credential_pool=cached_pool,
        _credential_pool_entry_id=None,
        _is_entitlement_failure=lambda *_: False,
        _swap_credential=swap,
    )

    recovered, _ = recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False,
        error_context={"reason": "token_invalidated"},
    )

    assert recovered is False
    swap.assert_not_called()
    assert agent._credential_pool is cached_pool
    assert all(entry.last_status != "exhausted" for entry in cached_pool.entries())


def test_codex_401_stale_id_uses_failed_key_account(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.agent_runtime_helpers._ra",
        lambda: SimpleNamespace(logger=logging.getLogger(__name__)),
    )
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    auth_file = home / "auth.json"

    def write_pool(first_token):
        auth_file.write_text(json.dumps({
            "version": 1,
            "credential_pool": {"openai-codex": [
                {"id": "account-a", "label": "a", "auth_type": "oauth", "priority": 0,
                 "source": "manual:device_code", "access_token": first_token,
                 "refresh_token": "unused-a"},
                {"id": "account-b", "label": "b", "auth_type": "oauth", "priority": 1,
                 "source": "manual:device_code", "access_token": "account-b-token",
                 "refresh_token": "unused-b"},
            ]},
        }), encoding="utf-8")

    write_pool("old-a-token")
    cached_pool = load_pool("openai-codex")
    write_pool("new-a-token")
    swap = Mock(return_value=True)
    agent = SimpleNamespace(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-6.1-sol",
        api_key="old-a-token",
        _credential_pool=cached_pool,
        _credential_pool_entry_id="account-b",
        _is_entitlement_failure=lambda *_: False,
        _swap_credential=swap,
    )

    recovered, _ = recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False,
        error_context={"reason": "token_invalidated"},
    )

    assert recovered is True
    assert swap.call_args.args[0].id == "account-a"
    assert swap.call_args.args[0].runtime_api_key == "new-a-token"


def test_codex_401_sole_entry_replaced_by_another_account(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "agent.agent_runtime_helpers._ra",
        lambda: SimpleNamespace(logger=logging.getLogger(__name__)),
    )
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    auth_file = home / "auth.json"

    def write_entry(entry_id, token):
        auth_file.write_text(json.dumps({
            "version": 1,
            "credential_pool": {"openai-codex": [{
                "id": entry_id, "label": entry_id, "auth_type": "oauth", "priority": 0,
                "source": "manual:device_code", "access_token": token,
                "refresh_token": "unused-refresh-token",
            }]},
        }), encoding="utf-8")

    write_entry("account-a", "old-a-token")
    cached_pool = load_pool("openai-codex")
    write_entry("account-b", "new-b-token")
    swap = Mock(return_value=True)
    agent = SimpleNamespace(
        provider="openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        model="gpt-6.1-sol",
        api_key="unmatched-wrapper-token",
        _credential_pool=cached_pool,
        _credential_pool_entry_id=None,
        _is_entitlement_failure=lambda *_: False,
        _swap_credential=swap,
    )

    recovered, _ = recover_with_credential_pool(
        agent, status_code=401, has_retried_429=False,
        error_context={"reason": "token_invalidated"},
    )

    assert recovered is False
    swap.assert_not_called()
    assert agent._credential_pool is cached_pool
