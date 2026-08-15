from __future__ import annotations

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB


def _compressor() -> ContextCompressor:
    return ContextCompressor(model="test/model", quiet_mode=True)


def test_bind_session_state_restores_both_durable_counts(tmp_path) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("resumed", source="webui")
        db.archive_and_compact(
            "resumed",
            [{"role": "user", "content": "batch"}],
            compaction_kind="batch",
        )
        db.archive_and_compact(
            "resumed",
            [{"role": "user", "content": "micro"}],
            compaction_kind="micro",
        )

        compressor = _compressor()
        compressor.bind_session_state(db, "resumed")

        assert compressor.compression_count == 1
        assert compressor.micro_compaction_count == 1
    finally:
        db.close()


def test_micro_sync_counts_only_after_durable_commit(tmp_path, monkeypatch) -> None:
    db = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("micro", source="webui")
        compressor = _compressor()
        compressor.bind_session_state(db, "micro")
        messages = [{"role": "user", "content": "summary"}]

        assert compressor._sync_micro_compact_to_db(messages) is True
        assert compressor.micro_compaction_count == 1
        assert db.get_compaction_counts("micro") == (0, 1)

        def _boom(*_args, **_kwargs):
            raise RuntimeError("write failed")

        monkeypatch.setattr(db, "archive_and_compact", _boom)
        assert compressor._sync_micro_compact_to_db(messages) is False
        assert compressor.micro_compaction_count == 1
    finally:
        db.close()
