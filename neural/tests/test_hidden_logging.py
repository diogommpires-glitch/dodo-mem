import logging
import logging.config

import uvicorn

from dodo_mem_neural.service import log_configuration


def test_hidden_service_without_console_keeps_bounded_diagnostics(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.stdout", None)
    monkeypatch.setattr("sys.stderr", None)
    uvicorn.Config(
        "fixture:app", access_log=False, log_level="warning", log_config=log_configuration(tmp_path)
    )
    logger = logging.getLogger("uvicorn.error")
    logger.warning("hidden startup fixture")
    handler = logging.getLogger("uvicorn").handlers[0]
    handler.flush()
    assert "hidden startup fixture" in (tmp_path / "service.log").read_text()
    assert handler.maxBytes == 1024 * 1024 and handler.backupCount == 1
    handler.close()
    logging.getLogger("uvicorn").removeHandler(handler)
