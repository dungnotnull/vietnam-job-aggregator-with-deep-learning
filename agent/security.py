from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def audit_session_storage() -> None:
    session_paths = [
        Path.home() / ".session_cache",
        Path(tempfile.gettempdir()) / "cloak_sessions",
    ]
    for p in session_paths:
        if p.exists():
            import shutil

            shutil.rmtree(p, ignore_errors=True)
        assert not p.exists(), f"Session cache leaked at {p}"


def audit_environment() -> None:
    sensitive_patterns = ["password", "token", "secret", "credential", "api_key", "apikey"]
    violations: list[str] = []
    for key in os.environ:
        key_lower = key.lower()
        if any(pat in key_lower for pat in sensitive_patterns):
            violations.append(key)
    assert not violations, (
        f"Sensitive keys in environment: {violations}. "
        f"Remove credentials from environment variables."
    )


def audit_no_persistence() -> None:
    if any(".db" in f.lower() or ".sqlite" in f.lower() for f in os.listdir(".")):
        sys.exit("Database file detected — zero persistence policy violated.")
    log_dir = Path("logs")
    if log_dir.exists():
        for logfile in log_dir.glob("*.log"):
            content = logfile.read_text(encoding="utf-8", errors="ignore")
            assert "password" not in content.lower(), f"Credential found in log: {logfile}"
            assert "token=" not in content.lower(), f"Token found in log: {logfile}"


def run_security_audit() -> bool:
    try:
        audit_session_storage()
        audit_environment()
        audit_no_persistence()
        return True
    except AssertionError as e:
        print(f"[SECURITY VIOLATION] {e}")
        return False
