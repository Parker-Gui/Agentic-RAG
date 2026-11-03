import subprocess
from pathlib import Path

import app.services.crawler.fetch.browser as browser


def test_linux_candidates_include_linux_chrome_paths(monkeypatch) -> None:
    monkeypatch.setattr(browser.sys, "platform", "linux", raising=False)

    assert browser._chrome_binary_candidates()[:2] == (
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
    )


def test_windows_candidates_include_windows_chrome_paths(monkeypatch) -> None:
    monkeypatch.setattr(browser.sys, "platform", "win32", raising=False)

    candidates = browser._chrome_binary_candidates()

    assert "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" in candidates
    assert "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe" in candidates


def test_chrome_major_version_prefers_env_override(monkeypatch) -> None:
    monkeypatch.setenv("MONITOR_CHROME_MAJOR", "138")
    monkeypatch.setattr(browser, "_version_from_cli", lambda _: 151)

    assert browser._chrome_major_version("/usr/bin/google-chrome") == 138


def test_chrome_major_version_on_linux_uses_cli(monkeypatch) -> None:
    monkeypatch.delenv("MONITOR_CHROME_MAJOR", raising=False)
    monkeypatch.setattr(browser.sys, "platform", "linux", raising=False)
    monkeypatch.setattr(
        browser.subprocess,
        "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, "Google Chrome 151.0.7922.174 \n", ""),
    )

    assert browser._chrome_major_version("/usr/bin/google-chrome") == 151


def test_chrome_major_version_on_windows_reads_versioned_install_folder(monkeypatch, tmp_path: Path) -> None:
    """`chrome.exe --version` exits 0 with no output on Windows, so the version
    has to come from the registry or the versioned install folder."""
    monkeypatch.delenv("MONITOR_CHROME_MAJOR", raising=False)
    monkeypatch.setattr(browser.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(browser, "_version_from_registry", lambda _: None)
    (tmp_path / "151.0.7922.174").mkdir()
    (tmp_path / "SetupMetrics").mkdir()

    assert browser._chrome_major_version(str(tmp_path / "chrome.exe")) == 151


def test_chrome_major_version_undetectable_returns_none(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MONITOR_CHROME_MAJOR", raising=False)
    monkeypatch.setattr(browser.sys, "platform", "win32", raising=False)
    monkeypatch.setattr(browser, "_version_from_registry", lambda _: None)
    monkeypatch.setattr(browser, "_version_from_cli", lambda _: None)

    assert browser._chrome_major_version(str(tmp_path / "chrome.exe")) is None


def test_version_from_error_reads_the_version_chromedriver_refused() -> None:
    exc = Exception(
        "session not created: This version of ChromeDriver only supports Chrome "
        "version 152\nCurrent browser version is 151.0.7922.174"
    )

    assert browser._version_from_error(exc) == 151


def test_version_from_error_on_unrelated_failure_returns_none() -> None:
    assert browser._version_from_error(Exception("connection refused")) is None
