from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import batteryops.cli as cli_module
import batteryops.streamlit_app as streamlit_app_module
from batteryops.cli import launch_demo


def test_launch_demo_prefers_repo_local_streamlit_wrapper(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_cwd = Path.cwd()

    def fake_run(cmd: list[str], check: bool) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = Path.cwd()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("batteryops.cli.subprocess.run", fake_run)

    result = launch_demo(["--server.headless=true"])

    assert result == 0
    assert captured["check"] is False
    assert captured["cmd"][:4] == [
        __import__("sys").executable,
        "-m",
        "streamlit",
        "run",
    ]
    assert captured["cmd"][4] == str(
        (Path(cli_module.__file__).resolve().parents[2] / "app" / "streamlit_app.py").resolve()
    )
    assert captured["cmd"][5:] == ["--server.headless=true"]
    assert captured["cwd"] == Path(cli_module.__file__).resolve().parents[2]
    assert Path.cwd() == original_cwd


def test_launch_demo_falls_back_to_packaged_streamlit_app(monkeypatch) -> None:
    captured: dict[str, object] = {}
    original_cwd = Path.cwd()

    def fake_run(cmd: list[str], check: bool) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = Path.cwd()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("batteryops.cli.subprocess.run", fake_run)
    monkeypatch.setattr("batteryops.cli._resolve_local_demo_app_path", lambda: None)

    result = launch_demo(["--server.headless=true"])

    assert result == 0
    assert captured["check"] is False
    assert captured["cmd"][:4] == [
        __import__("sys").executable,
        "-m",
        "streamlit",
        "run",
    ]
    assert captured["cmd"][4] == str(Path(streamlit_app_module.__file__).resolve())
    assert captured["cmd"][5:] == ["--server.headless=true"]
    assert captured["cwd"] == original_cwd
    assert Path.cwd() == original_cwd


def test_launch_demo_packaged_install_keeps_launch_workspace(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    installed_root = tmp_path / "venv" / "lib" / "python3.13"
    packaged_app = installed_root / "site-packages" / "batteryops" / "streamlit_app.py"
    packaged_app.parent.mkdir(parents=True)
    packaged_app.write_text("print('packaged')\n", encoding="utf-8")
    original_cwd = Path.cwd()

    class _FakeTraversable:
        def joinpath(self, name: str) -> Path:
            assert name == "streamlit_app.py"
            return packaged_app

    @contextmanager
    def fake_as_file(resource: Path):
        yield resource

    def fake_run(cmd: list[str], check: bool) -> SimpleNamespace:
        captured["cmd"] = cmd
        captured["check"] = check
        captured["cwd"] = Path.cwd()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("batteryops.cli.subprocess.run", fake_run)
    monkeypatch.setattr("batteryops.cli._resolve_local_demo_app_path", lambda: None)
    monkeypatch.setattr("batteryops.cli._resolve_repo_root", lambda: installed_root)
    monkeypatch.setattr("batteryops.cli.files", lambda _: _FakeTraversable())
    monkeypatch.setattr("batteryops.cli.as_file", fake_as_file)

    result = launch_demo(["--server.headless=true"])

    assert result == 0
    assert captured["check"] is False
    assert captured["cmd"][:4] == [
        __import__("sys").executable,
        "-m",
        "streamlit",
        "run",
    ]
    assert captured["cmd"][4] == str(packaged_app.resolve())
    assert captured["cmd"][5:] == ["--server.headless=true"]
    assert captured["cwd"] == original_cwd
    assert Path.cwd() == original_cwd
