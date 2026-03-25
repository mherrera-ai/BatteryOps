from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path

APP_ENV_VAR = "BATTERYOPS_STREAMLIT_APP"
APP_PACKAGE_RESOURCE = "streamlit_app.py"
APP_WRAPPER_PATH = Path("app") / APP_PACKAGE_RESOURCE
APP_SOURCE_PATH = Path("src") / "batteryops" / APP_PACKAGE_RESOURCE


def launch_demo(argv: list[str] | None = None) -> int:
    """Launch the Streamlit demo from the checkout when possible."""
    args = argv if argv is not None else sys.argv[1:]
    explicit_path = os.environ.get(APP_ENV_VAR)
    if explicit_path:
        app_path = _resolve_explicit_app_path(explicit_path)
        return _launch_streamlit(app_path, args, cwd=_resolve_checkout_app_cwd(app_path))

    local_app_path = _resolve_local_demo_app_path()
    if local_app_path is not None:
        return _launch_streamlit(local_app_path, args, cwd=_resolve_repo_root())

    app_resource = files("batteryops").joinpath(APP_PACKAGE_RESOURCE)
    with as_file(app_resource) as app_path:
        resolved_app_path = Path(app_path).resolve()
        # Installed wheels/sdists should keep the caller's working directory so the app can
        # resolve local artifacts and processed parquet relative to the launch workspace.
        return _launch_streamlit(resolved_app_path, args)


def _resolve_explicit_app_path(explicit_path: str) -> Path:
    app_path = Path(explicit_path).expanduser().resolve()
    if app_path.is_file():
        return app_path

    raise FileNotFoundError(
        f"{APP_ENV_VAR} points to a missing Streamlit app: {app_path}"
    )


def _launch_streamlit(
    app_path: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
) -> int:
    with _temporary_working_directory(cwd):
        completed = subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path), *args],
            check=False,
        )
    return completed.returncode


def _resolve_local_demo_app_path() -> Path | None:
    """Resolve the checked-out Streamlit wrapper when running from the repo."""
    repo_root = _resolve_repo_root()
    app_path = repo_root / APP_WRAPPER_PATH
    if app_path.is_file():
        return app_path.resolve()

    return None


def _resolve_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_checkout_app_cwd(app_path: Path) -> Path | None:
    repo_root = _resolve_repo_root()
    checkout_app_paths = (
        (repo_root / APP_WRAPPER_PATH).resolve(),
        (repo_root / APP_SOURCE_PATH).resolve(),
    )
    if app_path in checkout_app_paths:
        return repo_root
    return None


@contextmanager
def _temporary_working_directory(cwd: Path | None):
    if cwd is None:
        yield
        return

    previous_cwd = Path.cwd()
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def main() -> None:
    raise SystemExit(launch_demo())


if __name__ == "__main__":
    main()
