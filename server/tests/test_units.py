import stat
from pathlib import Path

import pytest

from degas.blobs import BlobStore
from degas.colab.bundle import build_bundle
from degas.colab.cli import ColabCli, ColabError
from degas.config import load_config
from degas.db import Database
from degas.families.base import SpecError
from degas.families.sdxl import Sdxl

# -- colab CLI -----------------------------------------------------------------------------

FAKE_COLAB = """#!/bin/sh
case "$1" in
  status) if [ -e "$(dirname "$0")/alive" ]; then
            printf '\\033[1m[degas] 10.0.0.1 | Hardware: L4 | Status: IDLE\\033[0m\\n'
          else echo "[colab] Session 'degas' not found."; fi ;;
  exec) cat; echo; echo done ;;
  stop) echo "[colab] Stopping session 'degas'..." ;;
  *) echo "boom"; exit 3 ;;
esac
"""


@pytest.fixture
def colab(tmp_path: Path) -> ColabCli:
    binary = tmp_path / "colab"
    binary.write_text(FAKE_COLAB)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    return ColabCli(str(binary), "degas")


async def test_colab_status(colab: ColabCli, tmp_path: Path) -> None:
    assert await colab.is_alive() is False
    (tmp_path / "alive").touch()
    assert await colab.is_alive() is True


async def test_colab_exec_sends_code_on_stdin(colab: ColabCli) -> None:
    out = await colab.exec("print('hi')")
    assert "print('hi')" in out
    assert "done" in out


async def test_colab_errors(colab: ColabCli) -> None:
    with pytest.raises(ColabError, match="exit 3"):
        await colab.new("L4", high_mem=False)


def test_proxy_command() -> None:
    cli = ColabCli("colab", "degas", auth="adc")
    assert cli.proxy_command("/k") == "colab --auth adc ssh --proxy-mode -s degas -i /k"


# -- SDXL descriptor -----------------------------------------------------------------------


def test_sdxl_validate_fills_defaults_and_clamps() -> None:
    spec = Sdxl().validate(
        {
            "family": "sdxl",
            "model": {"path": "models/sdxl/a.safetensors"},
            "params": {"prompt": "cat", "width": 1023, "steps": 500, "cfg": 7},
        }
    )
    params = spec["params"]
    assert (params["width"], params["height"]) == (1016, 1024)
    assert params["steps"] == 100
    assert params["cfg"] == 7.0
    assert params["scheduler"] == "dpmpp_2m_karras"
    assert spec["mode"] == "t2i"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"params": {"prompt": "  "}}, "empty"),
        ({"params": {"prompt": "x", "scheduler": "nope"}}, "one of"),
        ({"params": {"prompt": "x", "steps": "many"}}, "number"),
        ({"params": {"prompt": "x", "width": 2048, "height": 2048}}, "pixel"),
        ({"mode": "inpaint"}, "not supported"),
        ({"model": None}, "model"),
    ],
)
def test_sdxl_validate_rejects(change: dict[str, object], message: str) -> None:
    spec = {"model": {"path": "m"}, "params": {"prompt": "x"}, **change}
    with pytest.raises(SpecError, match=message):
        Sdxl().validate(spec)


# -- storage, config, bundle ---------------------------------------------------------------


def test_blob_store(tmp_path: Path) -> None:
    store = BlobStore(tmp_path)
    sha = store.put(b"data", "image/png")
    assert store.put(b"data", "image/png") == sha
    path = store.path(sha)
    assert path is not None
    assert path.name == f"{sha}.png"
    assert store.path("../../etc/passwd") is None
    assert store.thumb(sha) is None  # not an image


def test_results_expire_when_their_session_ends(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite")
    session = db.create_session("T4", False)
    job = db.insert_job({"family": "sdxl"}, [1])
    db.update_job(job["id"], session_id=session["id"])
    result = db.insert_result(job["id"], 0, "a" * 64, "image/png", 1, 8, 8)
    assert result["expires_at"] is None
    db.end_session(session["id"], "stopped")
    expires = db.get_result(result["id"])["expires_at"]  # type: ignore[index]
    assert expires > db.get_session(session["id"])["ended_at"]  # type: ignore[index]
    db.close()


def test_load_config_resolves_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "degas.toml").write_text(
        'data_dir = "var"\nidle_timeout_min = 5\n[drive]\nclient_file = "client.json"\n'
    )
    config = load_config(tmp_path / "degas.toml")
    assert config.data_dir == tmp_path / "var"
    assert config.drive.client_file == tmp_path / "client.json"
    assert config.ssh_key == tmp_path / "var" / "ssh" / "id_ed25519"
    assert config.idle_timeout_min == 5


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    (tmp_path / "degas.toml").write_text("[colab]\ngpu = 'T4'\n")
    with pytest.raises(ValueError, match="gpu"):
        load_config(tmp_path / "degas.toml")


def test_bundle_is_reproducible(tmp_path: Path) -> None:
    pkg = tmp_path / "degas_worker"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("x = 1\n")
    (pkg / "__pycache__").mkdir()
    (pkg / "__pycache__" / "junk.pyc").write_bytes(b"junk")
    first = build_bundle(pkg)
    (pkg / "__init__.py").touch()
    assert build_bundle(pkg).sha256 == first.sha256
    (pkg / "__init__.py").write_text("x = 2\n")
    assert build_bundle(pkg).sha256 != first.sha256
    assert b"junk" not in first.data
