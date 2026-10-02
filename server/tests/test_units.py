import stat
from pathlib import Path

import pytest

from degas.blobs import BlobStore
from degas.colab.bundle import build_bundle
from degas.colab.cli import ColabCli, ColabError
from degas.config import load_config
from degas.db import Database
from degas.families.base import SpecError, spec_assets
from degas.families.sdxl import Sdxl
from degas.families.wan22 import Wan22

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
    assert (params["scheduler"], params["schedule"]) == ("dpmpp_2m", "karras")
    assert spec["mode"] == "t2i"


def test_sdxl_inpaint_variant_defaults_to_dpmpp_2m_sde() -> None:
    def sampler(variant: str, mode: str) -> object:
        return Sdxl().param_schema(variant, mode)["properties"]["scheduler"]["default"]

    assert sampler("inpaint", "inpaint") == "dpmpp_2m_sde"
    assert sampler("base", "inpaint") == "dpmpp_2m"


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ({"scheduler": "dpmpp_3m_sde", "schedule": "exponential"}, ("dpmpp_3m_sde", "exponential")),
        ({"scheduler": "ddim", "schedule": "karras"}, ("ddim", "default")),
        # From before the schedule was its own control.
        ({"scheduler": "dpmpp_2m_karras"}, ("dpmpp_2m", "karras")),
        ({"scheduler": "euler"}, ("euler", "default")),
    ],
)
def test_sdxl_schedule(given: dict[str, str], expected: tuple[str, str]) -> None:
    spec = Sdxl().validate({"model": {"path": "m"}, "params": {"prompt": "x", **given}})
    assert (spec["params"]["scheduler"], spec["params"]["schedule"]) == expected


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"params": {"prompt": "  "}}, "empty"),
        ({"params": {"prompt": "x", "scheduler": "nope"}}, "one of"),
        ({"params": {"prompt": "x", "steps": "many"}}, "number"),
        ({"params": {"prompt": "x", "width": 2048, "height": 2048}}, "pixel"),
        ({"mode": "t2v"}, "can.t do"),
        ({"model": None}, "model"),
    ],
)
def test_sdxl_validate_rejects(change: dict[str, object], message: str) -> None:
    spec = {"model": {"path": "m"}, "params": {"prompt": "x"}, **change}
    with pytest.raises(SpecError, match=message):
        Sdxl().validate(spec)


def test_sdxl_loras() -> None:
    spec = Sdxl().validate(
        {
            "model": {"path": "m", "size": 5},
            "params": {"prompt": "x"},
            "loras": [{"path": "a", "weight": -3}, {"path": "b"}],
        }
    )
    assert spec["loras"] == [
        {"path": "a", "weight": -2.0, "size": None},
        {"path": "b", "weight": 1.0, "size": None},
    ]
    assert spec_assets(spec) == [
        {"path": "m", "size": 5, "kind": "model"},
        {"path": "configs/sdxl/stable-diffusion-xl-base-1.0", "size": None, "kind": "config"},
        {"path": "vae/sdxl/sdxl-vae-fp16-fix", "size": None, "kind": "vae"},
        {"path": "a", "size": None, "kind": "lora"},
        {"path": "b", "size": None, "kind": "lora"},
    ]


@pytest.mark.parametrize(
    ("loras", "message"),
    [
        ("a", "list"),
        ([{"path": "a"}, {"path": "a"}], "twice"),
        ([{"weight": 1}], "path"),
        ([{"path": str(i)} for i in range(9)], "At most 8"),
        ([{"path": "a", "weight": "high"}], "number"),
    ],
)
def test_sdxl_rejects_bad_loras(loras: object, message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Sdxl().validate({"model": {"path": "m"}, "params": {"prompt": "x"}, "loras": loras})


def test_asset_index_keeps_previews_referenced(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite")
    lora = {"path": "loras/sdxl/a.safetensors", "kind": "lora", "drive_file_id": "a"}
    db.replace_assets([{**lora, "preview_thumb": "p1", "sidecar": {"label": "A"}}])
    assert db.list_assets(kind="lora")[0]["sidecar"] == {"label": "A"}
    db.replace_assets([{**lora, "preview_thumb": "p2"}])
    refs = db.conn.execute("SELECT blob_sha FROM blob_refs WHERE ref_type = 'asset'").fetchall()
    assert [r[0] for r in refs] == ["p2"]
    db.close()


def test_database_migrates_old_asset_tables(tmp_path: Path) -> None:
    import sqlite3  # noqa: PLC0415

    conn = sqlite3.connect(tmp_path / "db.sqlite")
    conn.execute(
        "CREATE TABLE assets (path TEXT PRIMARY KEY, family TEXT, kind TEXT NOT NULL,"
        " drive_file_id TEXT NOT NULL, size INTEGER, mtime TEXT, md5 TEXT, sidecar TEXT,"
        " preview_thumb TEXT, indexed_at TEXT NOT NULL)"
    )
    conn.close()
    db = Database(tmp_path / "db.sqlite")
    db.replace_assets([{"path": "p", "kind": "lora", "drive_file_id": "a", "sidecar_rev": "r"}])
    assert db.get_asset("p")["sidecar_rev"] == "r"  # type: ignore[index]
    db.close()


def test_database_upgrades_sdxl_schedules(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite")
    specs = [
        {"family": "sdxl", "params": {"scheduler": "dpmpp_2m_karras"}},
        {"family": "sdxl", "params": {"scheduler": "euler_a"}},
        {"family": "sdxl", "params": {"scheduler": "unipc", "schedule": "exponential"}},
        {"family": "wan22", "params": {"scheduler": "unipc"}},
    ]
    ids = [db.insert_job(spec, [1])["id"] for spec in specs]
    db.close()
    db = Database(tmp_path / "db.sqlite")
    params = [db.get_job(id_)["spec"]["params"] for id_ in ids]  # type: ignore[index]
    assert params == [
        {"scheduler": "dpmpp_2m", "schedule": "karras"},
        {"scheduler": "euler_a", "schedule": "default"},
        {"scheduler": "unipc", "schedule": "exponential"},
        {"scheduler": "unipc"},
    ]
    db.close()


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


WAN_SPEC = {
    "family": "wan22",
    "variant": "t2v-a14b",
    "mode": "t2v",
    "model": {"path": "models/wan22/t2v-a14b/Wan2.2-T2V-A14B-Diffusers"},
    "params": {"prompt": "waves", "num_frames": 50, "width": 1283, "height": 720},
}


def test_wan_validate() -> None:
    spec = Wan22().validate(WAN_SPEC)
    params = spec["params"]
    assert params["num_frames"] == 49  # 4k + 1
    assert (params["width"], params["height"]) == (1280, 720)
    assert (params["fps"], params["steps"], params["cfg_low"]) == (16, 40, 3.0)
    assert spec["inputs"] == {}


def test_wan_i2v_needs_a_source_and_a_matching_model() -> None:
    i2v = {
        **WAN_SPEC,
        "variant": "ti2v-5b",
        "mode": "i2v",
        "model": {"path": "models/wan22/ti2v-5b"},
    }
    with pytest.raises(SpecError, match="source"):
        Wan22().validate(i2v)
    source = "sha256:" + "a" * 64
    spec = Wan22().validate({**i2v, "inputs": {"source": source, "transforms": {"x": 1}}})
    assert spec["inputs"] == {"source": source, "fit": "crop"}  # transforms are the server's
    assert "cfg_low" not in spec["params"]
    with pytest.raises(SpecError, match=r"isn.t a Wan 2\.2 TI2V 5B model"):
        Wan22().validate({**i2v, "model": WAN_SPEC["model"], "inputs": {"source": source}})
    with pytest.raises(SpecError, match="can't do"):
        Wan22().validate({**WAN_SPEC, "mode": "i2v"})


def test_wan_paired_loras() -> None:
    loras = [
        {"high": {"path": "h", "weight": 0.5}, "low": {"path": "l", "size": 3}},
        {"low": {"path": "only-low", "weight": 3}},
    ]
    spec = Wan22().validate({**WAN_SPEC, "loras": loras})
    assert spec["loras"] == [
        {
            "high": {"path": "h", "weight": 0.5, "size": None},
            "low": {"path": "l", "weight": 1.0, "size": 3},
        },
        {"low": {"path": "only-low", "weight": 2.0, "size": None}},
    ]
    assert [a["path"] for a in spec_assets(spec)] == [
        WAN_SPEC["model"]["path"],
        "h",
        "l",
        "only-low",
    ]
    # One file for both experts (a single-file Wan 2.1 LoRA) is fetched once.
    both = {"high": {"path": "h", "weight": 0.5}, "low": {"path": "h"}}
    spec = Wan22().validate({**WAN_SPEC, "loras": [both]})
    assert spec["loras"] == [
        {
            "high": {"path": "h", "weight": 0.5, "size": None},
            "low": {"path": "h", "weight": 1.0, "size": None},
        }
    ]
    assert [a["path"] for a in spec_assets(spec)] == [WAN_SPEC["model"]["path"], "h"]
    with pytest.raises(SpecError, match="twice"):
        Wan22().validate({**WAN_SPEC, "loras": [both, {"low": {"path": "h"}}]})
    with pytest.raises(SpecError, match="high-noise or low-noise"):
        Wan22().validate({**WAN_SPEC, "loras": [{"path": "single"}]})
