#!/usr/bin/env python3
"""End-to-end test for engine git-update checks, backups and rebuilds.

Creates a throwaway git upstream plus a tiny CMake project shaped like an
engine (binary ``llama-server``) and exercises the real helper functions in
router.py: check, dirty-tree refusal, fast-forward + backup + compile, and the
rollback path when the new source does not compile.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import router  # noqa: E402


def run(command, cwd=None, check=True, env=None):
    proc = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed ({proc.returncode}): {' '.join(map(str, command))}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def git(repo, *args, check=True):
    return run(["git", "-C", str(repo), *args], check=check)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def version_main(version: int, broken: bool = False) -> str:
    if broken:
        return "#error intentionally broken by engine-update-e2e\n"
    return (
        "#include <iostream>\n"
        "int main(int argc, char **argv) {\n"
        "  (void)argc; (void)argv;\n"
        f'  std::cout << "version: {version}" << std::endl;\n'
        "  return 0;\n"
        "}\n"
    )


CMAKELISTS = """cmake_minimum_required(VERSION 3.10)
project(fakeengine CXX)
add_executable(llama-server main.cpp)
set_target_properties(llama-server PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
"""


def engine_cfg(work: Path, models: Path, preset: Path) -> dict:
    return {
        "kind": "llama_router",
        "cwd": str(work),
        "command": [
            str(work / "build" / "bin" / "llama-server"),
            "--models-preset",
            str(preset),
            "--models-dir",
            str(models),
            "--host",
            "{host}",
            "--port",
            "{port}",
        ],
        "models": {
            "kind": "llama_dir",
            "dir": str(models),
            "preset": str(preset),
        },
    }


def commit_source(clone: Path, version: int, broken: bool = False) -> str:
    write(clone / "main.cpp", version_main(version, broken=broken))
    git(clone, "add", "main.cpp")
    git(clone, "commit", "-m", f"version {version}" + (" broken" if broken else ""))
    git(clone, "push", "origin", "master")
    return git(clone, "rev-parse", "HEAD").stdout.strip()


def main() -> int:
    base = ROOT / "tmp" / "engine-update-e2e"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)

    models = base / "models"
    models.mkdir()
    preset = base / "models.ini"
    preset.write_text("version = 1\n", encoding="utf-8")

    upstream = base / "upstream.git"
    run(["git", "init", "--bare", "--initial-branch=master", str(upstream)])
    seed = base / "seed"
    run(["git", "init", "--initial-branch=master", str(seed)])
    git(seed, "config", "user.name", "Engine Update Test")
    git(seed, "config", "user.email", "engine-update@example.invalid")
    git(seed, "remote", "add", "origin", str(upstream))
    write(seed / "CMakeLists.txt", CMAKELISTS)
    write(seed / "main.cpp", version_main(1))
    git(seed, "add", "CMakeLists.txt", "main.cpp")
    git(seed, "commit", "-m", "initial")
    git(seed, "push", "-u", "origin", "master")

    work = base / "work"
    run(["git", "clone", str(upstream), str(work)])
    git(work, "config", "user.name", "Engine Update Test")
    git(work, "config", "user.email", "engine-update@example.invalid")
    run(["cmake", "-S", str(work), "-B", str(work / "build")])
    run(["cmake", "--build", str(work / "build"), "--target", "llama-server"])
    binary = work / "build" / "bin" / "llama-server"
    assert binary.is_file(), binary
    cfg = engine_cfg(work, models, preset)

    developer = base / "developer"
    run(["git", "clone", str(upstream), str(developer)])
    git(developer, "config", "user.name", "Engine Update Test")
    git(developer, "config", "user.email", "engine-update@example.invalid")

    # -- proxy environment handling ------------------------------------
    proxy = {
        "enabled": True,
        "type": "socks5",
        "host": "10.0.0.9",
        "port": 1080,
        "username": "user",
        "password": "p@ss",
    }
    env = router._git_update_env(proxy, True)
    expected = "socks5h://user:p%40ss@10.0.0.9:1080"
    assert env["HTTPS_PROXY"] == expected, env.get("HTTPS_PROXY")
    assert env["ALL_PROXY"] == expected, env.get("ALL_PROXY")
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    inherited = {name: os.environ.get(name) for name in ("HTTPS_PROXY", "NO_PROXY")}
    try:
        os.environ.update({"HTTPS_PROXY": "http://inherited:1", "NO_PROXY": "*"})
        direct = router._git_update_env(proxy, False)
        assert "HTTPS_PROXY" not in direct, direct.get("HTTPS_PROXY")
        assert "NO_PROXY" not in direct, direct.get("NO_PROXY")
    finally:
        for name, value in inherited.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    # -- new upstream commit is detected -------------------------------
    latest2 = commit_source(developer, 2)
    state = router.engine_update_check("llama", cfg)
    assert state["update_available"], state
    assert state["behind"] == 1 and state["ahead"] == 0, state
    assert state["build"] and state["build"]["ready"], state["build"]
    assert state["build"]["target"] == "llama-server", state["build"]
    assert state["can_install"], state["blockers"]

    # -- install refuses a dirty tree ----------------------------------
    before = sha256(binary)
    head_before = git(work, "rev-parse", "HEAD").stdout.strip()
    write(work / "main.cpp", version_main(1) + "// local edit\n")
    dirty = router.engine_update_check("llama", cfg)
    assert dirty["dirty"] and dirty["dirty_files"] == 1, dirty
    try:
        router.install_engine_update("llama", cfg)
    except router.EngineUpdateError as exc:
        assert "未提交" in str(exc), exc
    else:
        raise AssertionError("install accepted a dirty working tree")
    git(work, "checkout", "--", "main.cpp")
    assert sha256(binary) == before
    assert git(work, "rev-parse", "HEAD").stdout.strip() == head_before

    # -- successful install: fast-forward, backup, compile -------------
    result = router.install_engine_update("llama", cfg)
    assert result["installed"] and result["compiled"], result
    backup = Path(result["backup"])
    assert backup.is_file(), backup
    assert sha256(backup) == before, "backup must be the pre-update binary"
    assert git(work, "rev-parse", "HEAD").stdout.strip() == latest2
    assert sha256(binary) != before
    version = run([str(binary), "--version"]).stdout
    assert "version: 2" in version, version
    after = router.engine_update_check("llama", cfg)
    assert not after["update_available"], after
    assert after["up_to_date"], after
    try:
        router.install_engine_update("llama", cfg)
    except router.EngineUpdateError as exc:
        assert "最新" in str(exc), exc
    else:
        raise AssertionError("install accepted an up-to-date tree")

    # -- failed build: binary restored and source rolled back ----------
    head_before = git(work, "rev-parse", "HEAD").stdout.strip()
    before = sha256(binary)
    commit_source(developer, 3, broken=True)
    broken = router.engine_update_check("llama", cfg)
    assert broken["update_available"] and broken["behind"] == 1, broken
    try:
        router.install_engine_update("llama", cfg)
    except router.EngineUpdateError as exc:
        message = str(exc)
        assert "编译失败" in message, message
        assert "已恢复原二进制" in message, message
        assert "已回滚源码" in message, message
    else:
        raise AssertionError("install accepted a source tree that does not compile")
    assert sha256(binary) == before, "binary was not restored after a failed build"
    assert git(work, "rev-parse", "HEAD").stdout.strip() == head_before
    retry = router.engine_update_check("llama", cfg)
    assert retry["update_available"], retry

    # -- non-git working directory is reported clearly -----------------
    plain = Path(tempfile.mkdtemp(prefix="engine-update-plain-"))
    try:
        plain_cfg = engine_cfg(plain, models, preset)
        try:
            router.engine_update_check("llama", plain_cfg)
        except router.EngineUpdateError as exc:
            assert "不是 git 仓库" in str(exc), exc
        else:
            raise AssertionError("non-git cwd was accepted")
    finally:
        shutil.rmtree(plain, ignore_errors=True)

    print("PASS: engine update check/install/rollback checks")
    print(json.dumps(
        {
            "repo": result["repo"],
            "previous_head": result["previous_head"],
            "head": result["head"],
            "backup": result["backup"],
            "compiled": result["compiled"],
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
