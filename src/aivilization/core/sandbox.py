from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from aivilization.config import AIvilizationConfig


class ExecResult(BaseModel):
    """Result of executing code/commands in a sandbox container."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_ms: float = 0
    timed_out: bool = False
    error: str | None = None


# ────────────────────────────────────────────────
# Local (subprocess) sandbox — no Docker required
# ────────────────────────────────────────────────

class LocalSandbox:
    """
    A lightweight sandbox using subprocess + temp directories.
    Used as a fallback when Docker is not available.
    Each agent gets an isolated workspace directory.
    """

    def __init__(self, base_dir: Path, timeout: int = 30) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._workspaces: dict[str, Path] = {}

    def _workspace(self, agent_id: str) -> Path:
        if agent_id not in self._workspaces:
            ws = self.base_dir / agent_id[:12]
            ws.mkdir(parents=True, exist_ok=True)
            self._workspaces[agent_id] = ws
        return self._workspaces[agent_id]

    def exec_python(self, agent_id: str, code: str, timeout: int | None = None) -> ExecResult:
        timeout = timeout or self.timeout
        ws = self._workspace(agent_id)
        script = ws / "_exec.py"
        script.write_text(code, encoding="utf-8")

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["python3", str(script)],
                capture_output=True,
                timeout=timeout,
                cwd=str(ws),
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            duration = (time.monotonic() - start) * 1000
            return ExecResult(
                stdout=result.stdout.decode("utf-8", errors="replace")[:10000],
                stderr=result.stderr.decode("utf-8", errors="replace")[:5000],
                exit_code=result.returncode,
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(
                stdout="",
                stderr="Execution timed out",
                exit_code=-1,
                duration_ms=duration,
                timed_out=True,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(error=str(e), exit_code=-1, duration_ms=duration)

    def exec_shell(self, agent_id: str, command: str, timeout: int | None = None) -> ExecResult:
        timeout = timeout or self.timeout
        ws = self._workspace(agent_id)

        start = time.monotonic()
        try:
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                capture_output=True,
                timeout=timeout,
                cwd=str(ws),
            )
            duration = (time.monotonic() - start) * 1000
            return ExecResult(
                stdout=result.stdout.decode("utf-8", errors="replace")[:10000],
                stderr=result.stderr.decode("utf-8", errors="replace")[:5000],
                exit_code=result.returncode,
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(
                stderr="Execution timed out", exit_code=-1,
                duration_ms=duration, timed_out=True,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(error=str(e), exit_code=-1, duration_ms=duration)

    def read_file(self, agent_id: str, path: str) -> str:
        ws = self._workspace(agent_id)
        if path.startswith("/workspace/"):
            path = path[len("/workspace/"):]
        elif path.startswith("/"):
            raise PermissionError("Can only read files within workspace")
        fp = ws / path
        if not fp.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return fp.read_text(encoding="utf-8", errors="replace")

    def write_file(self, agent_id: str, path: str, content: str) -> None:
        ws = self._workspace(agent_id)
        if path.startswith("/workspace/"):
            path = path[len("/workspace/"):]
        elif path.startswith("/"):
            raise PermissionError("Can only write files within workspace")
        fp = ws / path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")

    def list_files(self, agent_id: str, path: str = "") -> list[str]:
        ws = self._workspace(agent_id)
        if path.startswith("/workspace"):
            path = path[len("/workspace"):].lstrip("/")
        target = ws / path if path else ws
        if not target.exists():
            return []
        files = []
        for fp in target.rglob("*"):
            if fp.is_file() and fp.name != "_exec.py":
                files.append(str(fp.relative_to(ws)))
        return files

    def install_package(self, agent_id: str, package: str) -> ExecResult:
        start = time.monotonic()
        try:
            result = subprocess.run(
                ["pip", "install", "--user", package],
                capture_output=True,
                timeout=60,
            )
            duration = (time.monotonic() - start) * 1000
            return ExecResult(
                stdout=result.stdout.decode("utf-8", errors="replace"),
                stderr=result.stderr.decode("utf-8", errors="replace"),
                exit_code=result.returncode,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(error=str(e), exit_code=-1, duration_ms=duration)

    def destroy(self, agent_id: str) -> None:
        if agent_id in self._workspaces:
            ws = self._workspaces.pop(agent_id)
            shutil.rmtree(ws, ignore_errors=True)

    def destroy_all(self) -> None:
        for agent_id in list(self._workspaces.keys()):
            self.destroy(agent_id)


# ────────────────────────────────────────────────
# Docker-based sandbox
# ────────────────────────────────────────────────

class DockerSandbox:
    """Docker-based sandbox (original implementation)."""

    def __init__(self, config: AIvilizationConfig) -> None:
        self.config = config
        self._client = None
        self._containers: dict[str, Any] = {}

    @property
    def client(self):
        if self._client is None:
            import docker
            self._client = docker.from_env()
        return self._client

    def _ensure_base_image(self) -> None:
        import docker
        try:
            self.client.images.get(self.config.sandbox_base_image)
        except docker.errors.ImageNotFound:
            dockerfile_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                "docker",
            )
            if os.path.exists(os.path.join(dockerfile_dir, "Dockerfile.agent")):
                self.client.images.build(
                    path=dockerfile_dir,
                    dockerfile="Dockerfile.agent",
                    tag=self.config.sandbox_base_image,
                    rm=True,
                )
            else:
                self.client.images.pull("python:3.12-slim")

    def create_sandbox(self, agent_id: str) -> None:
        import docker
        self._ensure_base_image()
        container_name = f"aiv-{agent_id[:12]}"
        try:
            existing = self.client.containers.get(container_name)
            if existing.status != "running":
                existing.start()
            self._containers[agent_id] = existing
            return
        except docker.errors.NotFound:
            pass

        nano_cpus = int(self.config.sandbox_cpu_fraction * 1_000_000_000)
        container = self.client.containers.run(
            image=self.config.sandbox_base_image,
            name=container_name,
            detach=True,
            tty=True,
            mem_limit=f"{self.config.sandbox_memory_mb}m",
            memswap_limit=f"{self.config.sandbox_memory_mb}m",
            nano_cpus=nano_cpus,
            pids_limit=self.config.sandbox_pids_limit,
            network_disabled=True,
            working_dir="/workspace",
            security_opt=["no-new-privileges"],
        )
        self._containers[agent_id] = container

    def _get_container(self, agent_id: str):
        if agent_id not in self._containers:
            self.create_sandbox(agent_id)
        container = self._containers[agent_id]
        container.reload()
        if container.status != "running":
            container.start()
        return container

    def exec_python(self, agent_id: str, code: str, timeout: int | None = None) -> ExecResult:
        timeout = timeout or self.config.sandbox_timeout_seconds
        container = self._get_container(agent_id)

        start = time.monotonic()
        timed_out_flag: list[bool] = []
        self._write_to_container(container, "/workspace/_exec.py", code)

        try:
            exec_id = self.client.api.exec_create(
                container.id,
                ["python3", "/workspace/_exec.py"],
                workdir="/workspace",
                stdout=True,
                stderr=True,
            )

            def watchdog():
                time.sleep(timeout)
                timed_out_flag.append(True)

            timer = threading.Thread(target=watchdog, daemon=True)
            timer.start()

            output = self.client.api.exec_start(exec_id, demux=True)
            inspect = self.client.api.exec_inspect(exec_id["Id"])

            duration = (time.monotonic() - start) * 1000
            stdout = (output[0] or b"").decode("utf-8", errors="replace")
            stderr = (output[1] or b"").decode("utf-8", errors="replace")

            return ExecResult(
                stdout=stdout[:10000],
                stderr=stderr[:5000],
                exit_code=inspect.get("ExitCode", -1),
                duration_ms=duration,
                timed_out=bool(timed_out_flag),
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(
                error=str(e), exit_code=-1,
                duration_ms=duration, timed_out=bool(timed_out_flag),
            )

    def exec_shell(self, agent_id: str, command: str, timeout: int | None = None) -> ExecResult:
        timeout = timeout or self.config.sandbox_timeout_seconds
        container = self._get_container(agent_id)

        start = time.monotonic()
        try:
            result = container.exec_run(
                ["/bin/sh", "-c", command],
                workdir="/workspace",
                demux=True,
            )
            duration = (time.monotonic() - start) * 1000
            stdout_bytes, stderr_bytes = result.output
            return ExecResult(
                stdout=(stdout_bytes or b"").decode("utf-8", errors="replace")[:10000],
                stderr=(stderr_bytes or b"").decode("utf-8", errors="replace")[:5000],
                exit_code=result.exit_code,
                duration_ms=duration,
            )
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            return ExecResult(error=str(e), exit_code=-1, duration_ms=duration)

    def read_file(self, agent_id: str, path: str) -> str:
        container = self._get_container(agent_id)
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        if not path.startswith("/workspace"):
            raise PermissionError("Can only read files within /workspace")
        try:
            bits, stat = container.get_archive(path)
            buf = BytesIO()
            for chunk in bits:
                buf.write(chunk)
            buf.seek(0)
            with tarfile.open(fileobj=buf) as tar:
                members = tar.getmembers()
                if not members:
                    raise FileNotFoundError(f"No file at {path}")
                f = tar.extractfile(members[0])
                if f is None:
                    raise FileNotFoundError(f"Cannot read {path}")
                return f.read().decode("utf-8", errors="replace")
        except Exception as e:
            if "NotFound" in type(e).__name__:
                raise FileNotFoundError(f"File not found: {path}")
            raise

    def write_file(self, agent_id: str, path: str, content: str) -> None:
        container = self._get_container(agent_id)
        if not path.startswith("/"):
            path = f"/workspace/{path}"
        if not path.startswith("/workspace"):
            raise PermissionError("Can only write files within /workspace")
        parent_dir = "/".join(path.split("/")[:-1])
        container.exec_run(["mkdir", "-p", parent_dir])
        self._write_to_container(container, path, content)

    def list_files(self, agent_id: str, path: str = "/workspace") -> list[str]:
        container = self._get_container(agent_id)
        if not path.startswith("/workspace"):
            path = f"/workspace/{path}" if path else "/workspace"
        result = container.exec_run(
            ["find", path, "-type", "f", "-not", "-name", "_exec.py"],
            demux=True,
        )
        stdout = (result.output[0] or b"").decode("utf-8", errors="replace")
        files = [f.strip() for f in stdout.strip().split("\n") if f.strip()]
        return [f.replace("/workspace/", "") for f in files if f != "/workspace"]

    def install_package(self, agent_id: str, package: str) -> ExecResult:
        container = self._get_container(agent_id)
        result = container.exec_run(["pip", "install", "--user", package], demux=True)
        stdout_bytes, stderr_bytes = result.output
        return ExecResult(
            stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
            exit_code=result.exit_code,
        )

    def snapshot(self, agent_id: str) -> str:
        container = self._get_container(agent_id)
        tag = f"aiv-snapshot-{agent_id[:12]}"
        container.commit(repository=tag, tag="latest")
        return f"{tag}:latest"

    def restore(self, agent_id: str, image_tag: str) -> None:
        self.destroy(agent_id)
        container_name = f"aiv-{agent_id[:12]}"
        nano_cpus = int(self.config.sandbox_cpu_fraction * 1_000_000_000)
        container = self.client.containers.run(
            image=image_tag,
            name=container_name,
            detach=True,
            tty=True,
            mem_limit=f"{self.config.sandbox_memory_mb}m",
            memswap_limit=f"{self.config.sandbox_memory_mb}m",
            nano_cpus=nano_cpus,
            pids_limit=self.config.sandbox_pids_limit,
            network_disabled=True,
            working_dir="/workspace",
            security_opt=["no-new-privileges"],
        )
        self._containers[agent_id] = container

    def destroy(self, agent_id: str) -> None:
        if agent_id in self._containers:
            try:
                self._containers[agent_id].stop(timeout=5)
            except Exception:
                pass
            try:
                self._containers[agent_id].remove(force=True)
            except Exception:
                pass
            del self._containers[agent_id]

    def destroy_all(self) -> None:
        for agent_id in list(self._containers.keys()):
            self.destroy(agent_id)

    def _write_to_container(self, container, dest_path: str, content: str) -> None:
        data = content.encode("utf-8")
        filename = dest_path.split("/")[-1]
        dest_dir = "/".join(dest_path.split("/")[:-1]) or "/"
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(data)
            info.mtime = int(time.time())
            info.mode = 0o644
            tar.addfile(info, BytesIO(data))
        buf.seek(0)
        container.put_archive(path=dest_dir, data=buf)


# ────────────────────────────────────────────────
# SandboxManager — auto-selects Docker or Local
# ────────────────────────────────────────────────

class SandboxManager:
    """
    Manages sandboxes for agent code execution.

    Tries Docker first; falls back to local subprocess execution
    if Docker is not available.
    """

    def __init__(self, config: AIvilizationConfig) -> None:
        self.config = config
        self._backend: DockerSandbox | LocalSandbox | None = None
        self._using_local = False

    def _get_backend(self) -> DockerSandbox | LocalSandbox:
        if self._backend is not None:
            return self._backend

        # Try Docker first
        try:
            import docker
            client = docker.from_env()
            client.ping()
            self._backend = DockerSandbox(self.config)
            return self._backend
        except Exception:
            pass

        # Fall back to local subprocess
        sandbox_dir = Path(self.config.data_dir).parent / "sandboxes"
        self._backend = LocalSandbox(
            base_dir=sandbox_dir,
            timeout=self.config.sandbox_timeout_seconds,
        )
        self._using_local = True
        return self._backend

    @property
    def is_local(self) -> bool:
        self._get_backend()
        return self._using_local

    def exec_python(self, agent_id: str, code: str, timeout: int | None = None) -> ExecResult:
        return self._get_backend().exec_python(agent_id, code, timeout)

    def exec_shell(self, agent_id: str, command: str, timeout: int | None = None) -> ExecResult:
        return self._get_backend().exec_shell(agent_id, command, timeout)

    def read_file(self, agent_id: str, path: str) -> str:
        return self._get_backend().read_file(agent_id, path)

    def write_file(self, agent_id: str, path: str, content: str) -> None:
        return self._get_backend().write_file(agent_id, path, content)

    def list_files(self, agent_id: str, path: str = "/workspace") -> list[str]:
        return self._get_backend().list_files(agent_id, path)

    def install_package(self, agent_id: str, package: str) -> ExecResult:
        return self._get_backend().install_package(agent_id, package)

    def snapshot(self, agent_id: str) -> str:
        backend = self._get_backend()
        if isinstance(backend, DockerSandbox):
            return backend.snapshot(agent_id)
        return ""  # Local sandbox doesn't support snapshots

    def restore(self, agent_id: str, image_tag: str) -> None:
        backend = self._get_backend()
        if isinstance(backend, DockerSandbox):
            backend.restore(agent_id, image_tag)

    def destroy(self, agent_id: str) -> None:
        self._get_backend().destroy(agent_id)

    def destroy_all(self) -> None:
        self._get_backend().destroy_all()
