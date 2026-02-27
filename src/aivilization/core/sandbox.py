from __future__ import annotations

import tarfile
import threading
import time
from io import BytesIO
from typing import Any

import docker
from docker.models.containers import Container
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


class SandboxManager:
    """
    Manages Docker containers for agent sandboxes.

    Each agent gets its own persistent Docker container with:
    - Isolated filesystem (/workspace)
    - Resource limits (memory, CPU, PIDs)
    - No network access by default
    - Python 3.12 + common packages pre-installed
    """

    def __init__(self, config: AIvilizationConfig) -> None:
        self.config = config
        self._client: docker.DockerClient | None = None
        self._containers: dict[str, Container] = {}  # agent_id -> Container

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def _ensure_base_image(self) -> None:
        """Build the base sandbox image if it doesn't exist."""
        try:
            self.client.images.get(self.config.sandbox_base_image)
        except docker.errors.ImageNotFound:
            # Build from Dockerfile
            import os

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
                # Fallback: use python:3.12-slim directly
                self.client.images.pull("python:3.12-slim")

    def create_sandbox(self, agent_id: str) -> None:
        """Create and start a Docker container for an agent."""
        self._ensure_base_image()

        container_name = f"aiv-{agent_id[:12]}"

        # Check if container already exists
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
            memswap_limit=f"{self.config.sandbox_memory_mb}m",  # Disable swap
            nano_cpus=nano_cpus,
            pids_limit=self.config.sandbox_pids_limit,
            network_disabled=True,
            working_dir="/workspace",
            security_opt=["no-new-privileges"],
        )
        self._containers[agent_id] = container

    def _get_container(self, agent_id: str) -> Container:
        """Get the container for an agent, creating it if needed."""
        if agent_id not in self._containers:
            self.create_sandbox(agent_id)
        container = self._containers[agent_id]
        container.reload()
        if container.status != "running":
            container.start()
        return container

    def exec_python(self, agent_id: str, code: str, timeout: int | None = None) -> ExecResult:
        """Execute Python code in agent's container."""
        timeout = timeout or self.config.sandbox_timeout_seconds
        container = self._get_container(agent_id)

        start = time.monotonic()
        timed_out_flag: list[bool] = []

        # Write code to a temp file in the container, then execute it
        self._write_to_container(container, "/workspace/_exec.py", code)

        try:
            # Start the execution
            exec_id = self.client.api.exec_create(
                container.id,
                ["python3", "/workspace/_exec.py"],
                workdir="/workspace",
                stdout=True,
                stderr=True,
            )

            # Set up timeout watchdog
            def watchdog():
                time.sleep(timeout)
                try:
                    container.reload()
                    # Kill the specific exec, not the container
                    # We'll use a different approach - exec with timeout
                    timed_out_flag.append(True)
                except Exception:
                    pass

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
                error=str(e),
                exit_code=-1,
                duration_ms=duration,
                timed_out=bool(timed_out_flag),
            )

    def exec_shell(self, agent_id: str, command: str, timeout: int | None = None) -> ExecResult:
        """Execute a shell command in agent's container."""
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
        """Read a file from agent's container."""
        container = self._get_container(agent_id)

        # Ensure path is within /workspace
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
                    raise FileNotFoundError(f"Cannot read {path} (is it a directory?)")
                return f.read().decode("utf-8", errors="replace")
        except docker.errors.NotFound:
            raise FileNotFoundError(f"File not found: {path}")

    def write_file(self, agent_id: str, path: str, content: str) -> None:
        """Write a file to agent's container."""
        container = self._get_container(agent_id)

        if not path.startswith("/"):
            path = f"/workspace/{path}"
        if not path.startswith("/workspace"):
            raise PermissionError("Can only write files within /workspace")

        # Ensure parent directory exists
        parent_dir = "/".join(path.split("/")[:-1])
        container.exec_run(["mkdir", "-p", parent_dir])

        filename = path.split("/")[-1]
        dest_dir = parent_dir

        self._write_to_container(container, path, content)

    def list_files(self, agent_id: str, path: str = "/workspace") -> list[str]:
        """List files in agent's container."""
        container = self._get_container(agent_id)

        if not path.startswith("/workspace"):
            path = f"/workspace/{path}" if path else "/workspace"

        result = container.exec_run(
            ["find", path, "-type", "f", "-not", "-name", "_exec.py"],
            demux=True,
        )
        stdout = (result.output[0] or b"").decode("utf-8", errors="replace")
        files = [f.strip() for f in stdout.strip().split("\n") if f.strip()]
        # Return relative to /workspace
        return [f.replace("/workspace/", "") for f in files if f != "/workspace"]

    def install_package(self, agent_id: str, package: str) -> ExecResult:
        """Install a Python package in agent's container."""
        container = self._get_container(agent_id)

        result = container.exec_run(
            ["pip", "install", "--user", package],
            demux=True,
        )
        stdout_bytes, stderr_bytes = result.output
        return ExecResult(
            stdout=(stdout_bytes or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_bytes or b"").decode("utf-8", errors="replace"),
            exit_code=result.exit_code,
        )

    def snapshot(self, agent_id: str) -> str:
        """Commit container state as image for persistence. Returns image tag."""
        container = self._get_container(agent_id)
        tag = f"aiv-snapshot-{agent_id[:12]}"
        container.commit(repository=tag, tag="latest")
        return f"{tag}:latest"

    def restore(self, agent_id: str, image_tag: str) -> None:
        """Restore container from snapshot image."""
        # Remove current container if exists
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
        """Stop and remove agent's container."""
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
        """Stop and remove all managed containers."""
        for agent_id in list(self._containers.keys()):
            self.destroy(agent_id)

    def _write_to_container(self, container: Container, dest_path: str, content: str) -> None:
        """Write content as a file into a container using tar archive."""
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
