"""Lifecycle manager for local inference engine servers (vLLM / Ollama).

Live servers are tracked in a registry so :func:`kill_all` can tear them down
from the SIGTERM handler and before ``os._exit``, when ``post_run()`` is skipped.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from utils.schemas.server import ServerModel

_logger = logging.getLogger("benchmark")

# Registry of live servers so teardown is guaranteed even when post_run() is
# skipped. Guarded by a lock since stages prepare()/post_run() on different
# threads and kill_all() may fire from a signal handler.
_REGISTRY: list["ServerManager"] = []
_REGISTRY_LOCK = threading.Lock()
_ATEXIT_REGISTERED = False


def _pick_free_port(host: str) -> int:
    """Bind a transient socket to find an unused port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def kill_all() -> None:
    """Terminate every live server. Idempotent; safe to call repeatedly."""
    with _REGISTRY_LOCK:
        managers = list(_REGISTRY)
    for manager in managers:
        try:
            manager.stop()
        except Exception:  # pylint: disable=broad-except
            _logger.exception("Error stopping inference server")


class _EngineAdapter:
    """Per-engine knowledge: launch command, health endpoint, litellm wiring."""

    def build_command(self, cfg: ServerModel, port: int) -> list[str]:
        raise NotImplementedError

    def build_env(self, cfg: ServerModel, port: int) -> dict[str, str]:
        return {}

    def health_url(self, base_url: str) -> str:
        raise NotImplementedError

    def base_url(self, cfg: ServerModel, port: int) -> str:
        return f"http://{cfg.host}:{port}"

    def litellm_model(self, cfg: ServerModel) -> str:
        raise NotImplementedError

    def litellm_api_base(self, base_url: str) -> str:
        return base_url

    def post_start(self, cfg: ServerModel, port: int) -> None:
        """Hook run after the server is healthy (e.g. ensure model present)."""


class _VllmAdapter(_EngineAdapter):
    def build_command(self, cfg, port):
        cmd = [cfg.executable or os.environ.get("VLLM_EXECUTABLE", "vllm"), "serve", cfg.model,
               "--host", cfg.host, "--port", str(port)]
        if cfg.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(cfg.gpu_memory_utilization)]
        if cfg.max_model_len is not None:
            cmd += ["--max-model-len", str(cfg.max_model_len)]
        if cfg.dtype is not None:
            cmd += ["--dtype", cfg.dtype]
        cmd += cfg.launch_args
        return cmd

    def health_url(self, base_url):
        return f"{base_url}/v1/models"

    def litellm_model(self, cfg):
        # vLLM's OpenAI server serves the model under its --served-model-name,
        # which defaults to the model path passed on the command line.
        return f"openai/{cfg.model}"

    def litellm_api_base(self, base_url):
        return f"{base_url}/v1"


class _OllamaAdapter(_EngineAdapter):
    def build_command(self, cfg, port):
        # `ollama serve` reads host/port from OLLAMA_HOST (set in build_env).
        return [cfg.executable or "ollama", "serve"] + cfg.launch_args

    def build_env(self, cfg, port):
        return {"OLLAMA_HOST": f"{cfg.host}:{port}"}

    def health_url(self, base_url):
        return f"{base_url}/api/tags"

    def litellm_model(self, cfg):
        return f"ollama/{cfg.model}"

    def post_start(self, cfg, port):
        # Ollama does not auto-pull on request, so make sure the model is local.
        env = dict(os.environ)
        env["OLLAMA_HOST"] = f"{cfg.host}:{port}"
        _logger.info("Ensuring Ollama model is present: %s", cfg.model)
        subprocess.run([cfg.executable or "ollama", "pull", cfg.model], env=env, check=True)


_ADAPTERS: dict[str, type[_EngineAdapter]] = {
    "vllm": _VllmAdapter,
    "ollama": _OllamaAdapter,
}


class ServerManager:
    """Launches, health-checks and tears down a single inference engine server."""

    def __init__(self, cfg: ServerModel):
        self._cfg = cfg
        self._adapter = _ADAPTERS[cfg.engine]()
        self._proc: subprocess.Popen | None = None
        self._stopped = False
        self._lock = threading.Lock()

        if cfg.api_base:
            # Attaching to an externally-running server: take its URL verbatim.
            self._port = cfg.port
            self.base_url = cfg.api_base.rstrip("/")
        else:
            self._port = _pick_free_port(cfg.host) if cfg.port == 0 else cfg.port
            self.base_url = self._adapter.base_url(cfg, self._port)

        # litellm wiring consumed by the inference stage.
        self.litellm_model = self._adapter.litellm_model(cfg)
        self.litellm_api_base = self._adapter.litellm_api_base(self.base_url)

    def start(self) -> None:
        """Launch the server (unless attaching) and block until it is healthy."""
        global _ATEXIT_REGISTERED  # pylint: disable=global-statement
        with _REGISTRY_LOCK:
            _REGISTRY.append(self)
            if not _ATEXIT_REGISTERED:
                atexit.register(kill_all)
                _ATEXIT_REGISTERED = True

        if self._cfg.api_base:
            _logger.info(
                "Attaching to existing %s server at %s",
                self._cfg.engine,
                self.base_url,
            )
        else:
            env = dict(os.environ)
            env.update(self._adapter.build_env(self._cfg, self._port))
            env.update(self._cfg.env)
            cmd = self._adapter.build_command(self._cfg, self._port)
            # An engine installed in another environment finds that environment's
            # tools (e.g. the ninja vLLM uses for kernel builds) first.
            if os.path.isabs(cmd[0]):
                env["PATH"] = os.path.dirname(cmd[0]) + os.pathsep + env.get("PATH", "")
            _logger.info("Launching inference server: %s", " ".join(cmd))
            # start_new_session=True puts the engine in its own process group so
            # killpg() reaps the worker processes vLLM forks, not just the parent.
            self._proc = subprocess.Popen(  # pylint: disable=consider-using-with
                cmd, env=env, start_new_session=True
            )

        self._wait_until_ready()
        self._adapter.post_start(self._cfg, self._port)

    def get_handle(self) -> tuple[str, str]:
        """Return (litellm_model, litellm_api_base) for dependent stages."""
        return self.litellm_model, self.litellm_api_base

    def _wait_until_ready(self) -> None:
        health_url = self._adapter.health_url(self.base_url)
        deadline = time.monotonic() + self._cfg.startup_timeout_s
        while time.monotonic() < deadline:
            # Fail fast if a launched server exited during startup, surfacing
            # its return code instead of hanging until the timeout.
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"{self._cfg.engine} server exited with code "
                    f"{self._proc.returncode} during startup"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=5) as resp:
                    if resp.status == 200:
                        _logger.info(
                            "%s server ready at %s", self._cfg.engine, self.base_url
                        )
                        return
            except (urllib.error.URLError, OSError):
                pass  # not up yet
            time.sleep(2.0)

        self.stop()
        raise TimeoutError(
            f"{self._cfg.engine} server did not become ready within "
            f"{self._cfg.startup_timeout_s}s ({health_url})"
        )

    def stop(self) -> None:
        """Terminate the server process group. Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            proc = self._proc

        if proc is None or proc.poll() is not None:
            return

        _logger.info("Stopping %s server (pid %s)", self._cfg.engine, proc.pid)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        # SIGKILL only as a last resort: force-killing the engine during an active
        # GPU profiling session can leave the profiler locked.
        try:
            proc.wait(timeout=float(os.environ.get("SERVER_STOP_TIMEOUT_S", "120")))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
