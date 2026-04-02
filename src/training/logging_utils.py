from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
import sys


class TeeStream:
    def __init__(self, *streams: object) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def tee_output(log_path: Path, enabled: bool):
    if not enabled:
        yield
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        stdout_tee = TeeStream(sys.stdout, handle)
        stderr_tee = TeeStream(sys.stderr, handle)
        with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
            yield
