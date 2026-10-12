"""Observability: per-unit atomic history JSON, log lines with progress/ETA, deadline helper, process stats."""
import json, os, resource, time
from .common import atomic_json


class History:
    def __init__(self, path, meta):
        self.path, self.rows, self.meta, self.t0 = path, [], dict(meta), time.time()
        self.flush()

    def add(self, row):
        row = dict(row); row["t"] = round(time.time() - self.t0, 2)
        self.rows.append(row); self.flush()

    def flush(self):
        atomic_json({"meta": self.meta, "rows": self.rows, "updated": time.time()}, self.path)


def log(logf, msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if logf:
        with open(logf, "a") as f: f.write(line + "\n")


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Deadline:
    def __init__(self, seconds):
        self.t_end = time.time() + seconds
    def left(self): return self.t_end - time.time()
    def hit(self): return time.time() >= self.t_end
