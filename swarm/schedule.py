"""The schedule: the one file that drives every configuration.

A schedule directory holds:

    meta.json        generator arguments, tokenizer, counts
    sessions.jsonl   one line per agent

Each agent line:

    {"agent": "d0-a1", "dev": 0, "instance_id": "...", "start_s": 3.2,
     "messages": [{"role": ..., "content": ...}, ...],
     "turns": [{"msg": 5, "max_tokens": 120, "cleared": [3], "worker": 0,
                "after_s": 2.1, "idle_s": 0.0, "moved": false}, ...]}

Warm-up agents (a developer cohort replayed before measuring, untimed) carry
"warmup": true and come after the measured agents; measured agents omit the key.

Turn k sends messages[:msg] (every message before that assistant turn), with
the messages listed in `cleared` replaced by CLEARED_TEXT. After the reply the
agent waits `after_s` (tool latency) plus `idle_s` (an injected idle gap), then
sends turn k+1 to that turn's worker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

CLEARED_TEXT = "OBSERVATION:\n[Old tool result content cleared]"


@dataclass
class Turn:
    msg: int
    max_tokens: int
    worker: int = 0
    after_s: float = 0.0
    idle_s: float = 0.0
    moved: bool = False
    cleared: list[int] = field(default_factory=list)


@dataclass
class Agent:
    agent: str
    dev: int
    instance_id: str
    start_s: float
    messages: list[dict]
    turns: list[Turn]
    warmup: bool = False  # part of the untimed warm-up cohort, not measured

    def request(self, k: int) -> list[dict]:
        """Messages sent for turn k."""
        t = self.turns[k]
        cleared = set(t.cleared)
        return [
            {"role": m["role"], "content": CLEARED_TEXT if i in cleared else m["content"]}
            for i, m in enumerate(self.messages[: t.msg])
        ]


def save(path: Path, meta: dict, agents: list[Agent]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with open(path / "sessions.jsonl", "w") as f:
        for a in agents:
            d = asdict(a)
            if not d["warmup"]:
                del d["warmup"]  # measured agents: same bytes as before warm-up existed
            f.write(json.dumps(d) + "\n")


def load(path: Path) -> tuple[dict, list[Agent]]:
    meta = json.loads((path / "meta.json").read_text())
    agents = []
    with open(path / "sessions.jsonl") as f:
        for line in f:
            d = json.loads(line)
            d["turns"] = [Turn(**t) for t in d["turns"]]
            agents.append(Agent(**d))
    return meta, agents


def split(agents: list[Agent]) -> tuple[list[Agent], list[Agent]]:
    """(warm-up agents, measured agents)."""
    return [a for a in agents if a.warmup], [a for a in agents if not a.warmup]
