"""SWE-smith trajectories -> normalized agent sessions.

Source: https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories (MIT).
5,016 SWE-agent runs by Claude 3.7 Sonnet, stored in three formats (splits
`tool`, `xml`, `ticks`). We use the `tool` split: native tool calls, with
observations in `tool` messages.

SWE-smith builds one environment per repository and every task in a repo runs
against it, so a file reads the same for every task in that repo, except where
that task's injected bug changed it. That is what lets a group of tasks stand
in for several developers working on one codebase.

Normalization turns every session into plain-text system / user / assistant
messages: tool calls are rendered into the assistant text, tool results become
user messages that start with "OBSERVATION:". Plain text keeps the chat
template identical across engines and makes the replay independent of tool
schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import urllib.request

HF_PARQUET = (
    "https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories/resolve/"
    "refs%2Fconvert%2Fparquet/default/{split}/{shard:04d}.parquet"
)
NUM_SHARDS = 8

_FN = re.compile(r"<function=(\w+)>(.*?)</function>", re.S)
_PAR = re.compile(r"<parameter=(\w+)>(.*?)</parameter>", re.S)
_CAT_N = "Here's the result of running `cat -n` on "


@dataclass
class Session:
    """One recorded agent run, normalized."""

    instance_id: str
    repo: str
    resolved: bool
    messages: list[dict]  # {"role": system|user|assistant, "content": str}
    # (message index, path) of observations that show a whole file (cat -n, no range)
    file_views: list[tuple[int, str]] = field(default_factory=list)

    @property
    def assistant_turns(self) -> list[int]:
        return [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]


def download(dest: Path, split: str = "tool", shards: list[int] | None = None) -> list[Path]:
    """Download parquet shards of the trajectories (about 100-250 MB each)."""
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for s in shards if shards is not None else range(NUM_SHARDS):
        p = dest / f"{split}-{s:04d}.parquet"
        if not p.exists():
            url = HF_PARQUET.format(split=split, shard=s)
            print(f"[corpus] downloading {url}")
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
            tmp.rename(p)
        out.append(p)
    return out


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content or ""


def _render_call(name: str, args: dict) -> str:
    params = "".join(f"<parameter={k}>{v}</parameter>\n" for k, v in args.items())
    return f"<function={name}>\n{params}</function>"


def normalize(raw: list[dict]) -> tuple[list[dict], list[tuple[int, str]]]:
    """Plain-text messages, plus the whole-file views among them."""
    msgs: list[dict] = []
    views: list[tuple[int, str]] = []
    pending_view: str | None = None  # path of a whole-file view awaiting its observation
    for m in raw:
        role = m.get("role")
        text = _text(m.get("content"))
        if role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                f = tc.get("function", {})
                args = f.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                calls.append((f.get("name", ""), args or {}))
            body = text
            if calls:
                body = (text + "\n\n" if text else "") + "\n".join(_render_call(n, a) for n, a in calls)
            else:
                calls = [(n, dict(_PAR.findall(b))) for n, b in _FN.findall(text)]
            pending_view = None
            for name, args in calls[:1]:
                if name == "str_replace_editor" and args.get("command") == "view" and not args.get("view_range"):
                    pending_view = str(args.get("path", ""))
            msgs.append({"role": "assistant", "content": body})
        elif role in ("tool", "user"):
            if role == "tool" and not text.startswith("OBSERVATION:"):
                text = "OBSERVATION:\n" + text
            if pending_view and _CAT_N in text[:400]:
                views.append((len(msgs), pending_view))
            pending_view = None
            msgs.append({"role": "user", "content": text})
        elif role == "system":
            msgs.append({"role": "system", "content": text})
    return msgs, views


def load_sessions(paths: list[Path], repo: str | None = None) -> list[Session]:
    """One session per instance (the first run seen), optionally for one repo."""
    import pyarrow.parquet as pq  # type: ignore

    seen: set[str] = set()
    out: list[Session] = []
    for p in sorted(paths):
        table = pq.read_table(p, columns=["messages", "instance_id", "resolved"])
        for inst, msgs_json, resolved in zip(
            table.column("instance_id").to_pylist(),
            table.column("messages").to_pylist(),
            table.column("resolved").to_pylist(),
        ):
            r = inst.split(".")[0]
            if (repo and r != repo) or inst in seen:
                continue
            seen.add(inst)
            msgs, views = normalize(json.loads(msgs_json))
            if not any(m["role"] == "assistant" for m in msgs) or msgs[0]["role"] != "system":
                continue
            out.append(Session(inst, r, bool(resolved), msgs, views))
    return out


def repo_counts(paths: list[Path]) -> dict[str, int]:
    """Distinct instances per repo, largest first (to choose a repo)."""
    import pyarrow.parquet as pq  # type: ignore

    insts: dict[str, set[str]] = {}
    for p in paths:
        for inst in pq.read_table(p, columns=["instance_id"]).column("instance_id").to_pylist():
            insts.setdefault(inst.split(".")[0], set()).add(inst)
    return dict(sorted(((k, len(v)) for k, v in insts.items()), key=lambda kv: -kv[1]))
