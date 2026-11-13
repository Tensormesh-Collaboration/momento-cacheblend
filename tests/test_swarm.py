"""Offline tests for the swarm generator, simulator and replay driver.

    python -m pytest -q tests/

No GPU, no network: a one-token-per-character tokenizer, a synthetic parquet in
SWE-smith's tool format, and tests/mock_engine.py for the replay.
"""

from __future__ import annotations

from pathlib import Path
import json
import random
import socket
import subprocess
import sys
import time

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from swarm import generate, schedule, simulate, summarize  # noqa: E402
from swarm.tokens import CHUNK, Tokenizer, aligned_chunk_hashes, window_hashes  # noqa: E402


# --- hashing -------------------------------------------------------------------

def test_window_hashes_match_naive():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 150_000, size=CHUNK * 3 + 17, dtype=np.int64)
    wh = window_hashes(t)
    assert len(wh) == len(t) - CHUNK + 1
    B, M = 0x9E3779B97F4A7C15, 1 << 64
    for i in (0, 1, 100, len(wh) - 1):
        h = 0
        for x in t[i:i + CHUNK]:
            h = (h * B + int(x) + 1) % M
        assert int(wh[i]) == h
    # equal windows hash equally wherever they sit
    u = np.concatenate([rng.integers(0, 9, 37), t[:CHUNK], rng.integers(0, 9, 5)])
    assert int(window_hashes(u)[37]) == int(wh[0])
    assert len(aligned_chunk_hashes(t)) == 3


# --- synthetic SWE-smith data -------------------------------------------------------

FILE_A = "".join(f"{i:5d}\tdef helper_{i}(x):\n" for i in range(60))  # ~1.4K chars
FILE_B = "".join(f"{i:5d}\tclass Model{i}:\n" for i in range(60))


def _session(i: int, repo: str, files: list[tuple[str, str]], turns: int) -> dict:
    msgs = [{"role": "system", "content": "You are a helpful assistant. " * 20},
            {"role": "user", "content": f"<pr_description>Fix bug {i} in {repo}</pr_description>" + "x" * (50 + 7 * i)}]
    for k in range(turns):
        path, body = files[k % len(files)]
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": f"Let me look at {path} ({k})."}],
                     "tool_calls": [{"function": {"name": "str_replace_editor",
                                                  "arguments": json.dumps({"command": "view", "path": path})}}]})
        msgs.append({"role": "tool", "content": f"Here's the result of running `cat -n` on {path}:\n{body}"})
    msgs.append({"role": "assistant", "content": "Done.",
                 "tool_calls": [{"function": {"name": "submit", "arguments": "{}"}}]})
    return {"messages": json.dumps(msgs), "instance_id": f"{repo}.abc123.bug_{i}", "resolved": True}


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path_factory.mktemp("swesmith")
    files = [("/testbed/pkg/a.py", FILE_A), ("/testbed/pkg/b.py", FILE_B)]
    rows = [_session(i, "org__repo", files[i % 2:] + files[: i % 2], 4) for i in range(12)]
    rows += [_session(100 + i, "other__repo", files, 2) for i in range(3)]
    table = pa.table({k: [r[k] for r in rows] for k in ("messages", "instance_id", "resolved")})
    pq.write_table(table, d / "tool-0000.parquet")
    return d


def _gen(data_dir: Path, out: Path, *extra: str) -> tuple[dict, list[schedule.Agent]]:
    args = generate.parser().parse_args([
        "--data", str(data_dir), "--repo", "org__repo", "--tokenizer", "chars",
        "--devs", "2", "--agents", "3", "--harness-tokens", "600", "--dev-preamble-tokens", "100",
        "--workers", "2", "-o", str(out), *extra])
    generate.build(args)
    return schedule.load(out)


def test_generate_structure(data_dir, tmp_path):
    meta, agents = _gen(data_dir, tmp_path / "s")
    assert meta["agents"] == 6 and {a.dev for a in agents} == {0, 1}
    # one shared harness prompt, one preamble per developer
    assert len({a.messages[0]["content"] for a in agents}) == 1
    assert len(agents[0].messages[0]["content"]) >= 600
    pre = {a.dev: a.messages[1]["content"].split("<pr_description>")[0] for a in agents}
    assert pre[0] != pre[1]
    # turn k sends everything before assistant message k
    a = agents[0]
    for k, t in enumerate(a.turns):
        req = a.request(k)
        assert a.messages[t.msg]["role"] == "assistant" and len(req) == t.msg
    # deterministic for a seed
    _, again = _gen(data_dir, tmp_path / "s2")
    assert [x.messages for x in again] == [x.messages for x in agents]
    assert [[t.after_s for t in x.turns] for x in again] == [[t.after_s for t in x.turns] for x in agents]


def test_generate_subagents_bundle_and_clearing(data_dir, tmp_path):
    _, sub = _gen(data_dir, tmp_path / "sub", "--subagent-turns", "2", "--bundle-files", "1")
    assert all(len(a.turns) == 2 for a in sub)
    assert all("Context from the lead agent" in a.messages[1]["content"] for a in sub)
    _, clr = _gen(data_dir, tmp_path / "clr", "--clear-trigger", "2500", "--clear-keep", "1")
    assert any(t.cleared for a in clr for t in a.turns)
    a = next(a for a in clr if a.turns[-1].cleared)
    assert schedule.CLEARED_TEXT in [m["content"] for m in a.request(len(a.turns) - 1)]


def test_generate_idle_moves(data_dir, tmp_path):
    _, ags = _gen(data_dir, tmp_path / "mv", "--idle-prob", "1", "--move-prob", "1")
    moved = [t for a in ags for t in a.turns if t.moved]
    assert moved and all(a.turns[k - 1].idle_s for a in ags for k, t in enumerate(a.turns) if t.moved)


# --- simulator ---------------------------------------------------------------------

def _agent(name: str, dev: int, msgs: list[str], worker: int = 0, start: float = 0.0) -> schedule.Agent:
    full = [{"role": "system", "content": "S" * 300}]
    turns = []
    for m in msgs:
        full.append({"role": "user", "content": m})
        turns.append(schedule.Turn(msg=len(full), max_tokens=4, worker=worker))
        full.append({"role": "assistant", "content": "ok"})
    return schedule.Agent(agent=name, dev=dev, instance_id=name, start_s=start, messages=full, turns=turns)


def test_simulator_prefix_and_blend():
    rng = random.Random(1)
    shared = "".join(rng.choice("abcdefghij") for _ in range(4 * CHUNK))
    a1 = _agent("x", 0, ["task one " * 30 + shared, "next"], start=0.0)
    a2 = _agent("y", 1, ["task two is different " * 13 + shared, "next"], start=100.0)
    p = simulate.Prompts(Tokenizer("chars"), [a1, a2])
    A = simulate.run(p, "A", 1, 0, 0, False, 1e9, 1e9)["totals"]
    B = simulate.run(p, "B", 1, 0, 0, False, 1e9, 1e9)["totals"]
    assert A.get("blend_l1", 0) == 0
    assert A["prefix_l1"] > 0            # the shared system prompt, and each agent's own second turn
    assert B["blend_l1"] >= 2 * CHUNK    # the shared file, at another offset, from the other agent
    assert B.get("blend_from_other_dev", 0) == B["blend_l1"]
    assert B["new"] < A["new"]


def test_simulator_valkey_recovers_moves():
    big = "".join(random.Random(2).choice("klmnopqrst") for _ in range(8 * CHUNK))
    a = _agent("x", 0, [big, "more"], worker=0)
    a.turns[1].worker, a.turns[1].moved = 1, True
    p = simulate.Prompts(Tokenizer("chars"), [a])
    A = simulate.run(p, "A", 2, 0, 0, False, 1e9, 1e9)["totals"]
    C = simulate.run(p, "C", 2, 0, 0, False, 1e9, 1e9)["totals"]
    assert A.get("prefix_l2", 0) == 0 and C["prefix_l2"] >= 8 * CHUNK


def test_simulator_lru_eviction():
    rng = random.Random(3)
    docs = ["".join(rng.choice("uvwxyz") for _ in range(4 * CHUNK)) for _ in range(3)]
    ags = [_agent(f"a{i}", 0, [docs[i], "again"], start=i) for i in range(3)]
    p = simulate.Prompts(Tokenizer("chars"), ags)
    small = simulate.run(p, "A", 1, 4, 0, False, 1e9, 1e9)
    big = simulate.run(p, "A", 1, 0, 0, False, 1e9, 1e9)
    assert small["l1_evicted_chunks"] > 0
    assert small["totals"]["prefix_l1"] < big["totals"]["prefix_l1"]


# --- replay against the mock engine -------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def mock_engines():
    pytest.importorskip("fastapi")
    pytest.importorskip("openai")
    procs, ports = [], []
    for _ in range(2):
        e, m = _free_port(), _free_port()
        procs.append(subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_engine.py"), "--port", str(e),
                                       "--lmcache-port", str(m), "--prefill-tok-per-s", "5e6"],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        ports.append((e, m))
    import urllib.request
    for e, _ in ports:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{e}/v1/models", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
    yield ports
    for p in procs:
        p.terminate()


def test_replay_and_summarize(data_dir, tmp_path, mock_engines):
    _gen(data_dir, tmp_path / "s", "--idle-prob", "0.5", "--move-prob", "1")
    out = tmp_path / "run"
    engines = ",".join(f"http://127.0.0.1:{e}" for e, _ in mock_engines)
    lmc = ",".join(f"http://127.0.0.1:{m}/metrics" for _, m in mock_engines)
    subprocess.run([sys.executable, "-m", "swarm.replay", str(tmp_path / "s"), "--engines", engines,
                    "--lmcache-metrics", lmc, "--time-scale", "0.001", "--idle-scale", "0", "-o", str(out)],
                   cwd=ROOT, check=True, timeout=120)
    reqs = [json.loads(x) for x in open(out / "requests.jsonl")]
    _, agents = schedule.load(tmp_path / "s")
    assert len(reqs) == sum(len(a.turns) for a in agents)
    assert all(r["ok"] for r in reqs)
    assert {r["worker"] for r in reqs} == {0, 1}
    row = summarize.summarize(out, ttft_target=5.0, recompute_ratio=0.15)
    assert row["failed"] == 0 and row["lmcache_reachable"]
    # the mock reports 60% prefix + 20% non-prefix hits of its own token estimate
    assert row["non_prefix_pct"] is not None and row["reused_pct"] > row["non_prefix_pct"] > 0
    assert row["l2_share_of_prefix_hits_pct"] == pytest.approx(33.3, abs=1.0)


# --- CacheBlend smoke test ---------------------------------------------------------

@pytest.mark.parametrize("blend,expect", [(True, 0), (False, 1)])
def test_smoke_cb(blend, expect):
    pytest.importorskip("fastapi")
    e, m = _free_port(), _free_port()
    cmd = [sys.executable, str(ROOT / "tests" / "mock_engine.py"), "--port", str(e), "--lmcache-port", str(m)]
    proc = subprocess.Popen(cmd + ([] if blend else ["--no-blend"]), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{e}/v1/models", timeout=1)
                break
            except OSError:
                time.sleep(0.1)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "smoke_cb.py"), "--port", str(e),
                            "--metrics", f"http://127.0.0.1:{m}/metrics", "--settle", "0"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == expect, r.stdout + r.stderr
        assert ("PASS" if blend else "FAIL") in r.stdout
    finally:
        proc.terminate()
