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
    small = simulate.run(p, "A", 1, 4, 0, False, 1e9, 1e9, watermark=None)  # ideal LRU
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


def test_simulator_smaller_chunks_recover_span_edges():
    # A shared span at an offset that is not a multiple of any chunk size: the
    # stored copy loses its partial edge chunks, and finer chunks lose less.
    rng = random.Random(4)
    shared = "".join(rng.choice("abcdefghij") for _ in range(1000))
    a1 = _agent("x", 0, ["p" * 37 + shared], start=0.0)
    a2 = _agent("y", 1, ["different preamble " * 9 + shared], start=10.0)
    p = simulate.Prompts(Tokenizer("chars"), [a1, a2])
    got = {c: simulate.run(p, "B", 1, 0, 0, False, 1e9, 1e9, chunk=c)["totals"].get("blend_l1", 0) for c in (256, 128, 64)}
    assert 0 < got[256] <= got[128] <= got[64] <= 1000
    assert got[64] > got[256]


# --- worker-move sweep, idle-gap pressure ------------------------------------------

def test_move_prob_sweep_changes_only_moves(data_dir, tmp_path, monkeypatch):
    out = tmp_path / "mv"
    monkeypatch.setattr(sys, "argv", ["generate", "--data", str(data_dir), "--repo", "org__repo", "--tokenizer", "chars",
                                      "--devs", "2", "--agents", "3", "--harness-tokens", "600",
                                      "--dev-preamble-tokens", "100", "--workers", "2", "--idle-prob", "0.6",
                                      "--move-prob", "0,0.5,1", "-o", str(out)])
    generate.main()
    loaded = {q: schedule.load(Path(f"{out}-q{q}")) for q in ("0", "0.5", "1")}
    sig = {q: [[(t.msg, t.after_s, t.idle_s, t.max_tokens) for t in a.turns] for a in ags]
           for q, (_, ags) in loaded.items()}
    assert sig["0"] == sig["0.5"] == sig["1"]  # same requests, gaps and idle periods
    moves = {q: {(a.agent, k) for a in ags for k, t in enumerate(a.turns) if t.moved} for q, (_, ags) in loaded.items()}
    assert not moves["0"] and moves["0.5"] <= moves["1"] and moves["1"]
    assert loaded["0.5"][0]["move_prob"] == 0.5


def test_simulator_idle_gap_pressure_and_l2_peak():
    rng = random.Random(5)
    docs = ["".join(rng.choice("klmnop") for _ in range(6 * CHUNK)) for _ in range(4)]
    paused = _agent("paused", 0, [docs[0], "back"], start=0.0)
    paused.turns[0].idle_s = 100.0
    busy = [_agent(f"b{i}", 1, [docs[i + 1], "more"], start=10.0 + i) for i in range(3)]
    p = simulate.Prompts(Tokenizer("chars"), [paused] + busy)
    small = simulate.run(p, "A", 1, 8, 0, False, 1e9, 1e9, watermark=None)   # ideal LRU of 8 chunks: the busy agents evict it
    big = simulate.run(p, "C", 1, 0, 0, False, 1e9, 1e9)
    assert set(small["idle_gap_pressure"]) == {">=1"}
    g = small["idle_gaps"][0]
    assert g["stored_tokens"] >= 3 * 6 * CHUNK and g["served"] < g["tokens"]
    assert big["l2_peak_chunks"] > 0 and big["idle_gap_pressure"]["<0.25"]["requests"] == 1


def test_replay_timeseries_feeds_gap_pressure(data_dir, tmp_path, mock_engines):
    _gen(data_dir, tmp_path / "s", "--idle-prob", "0.7", "--move-prob", "1")
    out = tmp_path / "run"
    engines = ",".join(f"http://127.0.0.1:{e}" for e, _ in mock_engines)
    lmc = ",".join(f"http://127.0.0.1:{m}/metrics" for _, m in mock_engines)
    subprocess.run([sys.executable, "-m", "swarm.replay", str(tmp_path / "s"), "--engines", engines,
                    "--lmcache-metrics", lmc, "--time-scale", "0.01", "--idle-scale", "0.002",
                    "--metrics-interval", "0.05", "-o", str(out)], cwd=ROOT, check=True, timeout=120)
    assert (out / "metrics_timeseries.jsonl").stat().st_size > 0
    (out / "config.env").write_text("L1_SIZE_GB=1\nCHUNK_SIZE=256\nCACHE_MODE=blend\nL2_BACKEND=valkey\n")
    (out / "valkey_before.json").write_text('{"evicted_keys": 5}')
    (out / "valkey_after.json").write_text('{"evicted_keys": 5}')
    row = summarize.summarize(out, ttft_target=5.0, recompute_ratio=0.15)
    assert row["l1_gb"] == 1.0 and row["chunk"] == 256 and row["move_prob"] == 1.0
    assert row["valkey_evicted_keys"] == 0 and row["leg"] == "blend+valkey"
    assert row["idle_gaps"] and all(g["stored_tokens"] >= 0 for g in row["idle_gaps"])
    assert sum(v["requests"] for v in row["idle_gap_pressure"].values()) == len(row["idle_gaps"])


def test_metrics_counter_does_not_mix_units():
    """Series names as v0.5.8rc1 exports them (scraped from a live worker on dev-2)."""
    from swarm import metrics
    d = {"lmcache_mp_lookup_hit_l1_tokens_total": 18432.0, "lmcache_mp_lookup_hit_l1_keys_total": 72.0,
         "lmcache_mp_l1_write_chunks_total": 72.0, "lmcache_mp_lookup_request_hit_tokens_total": 18640.0}
    m = metrics.named(d, metrics.LMCACHE)
    assert m["hit_l1"] == 18432.0
    assert m["l1_write"] == 72.0
    assert m["request_hit"] == 18640.0
    assert m["hit_l2"] is None


def test_simulator_lmcache_l1_all_or_nothing_and_watermark():
    """L1 as LMCache runs it: a prompt that does not fit in free space stores nothing;
    eviction happens only above the watermark, once per simulated second."""
    rng = random.Random(9)
    docs = ["".join(rng.choice("qrstuv") for _ in range(6 * CHUNK)) for _ in range(4)]
    # ~7-chunk prompts. 20-chunk L1 evicting at 50%: 10 chunks stay free, enough for one prompt
    ags = [_agent(f"a{i}", 0, [docs[i], "again"], start=10.0 * i) for i in range(4)]
    for ag in ags:
        ag.turns[0].after_s = 2.0
    p = simulate.Prompts(Tokenizer("chars"), ags)
    ok = simulate.run(p, "A", 1, 20, 0, False, 1e9, 1e9, watermark=0.5, ratio=0.5)
    assert ok["failed_stores"] == 0 and ok["l1_evicted_chunks"] > 0
    assert ok["totals"]["prefix_l1"] > 0           # each agent's repeat 2 s later hits
    # 10-chunk L1 at a 0.9 watermark: after one 7-chunk prompt L1 never reaches 90%,
    # so it never evicts, and the next prompt does not fit in the 3 chunks left
    stuck = simulate.run(p, "A", 1, 10, 0, False, 1e9, 1e9, watermark=0.9, ratio=0.5)
    assert stuck["failed_stores"] >= 3 and stuck["l1_evicted_chunks"] == 0
    l1 = simulate.LMCacheL1(10, 0.5, 0.5)
    for k in range(8):
        l1.touch(k)
    l1.tick(0.0)
    l1.tick(0.5)                                     # same second: no eviction round yet
    assert len(l1.d) == 8
    l1.tick(1.0)                                     # one round: evict half of the 8
    assert len(l1.d) == 4 and 0 not in l1 and 7 in l1
