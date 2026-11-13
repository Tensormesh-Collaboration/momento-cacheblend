"""Replay a schedule against live OpenAI-compatible engines (one per worker).

    python -m swarm.replay schedules/moto-4x5 \
        --engines http://w0:8000,http://w1:8000 \
        --lmcache-metrics http://w0:8080/metrics,http://w1:8080/metrics \
        -o runs/D-moto-4x5

Closed loop per agent: send turn k to that turn's worker, stream the reply
(max_tokens from the schedule, ignore_eos so decode load is the same in every
configuration), wait the turn's gap and idle time, send turn k+1. Agents run
concurrently. Outputs are discarded: the next turn sends the recorded history,
so every configuration sees identical prompts.

Writes to the output directory:

    requests.jsonl   one line per request (TTFT, latency, tokens, worker, ...)
    metrics.json     LMCache and vLLM counter deltas per worker over the run
    run.json         arguments, schedule meta, start and end time
"""

from __future__ import annotations

from pathlib import Path
import argparse
import asyncio
import json
import time

from swarm import metrics, schedule


async def send(client, model: str, messages: list[dict], max_tokens: int, timeout: float) -> dict:
    t0 = time.perf_counter()
    ttft = None
    usage = None
    n_chunks = 0
    try:
        stream = await asyncio.wait_for(client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=0.0, stream=True,
            stream_options={"include_usage": True}, extra_body={"ignore_eos": True},
        ), timeout)
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                text = getattr(delta, "content", None) or getattr(delta, "reasoning_content", None)
                if text and ttft is None:
                    ttft = time.perf_counter() - t0
                if text:
                    n_chunks += 1
            if getattr(chunk, "usage", None):
                usage = chunk.usage
        ok, err = True, ""
    except Exception as exc:  # noqa: BLE001 - a failed request is a data point, not a crash
        ok, err = False, f"{type(exc).__name__}: {exc}"[:300]
    latency = time.perf_counter() - t0
    out = {"ok": ok and ttft is not None, "error": err, "ttft_s": ttft, "latency_s": round(latency, 4)}
    if usage is not None:
        out["prompt_tokens"] = usage.prompt_tokens
        out["completion_tokens"] = usage.completion_tokens
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None and getattr(details, "cached_tokens", None) is not None:
            out["cached_tokens"] = details.cached_tokens
    if ttft is not None:
        out["ttft_s"] = round(ttft, 4)
    return out


async def agent_loop(ag: schedule.Agent, clients: list, model: str, a, t_start: float, sink) -> None:
    await asyncio.sleep(ag.start_s * a.time_scale)
    for k, turn in enumerate(ag.turns):
        w = turn.worker % len(clients)
        t_submit = time.perf_counter() - t_start
        r = await send(clients[w], model, ag.request(k), turn.max_tokens, a.timeout)
        r.update(agent=ag.agent, dev=ag.dev, turn=k, worker=w, moved=turn.moved,
                 after_idle=bool(k and ag.turns[k - 1].idle_s), t_submit_s=round(t_submit, 3),
                 max_tokens=turn.max_tokens)
        sink(r)
        if k + 1 < len(ag.turns):
            await asyncio.sleep(min(a.gap_cap, turn.after_s) * a.time_scale + turn.idle_s * a.idle_scale)


def scrape_all(engines: list[str], lmc: list[str]) -> list[dict]:
    out = []
    for i, e in enumerate(engines):
        out.append({
            "vllm": metrics.scrape(e.rstrip("/").removesuffix("/v1") + "/metrics", "vllm:"),
            "lmcache": metrics.scrape(lmc[i], "lmcache") if i < len(lmc) else None,
        })
    return out


async def main_async(a) -> None:
    from openai import AsyncOpenAI  # type: ignore

    meta, agents = schedule.load(a.schedule)
    if a.limit_agents:
        agents = agents[: a.limit_agents]
    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    lmc = [u.strip() for u in a.lmcache_metrics.split(",") if u.strip()] if a.lmcache_metrics else []
    bases = [e.rstrip("/") + ("" if e.rstrip("/").endswith("/v1") else "/v1") for e in engines]
    clients = [AsyncOpenAI(base_url=b, api_key="none", timeout=a.timeout, max_retries=0) for b in bases]
    model = a.model
    if not model:
        model = (await clients[0].models.list()).data[0].id
    need = max(t.worker for ag in agents for t in ag.turns) + 1
    if need > len(engines):
        print(f"[replay] schedule uses {need} workers, {len(engines)} engines given: workers wrap modulo")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    before = await asyncio.to_thread(scrape_all, engines, lmc)
    n_done = 0
    n_total = sum(len(ag.turns) for ag in agents)
    f = open(out / "requests.jsonl", "w")

    def sink(r: dict) -> None:
        nonlocal n_done
        n_done += 1
        f.write(json.dumps(r) + "\n")
        f.flush()
        if not r["ok"]:
            print(f"[replay] {r['agent']} turn {r['turn']} failed: {r['error']}")
        elif n_done % max(1, n_total // 20) == 0:
            print(f"[replay] {n_done}/{n_total} requests")

    t_start = time.perf_counter()
    wall_start = time.time()
    await asyncio.gather(*(agent_loop(ag, clients, model, a, t_start, sink) for ag in agents))
    f.close()
    after = await asyncio.to_thread(scrape_all, engines, lmc)
    per_worker = []
    for i in range(len(engines)):
        dl = metrics.delta(before[i]["lmcache"], after[i]["lmcache"])
        dv = metrics.delta(before[i]["vllm"], after[i]["vllm"])
        per_worker.append({"engine": engines[i], "lmcache_reachable": after[i]["lmcache"] is not None,
                           "lmcache": metrics.named(dl, metrics.LMCACHE), "vllm": metrics.named(dv, metrics.VLLM),
                           "lmcache_raw_deltas": {k: v for k, v in sorted(dl.items()) if v}})
    (out / "metrics.json").write_text(json.dumps({"workers": per_worker}, indent=2))
    (out / "run.json").write_text(json.dumps({
        "schedule": str(a.schedule), "schedule_meta": meta, "model": model, "engines": engines,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "started": wall_start, "ended": time.time(), "requests": n_done,
    }, indent=2))
    for c in clients:
        await c.close()
    print(f"[replay] {n_done} requests in {time.time() - wall_start:.0f}s -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule", type=Path)
    ap.add_argument("--engines", required=True, help="comma-separated engine URLs, index = worker")
    ap.add_argument("--lmcache-metrics", default="", help="comma-separated LMCache /metrics URLs, same order")
    ap.add_argument("--model", default="", help="default: the first engine's /v1/models")
    ap.add_argument("--time-scale", type=float, default=1.0, help="multiply tool gaps and start offsets")
    ap.add_argument("--idle-scale", type=float, default=1.0, help="multiply idle gaps (0 removes them)")
    ap.add_argument("--gap-cap", type=float, default=120.0, help="cap on one tool gap, s (before scaling)")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--limit-agents", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
