#!/usr/bin/env python3
"""A CPU-only stand-in for the serving stack, for testing the replay driver.

Copied from netapp-cacheblend (tests/mock_engine.py) and extended with the
lookup and blend counters swarm/summarize.py reads.

It speaks just enough of vLLM's OpenAI API (/v1/models, /tokenize, streaming
/v1/chat/completions, /metrics) and LMCache's /metrics for
swarm/replay.py to run end to end without a GPU:

  * a fixed number of "GPU slots" (--slots); requests beyond that queue, so an
    open-loop rate above capacity shows up as rising TTFT
  * prefill time proportional to prompt length (--prefill-tok-per-s)
  * vLLM queue/prefill histograms and prompt counters on /metrics
  * LMCache-style counters on a second port: every request "reads" its chunks
    from L1, and a configurable fraction of them "came from L2" (--l2-fraction)

It checks plumbing, not performance: numbers it produces mean nothing.

    python3 tests/mock_engine.py --port 18000 --lmcache-port 19090 &
    python3 -m swarm.replay <schedule> --engines http://127.0.0.1:18000 \
        --lmcache-metrics http://127.0.0.1:19090/metrics -o runs/mock
"""

# Standard
import argparse
import asyncio
import json
import threading
import time

# Third Party
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
import uvicorn

CHUNK = 256


def build(args) -> tuple[FastAPI, FastAPI]:
    engine, lmc = FastAPI(), FastAPI()
    slots = asyncio.Semaphore(args.slots)
    stats = {"prompt": 0.0, "queue_s": 0.0, "prefill_s": 0.0, "n": 0}
    lmc_stats = {"l1_read": 0.0, "l2_loaded": 0.0, "l1_write": 0.0,
                 "req_tok": 0.0, "hit_tok": 0.0, "prefix_hit": 0.0, "non_prefix_hit": 0.0,
                 "hit_l1": 0.0, "hit_l2": 0.0}
    lock = threading.Lock()

    def n_tokens(messages) -> int:
        # ~1 token per 4 characters is close enough for a mock.
        return max(1, sum(len(m.get("content", "")) for m in messages) // 4)

    @engine.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": args.model, "object": "model", "max_model_len": args.max_model_len}]}

    @engine.post("/tokenize")
    async def tokenize(req: Request):
        body = await req.json()
        if "prompt" in body:  # vLLM's prompt form: one fake id per character
            toks = [ord(c) for c in body["prompt"]]
            return {"count": len(toks), "tokens": toks}
        return {"count": n_tokens(body.get("messages", []))}

    @engine.post("/v1/completions")
    async def completions(req: Request):
        # Non-streaming completions with a token-id prompt (scripts/smoke_cb.py).
        # Every request after the first "blends" 20% of its prompt, so the
        # smoke test's PASS path is exercised.
        body = await req.json()
        p = body["prompt"]
        ntok = len(p) if isinstance(p, list) else max(1, len(p) // 4)
        with lock:
            seen = lmc_stats.setdefault("completions", 0)
            lmc_stats["completions"] = seen + 1
            lmc_stats["req_tok"] += ntok
            if seen and not args.no_blend:
                lmc_stats["non_prefix_hit"] += int(ntok * 0.2)
        return {"id": "cmpl-mock", "object": "text_completion", "model": args.model,
                "choices": [{"index": 0, "text": "x", "finish_reason": "length"}],
                "usage": {"prompt_tokens": ntok, "completion_tokens": 1, "total_tokens": ntok + 1}}

    @engine.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        ntok = n_tokens(body["messages"])
        max_tokens = int(body.get("max_tokens") or 1)
        t_arrive = time.monotonic()

        async def gen():
            async with slots:
                t_start = time.monotonic()
                prefill = ntok / args.prefill_tok_per_s
                await asyncio.sleep(prefill)
                with lock:
                    stats["prompt"] += ntok
                    stats["queue_s"] += t_start - t_arrive
                    stats["prefill_s"] += prefill
                    stats["n"] += 1
                    chunks = ntok // CHUNK
                    lmc_stats["l1_read"] += chunks
                    lmc_stats["l2_loaded"] += round(chunks * args.l2_fraction)
                    lmc_stats["l1_write"] += chunks
                    lmc_stats["req_tok"] += ntok
                    lmc_stats["hit_tok"] += int(ntok * 0.8)
                    # fixed shares, so tests can check the accounting
                    lmc_stats["prefix_hit"] += int(ntok * 0.6)
                    lmc_stats["non_prefix_hit"] += int(ntok * 0.2)
                    lmc_stats["hit_l1"] += int(ntok * 0.4)
                    lmc_stats["hit_l2"] += int(ntok * 0.2)
                rid = f"chatcmpl-mock-{int(t_arrive * 1e6)}"
                for i in range(max_tokens):
                    chunk = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                             "model": args.model,
                             "choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    if i + 1 < max_tokens:
                        await asyncio.sleep(0.002)
                usage = {"id": rid, "object": "chat.completion.chunk", "created": int(time.time()),
                         "model": args.model, "choices": [],
                         "usage": {"prompt_tokens": ntok, "completion_tokens": max_tokens,
                                   "total_tokens": ntok + max_tokens}}
                yield f"data: {json.dumps(usage)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @engine.get("/metrics")
    async def metrics():
        with lock:
            s = dict(stats)
        lines = [
            f'vllm:request_prefill_time_seconds_sum{{engine="0"}} {s["prefill_s"]}',
            f'vllm:request_prefill_time_seconds_count{{engine="0"}} {s["n"]}',
            f'vllm:request_queue_time_seconds_sum{{engine="0"}} {s["queue_s"]}',
            f'vllm:request_queue_time_seconds_count{{engine="0"}} {s["n"]}',
            f'vllm:prompt_tokens_total{{engine="0"}} {s["prompt"]}',
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    @lmc.get("/metrics")
    async def lmc_metrics():
        with lock:
            s = dict(lmc_stats)
        lines = [
            "# HELP lmcache_mp_l1_read_chunks_total Total chunks read from L1",
            f'lmcache_mp_l1_read_chunks_total{{instance="mock"}} {s["l1_read"]}',
            f'lmcache_mp_l1_write_chunks_total{{instance="mock"}} {s["l1_write"]}',
            f'lmcache_mp_l2_prefetch_load_completed_chunks_total{{adapter="fs_native"}} {s["l2_loaded"]}',
            # same stem, different instrument: must NOT be counted as loaded chunks
            f'lmcache_mp_l2_prefetch_load_submitted_objects_total{{adapter="fs_native"}} {s["l2_loaded"] * 7}',
            f'lmcache_blend_lookup_requested_tokens_total {s["req_tok"]}',
            f'lmcache_blend_lookup_hit_tokens_total {s["hit_tok"]}',
            'lmcache_mp_l1_read_chunks_bucket{le="1"} 999',
            f'lmcache_mp_lookup_prompt_tokens_total {s["req_tok"]}',
            f'lmcache_mp_lookup_request_hit_tokens_total {s["hit_tok"]}',
            f'lmcache_mp_lookup_hit_l1_tokens_total {s["hit_l1"]}',
            f'lmcache_mp_lookup_hit_l2_tokens_total {s["hit_l2"]}',
            # v0.5.8rc1 also exports the tier hits in keys (chunks): must not be added to tokens
            f'lmcache_mp_lookup_hit_l1_keys_total {s["hit_l1"] // 256}',
            f'lmcache_mp_lookup_hit_l2_keys_total {s["hit_l2"] // 256}',
            f'lmcache_blend_lookup_prefix_hit_tokens_total {s["prefix_hit"]}',
            f'lmcache_blend_lookup_non_prefix_hit_tokens_total {s["non_prefix_hit"]}',
        ]
        return PlainTextResponse("\n".join(lines) + "\n")

    return engine, lmc


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=18000)
    p.add_argument("--lmcache-port", type=int, default=19090)
    p.add_argument("--model", default="mock-model")
    p.add_argument("--max-model-len", type=int, default=40960)
    p.add_argument("--slots", type=int, default=4, help="concurrent requests served")
    p.add_argument("--prefill-tok-per-s", type=float, default=400000.0)
    p.add_argument("--l2-fraction", type=float, default=0.6)
    p.add_argument("--no-blend", action="store_true", help="completions never report non-prefix hits")
    args = p.parse_args()
    engine, lmc = build(args)

    async def serve():
        s1 = uvicorn.Server(uvicorn.Config(engine, host="127.0.0.1", port=args.port, log_level="warning"))
        s2 = uvicorn.Server(uvicorn.Config(lmc, host="127.0.0.1", port=args.lmcache_port, log_level="warning"))
        await asyncio.gather(s1.serve(), s2.serve())

    asyncio.run(serve())


if __name__ == "__main__":
    main()
