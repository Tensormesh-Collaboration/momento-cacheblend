"""Build a swarm schedule from SWE-smith trajectories.

    python -m swarm.generate --data data/swesmith --repo getmoto__moto \
        --devs 4 --agents 5 -o schedules/moto-4x5

Each agent replays one recorded SWE-smith session (teacher-forced: turn k
sends the recorded conversation up to assistant turn k). Everything around the
sessions is synthetic and seeded: which developer owns which agent, the shared
harness prompt, each developer's preamble, sub-agent lifetimes and context
bundles, context clearing, gaps, idle periods and worker placement.

Prompt order, as real harnesses build it:

    [shared harness prompt][developer preamble + agent task][history ...]
     system message          first user message
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import argparse
import random
import statistics
import time

from swarm import corpus, schedule, timing
from swarm.tokens import Tokenizer

# A fixed, ordinary vocabulary for synthetic filler text (harness conventions,
# developer notes). Only its length matters to the cache; it reads as prose so
# that tokenizers treat it like prose.
_WORDS = (
    "the of and to in is for that on with as by this be are from at or an it not which have "
    "use when file test code run build module function class method value return error check "
    "update change review branch commit merge release config setting option default path "
    "project team service request response cache server client data type field record list "
    "should must always never only before after during while until unless keep make sure "
    "small large first last next each every other same different new old local remote shared "
    "agent tool task step plan note style lint format format docs comment issue bug fix patch"
).split()


def filler(n_tokens: int, tok: Tokenizer, seed: int, title: str) -> str:
    """Deterministic prose-like text of about n_tokens tokens."""
    if n_tokens <= 0:
        return ""
    rng = random.Random(seed)
    words: list[str] = [title, "\n"]
    text = ""
    while True:
        line = " ".join(rng.choice(_WORDS) for _ in range(rng.randint(8, 18))).capitalize() + ".\n"
        words.append(line)
        if len(words) % 16 == 0:
            text = "".join(words)
            if len(tok.encode(text)) >= n_tokens:
                break
    ids = tok.encode("".join(words))[:n_tokens]
    return tok.decode(ids)


def area_of(s: corpus.Session) -> str:
    """The module an agent opened first (top two path components)."""
    for _, path in s.file_views:
        parts = path.replace("/testbed/", "").split("/")
        return "/".join(parts[:2])
    return "~"


def build(a: argparse.Namespace) -> None:
    """Write one schedule (a single --move-prob value; main() loops over a list)."""
    a.move_prob = float(a.move_prob)
    tok = Tokenizer(a.tokenizer)
    paths = sorted(Path(a.data).glob("tool-*.parquet"))
    if a.download_shards:
        paths = corpus.download(Path(a.data), "tool", [int(x) for x in a.download_shards.split(",")])
    if not paths:
        raise SystemExit(f"no tool-*.parquet under {a.data}; pass --download-shards 0,1 (or download them)")
    repo = a.repo
    if repo == "auto":
        repo = next(iter(corpus.repo_counts(paths)))
    sessions = corpus.load_sessions(paths, repo)
    need = a.devs * a.agents
    wneed = a.warmup_devs * a.agents
    if len(sessions) < need + wneed:
        raise SystemExit(f"{repo}: {len(sessions)} distinct instances in these shards, need {need + wneed} "
                         f"({need} measured + {wneed} warm-up). Use more shards, a smaller swarm or "
                         f"fewer --warmup-devs (corpus.repo_counts lists repos).")
    rng = random.Random(a.seed)
    rng.shuffle(sessions)

    # --- developers -------------------------------------------------------
    if a.grouping == "area":
        pool = sorted(sessions[: need * 3], key=lambda s: (area_of(s), s.instance_id))[:need]
    else:
        pool = sessions[:need]
    groups = [pool[d * a.agents:(d + 1) * a.agents] for d in range(a.devs)]

    # --- shared text ------------------------------------------------------
    sys_prompt = pool[0].messages[0]["content"]
    base_tokens = len(tok.encode(sys_prompt))
    harness = sys_prompt
    if a.harness_tokens > base_tokens:
        harness += "\n\n" + filler(a.harness_tokens - base_tokens, tok, 0, "## Workspace conventions")

    gap = timing.parse(a.gap)
    # Worker moves draw from their own stream, one draw per idle gap whatever the
    # move probability, so schedules built with different --move-prob values are
    # identical except for which agents move (and the moves at a lower
    # probability are a subset of those at a higher one).
    move_rng = random.Random(a.seed * 7919 + 17)
    placeholder_tokens = len(tok.message("user", schedule.CLEARED_TEXT))
    agents: list[schedule.Agent] = []
    for d, group in enumerate(groups):
        preamble = filler(a.dev_preamble_tokens, tok, a.seed * 1000 + d + 1, f"## Developer {d} notes")
        bundle = ""
        if a.subagent_turns and a.bundle_files:
            pop: Counter[str] = Counter()
            text: dict[str, str] = {}
            for s in group:
                for idx, path in s.file_views:
                    pop[path] += 1
                    text.setdefault(path, s.messages[idx]["content"])
            files = [text[p] for p, _ in pop.most_common(a.bundle_files)]
            if files:
                bundle = "\n\nContext from the lead agent:\n\n" + "\n\n".join(files)
        for j, s in enumerate(group):
            msgs = [dict(m) for m in s.messages]
            msgs[0]["content"] = harness
            first_user = next(i for i, m in enumerate(msgs) if m["role"] == "user")
            msgs[first_user]["content"] = (preamble + "\n\n" if preamble else "") + msgs[first_user]["content"] + bundle
            turns_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
            cap = a.subagent_turns or a.max_turns
            turns_idx = turns_idx[:cap]
            msgs = msgs[: turns_idx[-1] + 1]
            mtoks = [len(tok.message(m["role"], m["content"])) for m in msgs]
            obs = [i for i, m in enumerate(msgs) if m["role"] == "user" and i > first_user]
            cleared: set[int] = set()
            idx = len(agents)
            worker = idx % a.workers
            turns: list[schedule.Turn] = []
            for k, ti in enumerate(turns_idx):
                def plen() -> int:
                    return sum(placeholder_tokens if i in cleared else mtoks[i] for i in range(ti))
                if a.clear_trigger and plen() > a.clear_trigger:
                    past = [i for i in obs if i < ti]
                    cleared |= set(past[:-a.clear_keep] if a.clear_keep else past)
                out_tokens = max(a.min_output_tokens, min(a.max_output_tokens, mtoks[ti] - 4))
                # An idle gap follows this turn's reply; after it, the agent may
                # resume on another worker (the move takes effect from turn k+1).
                idle = a.idle_s if (k + 1 < len(turns_idx) and rng.random() < a.idle_prob) else 0.0
                turns.append(schedule.Turn(msg=ti, max_tokens=out_tokens, worker=worker,
                                           after_s=round(gap(rng), 3), idle_s=idle, cleared=sorted(cleared)))
                if idle and a.workers > 1:
                    u = move_rng.random()
                    target = move_rng.choice([w for w in range(a.workers) if w != worker])
                    if u < a.move_prob:
                        worker = target
            for k in range(1, len(turns)):
                turns[k].moved = turns[k].worker != turns[k - 1].worker
            start = round(d * a.dev_stagger + rng.uniform(0, a.agent_stagger), 3)
            agents.append(schedule.Agent(agent=f"d{d}-a{j}", dev=d, instance_id=s.instance_id,
                                         start_s=start, messages=msgs, turns=turns))

    # --- warm-up cohort ---------------------------------------------------
    # Developers replayed before measuring, untimed (design decision 8): they
    # leave the shared harness prompt and the repo's popular files in L1 and
    # Valkey, as a fleet that has been running would. Drawn from instances the
    # measured developers do not use, with their own random stream, so the
    # measured agents are the same with or without them.
    wrng = random.Random(a.seed * 104729 + 3)
    used = {s.instance_id for s in pool}
    rest = [s for s in sessions if s.instance_id not in used]
    if a.grouping == "area":
        wpool = sorted(rest[: wneed * 3], key=lambda s: (area_of(s), s.instance_id))[:wneed]
    else:
        wpool = rest[:wneed]
    for w in range(a.warmup_devs):
        d = a.devs + w
        group = wpool[w * a.agents:(w + 1) * a.agents]
        preamble = filler(a.dev_preamble_tokens, tok, a.seed * 1000 + d + 1, f"## Developer {d} notes")
        bundle = ""
        if a.subagent_turns and a.bundle_files:
            pop = Counter()
            text = {}
            for s in group:
                for idx, path in s.file_views:
                    pop[path] += 1
                    text.setdefault(path, s.messages[idx]["content"])
            files = [text[p] for p, _ in pop.most_common(a.bundle_files)]
            if files:
                bundle = "\n\nContext from the lead agent:\n\n" + "\n\n".join(files)
        for j, s in enumerate(group):
            msgs = [dict(m) for m in s.messages]
            msgs[0]["content"] = harness
            first_user = next(i for i, m in enumerate(msgs) if m["role"] == "user")
            msgs[first_user]["content"] = (preamble + "\n\n" if preamble else "") + msgs[first_user]["content"] + bundle
            turns_idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"][: a.subagent_turns or a.max_turns]
            msgs = msgs[: turns_idx[-1] + 1]
            mtoks = [len(tok.message(m["role"], m["content"])) for m in msgs]
            obs = [i for i, m in enumerate(msgs) if m["role"] == "user" and i > first_user]
            cleared = set()
            worker = j % a.workers  # every worker gets warmed
            turns = []
            for ti in turns_idx:
                if a.clear_trigger and sum(placeholder_tokens if i in cleared else mtoks[i] for i in range(ti)) > a.clear_trigger:
                    past = [i for i in obs if i < ti]
                    cleared |= set(past[:-a.clear_keep] if a.clear_keep else past)
                out_tokens = max(a.min_output_tokens, min(a.max_output_tokens, mtoks[ti] - 4))
                turns.append(schedule.Turn(msg=ti, max_tokens=out_tokens, worker=worker,
                                           after_s=round(gap(wrng), 3), idle_s=0.0, cleared=sorted(cleared)))
            agents.append(schedule.Agent(agent=f"w{w}-a{j}", dev=d, instance_id=s.instance_id,
                                         start_s=round(wrng.uniform(0, a.agent_stagger), 3), messages=msgs,
                                         turns=turns, warmup=True))

    # --- stats (measured agents) ---------------------------------------------
    lens = []
    for ag in agents:
        if ag.warmup:
            continue
        mt = [len(tok.message(m["role"], m["content"])) for m in ag.messages]
        for t in ag.turns:
            c = set(t.cleared)
            lens.append(sum(placeholder_tokens if i in c else mt[i] for i in range(t.msg)))
    lens.sort()
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": "SWE-bench/SWE-smith-trajectories (tool split)",
        "shards": [p.name for p in paths],
        "repo": repo,
        "tokenizer": tok.name,
        "tokenizer_exact": tok.exact,
        "args": {k: v for k, v in vars(a).items() if k not in ("func",)},
        "agents": sum(1 for ag in agents if not ag.warmup),
        "warmup_agents": sum(1 for ag in agents if ag.warmup),
        "warmup_requests": sum(len(ag.turns) for ag in agents if ag.warmup),
        "requests": len(lens),
        "prompt_tokens": {"p50": lens[len(lens) // 2], "p90": lens[int(0.9 * (len(lens) - 1))],
                          "max": lens[-1], "mean": round(statistics.mean(lens))},
        "move_prob": a.move_prob,
        "idle_gaps": sum(1 for ag in agents for t in ag.turns if t.idle_s),
        "moves": sum(1 for ag in agents for t in ag.turns if t.moved),
    }
    schedule.save(Path(a.output), meta, agents)
    print(f"[generate] {repo}: {meta['agents']} agents, {len(lens)} requests (+ warm-up: {meta['warmup_agents']} agents, "
          f"{meta['warmup_requests']} requests), prompt p50 "
          f"{meta['prompt_tokens']['p50']} / p90 {meta['prompt_tokens']['p90']} tokens "
          f"({tok.name}), {meta['idle_gaps']} idle gaps, {meta['moves']} moves -> {a.output}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("source")
    g.add_argument("--data", default="data/swesmith", help="directory with tool-*.parquet shards")
    g.add_argument("--download-shards", default="", help="comma-separated shard numbers (0-7) to download first")
    g.add_argument("--repo", default="auto", help="SWE-smith repo, e.g. getmoto__moto; auto = the largest")
    g.add_argument("--tokenizer", default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    g = ap.add_argument_group("swarm")
    g.add_argument("--devs", type=int, default=4)
    g.add_argument("--agents", type=int, default=5, help="agents per developer")
    g.add_argument("--warmup-devs", type=int, default=1,
                   help="developers (of --agents each) replayed before measuring, untimed, to start from warm caches; "
                        "0 = no warm-up cohort (replay --cold skips it anyway)")
    g.add_argument("--grouping", choices=("area", "random"), default="area")
    g.add_argument("--max-turns", type=int, default=40, help="cap on turns per agent (long mode)")
    g.add_argument("--subagent-turns", type=int, default=0, help="sub-agent mode: keep the first K turns")
    g.add_argument("--bundle-files", type=int, default=0, help="sub-agent mode: developer's top files prepended")
    g = ap.add_argument_group("prompt")
    g.add_argument("--harness-tokens", type=int, default=8192, help="shared system prompt size")
    g.add_argument("--dev-preamble-tokens", type=int, default=512)
    g.add_argument("--clear-trigger", type=int, default=0, help="clear old tool results above this many tokens")
    g.add_argument("--clear-keep", type=int, default=3)
    g.add_argument("--max-output-tokens", type=int, default=512)
    g.add_argument("--min-output-tokens", type=int, default=16)
    g = ap.add_argument_group("timing and placement")
    g.add_argument("--gap", default="lognormal:median=4,sigma=1.2,max=60",
                   help="tool latency after each reply (timing.py specs)")
    g.add_argument("--idle-prob", type=float, default=0.1)
    g.add_argument("--idle-s", type=float, default=300.0)
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--move-prob", default="0.5",
                   help="chance an agent changes worker after an idle gap; comma-separated (e.g. 0,0.5,1) "
                        "writes one schedule per value, <output>-q<value>, identical except for the moves")
    g.add_argument("--dev-stagger", type=float, default=30.0)
    g.add_argument("--agent-stagger", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--output", required=True)
    return ap


def main() -> None:
    a = parser().parse_args()
    qs = [float(x) for x in str(a.move_prob).split(",")]
    base = a.output
    for q in qs:
        a.move_prob = q
        a.output = base if len(qs) == 1 else f"{base}-q{q:g}"
        build(a)


if __name__ == "__main__":
    main()
