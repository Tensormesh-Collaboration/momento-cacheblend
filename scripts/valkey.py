#!/usr/bin/env python3
"""Minimal Valkey helper for the kit: ping, stats, and wiping this kit's keys.

    scripts/valkey.py ping
    scripts/valkey.py info            # memory, key count, keys under our prefix
    scripts/valkey.py count           # number of keys under our prefix (all primaries)
    scripts/valkey.py blob            # BLOB.INFO on every primary: is valkey-large-object loaded?
    scripts/valkey.py stats           # JSON: evicted_keys, used/max memory, keys under our prefix (summed)
    scripts/valkey.py wipe            # delete keys under VALKEY_KEY_PREFIX@ only

Reads VALKEY_NODES, VALKEY_CLUSTER, VALKEY_TLS, VALKEY_USERNAME,
VALKEY_PASSWORD and VALKEY_KEY_PREFIX from the environment (lib/env.sh exports
them). Standard library only: it speaks RESP over a socket, so it runs on any
host without valkey-cli.

`wipe` never flushes the database. Momento's deployment may hold other data,
so it SCANs for `<prefix>@*` (LMCache joins key_prefix and its wire key with
'@') on every primary and UNLINKs what it finds. It refuses an empty prefix.
"""

from __future__ import annotations

import os
import socket
import ssl
import sys


class Conn:
    def __init__(self, host: str, port: int):
        s = socket.create_connection((host, port), timeout=10)
        if os.environ.get("VALKEY_TLS", "0") == "1":
            ctx = ssl.create_default_context()
            if os.environ.get("VALKEY_TLS_INSECURE", "0") == "1":
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            s = ctx.wrap_socket(s, server_hostname=host)
        self.s = s
        self.f = s.makefile("rb")
        user, pw = os.environ.get("VALKEY_USERNAME", ""), os.environ.get("VALKEY_PASSWORD", "")
        if pw:
            self.cmd("AUTH", *([user] if user else []), pw)

    def cmd(self, *args: str):
        out = f"*{len(args)}\r\n".encode()
        for a in args:
            b = str(a).encode()
            out += b"$%d\r\n%s\r\n" % (len(b), b)
        self.s.sendall(out)
        return self._read()

    def _read(self):
        line = self.f.readline().rstrip(b"\r\n")
        kind, rest = line[:1], line[1:]
        if kind == b"+":
            return rest.decode()
        if kind == b"-":
            raise RuntimeError(rest.decode())
        if kind == b":":
            return int(rest)
        if kind == b"$":
            n = int(rest)
            if n < 0:
                return None
            data = self.f.read(n + 2)[:-2]
            return data.decode(errors="replace")
        if kind == b"*":
            n = int(rest)
            return None if n < 0 else [self._read() for _ in range(n)]
        raise RuntimeError(f"unexpected reply {line!r}")


def nodes() -> list[tuple[str, int]]:
    out = []
    for hp in os.environ.get("VALKEY_NODES", "127.0.0.1:6379").split(","):
        host, _, port = hp.strip().rpartition(":")
        out.append((host or "127.0.0.1", int(port or 6379)))
    return out


def primaries() -> list[tuple[str, int]]:
    """Every primary: the seed alone in standalone mode, CLUSTER NODES otherwise."""
    seed = nodes()[0]
    if os.environ.get("VALKEY_CLUSTER", "0") != "1":
        return [seed]
    out = []
    for line in Conn(*seed).cmd("CLUSTER", "NODES").splitlines():
        parts = line.split()
        if len(parts) > 2 and "master" in parts[2] and "fail" not in parts[2]:
            hostport = parts[1].split("@")[0]
            host, _, port = hostport.rpartition(":")
            out.append((host or seed[0], int(port)))
    return out


def scan_prefix(c: Conn, pattern: str):
    cursor = "0"
    while True:
        cursor, keys = c.cmd("SCAN", cursor, "MATCH", pattern, "COUNT", "1000")
        yield from keys
        if cursor == "0":
            return


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in ("ping", "info", "count", "blob", "stats", "wipe"):
        print(__doc__)
        return 2
    action = sys.argv[1]
    prefix = os.environ.get("VALKEY_KEY_PREFIX", "")
    pattern = f"{prefix}@*"
    try:
        targets = primaries()
        if action == "stats":
            # Before/after each leg (run_leg.sh): evicted_keys must not grow, or
            # Valkey's own capacity, not L1, limited the run.
            import json
            out = {"evicted_keys": 0, "used_memory": 0, "maxmemory": 0, "keys_under_prefix": 0, "primaries": len(targets)}
            for h, p in targets:
                c = Conn(h, p)
                info = {}
                for section in ("stats", "memory"):
                    info.update(l.split(":", 1) for l in c.cmd("INFO", section).splitlines() if ":" in l)
                for k in ("evicted_keys", "used_memory", "maxmemory"):
                    out[k] += int(info.get(k, 0) or 0)
                if prefix:
                    out["keys_under_prefix"] += sum(1 for _ in scan_prefix(c, pattern))
            print(json.dumps(out))
            return 0
        if action == "count":
            print(sum(sum(1 for _ in scan_prefix(Conn(h, p), pattern)) for h, p in targets) if prefix else 0)
            return 0
        for host, port in targets:
            c = Conn(host, port)
            if action == "ping":
                print(f"{host}:{port} {c.cmd('PING')}")
            elif action == "blob":
                # The rdma and blob-tcp transports need the valkey-large-object
                # module; LMCache's adapter checks it the same way (BLOB.INFO).
                info = c.cmd("BLOB.INFO")
                flat = info if isinstance(info, str) else " ".join(str(x) for x in (info or []))
                print(f"{host}:{port} BLOB.INFO ok: {flat[:300]}")
            elif action == "info":
                info = dict(l.split(":", 1) for l in c.cmd("INFO", "memory").splitlines() if ":" in l)
                try:
                    mods = c.cmd("MODULE", "LIST") or []
                except RuntimeError:  # managed services may forbid MODULE
                    mods = []
                names = [m[1] for m in mods if isinstance(m, list) and len(m) > 1]
                ours = sum(1 for _ in scan_prefix(c, pattern)) if prefix else "n/a"
                print(f"{host}:{port} used_memory={info.get('used_memory_human', '?')} "
                      f"maxmemory={info.get('maxmemory_human', '?')} policy={info.get('maxmemory_policy', '?')} "
                      f"dbsize={c.cmd('DBSIZE')} keys[{pattern}]={ours} modules={names}")
            else:
                if not prefix:
                    print("refusing to wipe: VALKEY_KEY_PREFIX is empty", file=sys.stderr)
                    return 1
                n, batch = 0, []
                for k in scan_prefix(c, pattern):
                    batch.append(k)
                    if len(batch) >= 500:
                        n += c.cmd("UNLINK", *batch)
                        batch = []
                if batch:
                    n += c.cmd("UNLINK", *batch)
                print(f"{host}:{port} unlinked {n} keys matching {pattern}")
    except (OSError, RuntimeError) as exc:
        print(f"valkey {action} failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
