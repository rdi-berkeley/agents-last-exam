# Sandbox Network Isolation — Design & Rationale

**Status:** schema + plumbing landed (fail-closed); **Google Cloud enforcement implemented & validated e2e**; other providers still fail-closed
**Branch:** `feat/sandbox-network-isolation`

## Goal

Stop the agent under test from reaching the open internet — specifically from
**web-searching for the answer** — while still letting it call the model API it
needs to think. The policy should be **per-task / per-experiment configurable**
and work **identically across every provider** (`gcloud`, `qemu`, `docker`,
`static`).

This document explains the networking background from first principles, then
argues for the chosen design and specifies it. It assumes no prior networking
knowledge.

---

## Part 1 — Background, from scratch

You only need six ideas. Everything in Part 2 follows from them.

### 1.1 A network connection is `(destination IP, destination port)`

When any program "makes a request", underneath it opens a TCP connection to a
**destination IP address** (which machine) on a **destination port** (which
service on that machine). A port is just a number, 0–65535, that says "what kind
of traffic this is". Some are conventional:

| Port | Service |
|------|---------|
| 80   | HTTP (plaintext web) |
| 443  | HTTPS (encrypted web) |
| 53   | DNS (name lookups) |

So "the agent calls OpenRouter" and "the agent googles the answer" both come out
as: *open a TCP connection to some IP on port 443*.

### 1.2 HTTPS = HTTP inside TLS (encryption)

`https://…` is ordinary HTTP wrapped in **TLS**, an encryption layer. Once the
connection is established, everything inside — the URL path, headers, the request
body, the response — is **encrypted**. A machine sitting in the middle sees only
ciphertext; it cannot read *what* you asked for.

**This is the crux of the whole problem.** Because the payload is encrypted, you
cannot look at a packet and tell "this is a model API call" from "this is a
Google search". At the protocol level they are *identical*: both are TLS on
port 443.

### 1.3 …but the destination is *not* fully hidden

Two pieces of a TLS connection are visible in plaintext even though the payload
is encrypted:

- **The destination IP** — unavoidable; the network has to know where to route
  the packets.
- **The SNI (Server Name Indication)** — at the very start of a TLS handshake
  the client announces, *in plaintext*, the hostname it wants to talk to (e.g.
  `openrouter.ai`). This is how one IP can serve many HTTPS sites. It is sent
  before any encryption kicks in, so a middlebox **can read the hostname**
  without decrypting anything.

So: you can't filter HTTPS by *content*, but you **can** filter it by
**destination hostname or IP**. That single fact drives the entire design.

### 1.4 DNS: names → IPs (and a leak channel)

Programs speak in names (`openrouter.ai`); the network speaks in IPs
(`104.18.x.x`). **DNS** is the lookup that turns one into the other, normally a
request to a DNS server on port 53. Two consequences:

- If you cut a machine off from DNS, it usually can't reach anything by name.
- DNS is itself an exfiltration channel: a program can smuggle data out by
  encoding it into names it "looks up" (`SECRET.attacker.com`). So a serious
  isolation design has to think about DNS, not just ports 80/443.

### 1.5 Firewall / egress filtering

A **firewall** is a rule list that allows or drops connections. **Egress**
filtering = rules on *outbound* connections (from the VM to the world), as
opposed to **ingress** (inbound, from the world to the VM). Rules match on
`(direction, destination IP/range, port)`. Two default stances:

- **Denylist / blacklist** — allow everything, block a named few. Leaky: you can
  never enumerate every site an agent could cheat through.
- **Allowlist / whitelist** — block everything, allow a named few. Safe by
  default: anything you forgot stays blocked.

Plain firewalls match on **IP**, not hostname (they operate below the layer
where names exist).

### 1.6 A forward proxy (the key tool)

A **forward proxy** is a middleman you *route your own outbound traffic through*.
Instead of the VM connecting straight to the internet, it connects to the proxy
and says "please reach X for me". Why this matters for HTTPS:

- To tunnel HTTPS, the client sends the proxy a plaintext line:
  **`CONNECT openrouter.ai:443`**. The proxy therefore learns the destination
  **hostname in the clear** *before* any encrypted bytes flow — and can decide
  **allow or deny by hostname**, then blindly pipe the (still-encrypted) bytes
  through. **It never decrypts anything.** No fake certificates, no seeing your
  traffic — it only gates *where* you're allowed to go.
- The proxy does the **DNS lookup** on the client's behalf, so the VM itself
  needs no internet DNS at all.

This is exactly how corporate "you can't reach Facebook from the office"
controls work, and it is the mechanism this design uses.

### 1.7 Where does the agent run? (Recap — decides *where* to enforce)

In ALE the agent process runs in one of two places, chosen per-agent by its
executor (`ale_run/base_interface/executor.py`):

- **In-sandbox (`SandboxExecutor`, the default for 11/13 agents** — codex,
  claude_code, hermes, openhands, gemini, grok, cursor, droid, forgecode,
  openclaw, terminus_2): the CLI runs **inside the VM** and the model API key +
  `base_url` are injected into the VM (`_secrets.json` → in-VM env). **The VM
  itself makes the outbound HTTPS call to the model.** This is the case that
  makes "just cut the network" impossible.
- **Host-side (`LocalExecutor`/`DockerExecutor`** — `ale_claw`, `dummy`): the
  agent loop runs on the host and only drives the VM for tool actions over the
  cua control channel. Here the VM needs **no** model access and can be fully
  air-gapped.

**Enforcement must live somewhere the agent cannot reach.** The in-sandbox agent
may well have root inside the guest, so a firewall configured *inside the guest*
is something it could, in principle, tear down. Enforcing **outside the VM**
(host proxy + provider-level firewall) is tamper-proof by construction.

---

## Part 2 — Why this design is optimal (options compared)

The requirement, restated with the concepts above: *block all egress by
destination, except the one hostname the model API lives at; enforce it outside
the guest; make it work on every provider with near-zero per-experiment config.*

### The tempting-but-wrong idea: "block HTTP/HTTPS"

The intuition "just block http/https and the API still works" **does not hold**,
because the model API call *is* HTTPS on port 443 — the same protocol and port
as a web search (§1.2). Blocking the protocol blocks the model too; not blocking
it leaves search open. Protocol/port is the one axis on which cheating traffic
and model traffic are **indistinguishable**. The only axis that separates them is
**destination** (§1.3).

### Option comparison

| Option | How | Verdict |
|--------|-----|---------|
| **A. Block by protocol/port** | drop 80/443 | ✗ kills the model call too — see above |
| **B. In-VM denylist** (block known search engines) | iptables blocklist in guest | ✗ leaky (can't enumerate every cheat route) **and** tamperable (agent has root) |
| **C. In-VM allowlist** (block all, allow model) | iptables allowlist in guest | ✓ safe-by-default, but **tamperable** (in-guest) and can't match on hostname (CDN IPs churn — see below) |
| **D. Provider egress firewall, IP allowlist** | GCP VPC egress rules / container iptables allowing only the model's IPs | ✓ tamper-proof, but **breaks on CDN IP churn** and is **per-provider IP bookkeeping** |
| **E. Host forward proxy + provider egress cut-off, allowlist by hostname** | VM's only exit is a host-side proxy that allows by `CONNECT` hostname; provider firewall drops all other egress | ✓✓ **recommended** |

### Why E wins — point by point

1. **It filters on the one axis that actually separates the traffic:
   destination hostname.** Not protocol (can't), not payload (encrypted), not
   raw IP (churns) — the stable, human-meaningful *name*.

2. **No TLS decryption, no privacy/cert headaches.** The proxy gates on the
   plaintext `CONNECT host:443` / SNI and pipes the still-encrypted bytes
   through untouched (§1.6). We never see or tamper with model traffic, and we
   don't have to install a fake root CA in every guest (which would itself be a
   fragile, per-OS mess).

3. **Immune to CDN IP churn.** Model endpoints (OpenRouter, Anthropic, …) sit
   behind CDNs like Cloudflare: the set of IPs behind `openrouter.ai` is large
   and rotates. An **IP** allowlist (Options C/D) would need constant updating
   and would randomly break mid-run. A **hostname** allowlist is stable — the
   name doesn't change even as the IPs behind it do.

4. **Closes the DNS hole for free.** With all direct egress cut, the VM has no
   route to an internet DNS server; name resolution is done *by the proxy*. So
   DNS-tunnel exfiltration/lookups die automatically (§1.4) — no extra rule.

5. **Tamper-proof.** The proxy and the egress firewall live **outside** the
   guest. Even a root agent inside the VM cannot reach them; its only path to the
   world is the pinhole we control (§1.7).

6. **Cross-provider unification with a single policy brain.** The allow/deny
   logic lives **once**, in the proxy. Each provider only has to do the crude,
   easy thing — *cut all direct egress and route the VM at the proxy*. We are not
   writing four different allowlist engines; we're writing one, plus four
   two-line "point the VM here" adapters. (And `qemu` nearly gets it for free —
   its VM already runs inside a Docker container with `NET_ADMIN` and existing
   `iptables` NAT rules, `ale_run/environments/providers/qemu.py`.)

7. **Zero per-experiment allowlist maintenance.** This addresses the original
   worry directly. Different experiments use different `base_url`s — but **the
   framework already knows the `base_url`**: it's the value it injects as
   `ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL`
   (`ale_run/orchestration/lifecycle.py:_collect_env_passthrough`,
   `ale_run/agents/claude_code/deployer.py:_build_env`). The proxy's allowlist is
   **derived from that same value** — `host = urlparse(base_url).netloc`. Change
   the experiment's `base_url` and the allowlist follows automatically. Nobody
   hand-maintains a list.

8. **Auditable / cheat-detecting.** Because every outbound attempt transits the
   proxy, *denied* attempts can be logged. "Agent tried to reach `google.com` at
   step 14" becomes an observable signal, not a silent success — useful for
   benchmark integrity analysis, not just prevention.

The net trade: E costs us one small always-on host component (the proxy) and a
per-provider "cut egress + set proxy" adapter. In return every hard part
(hostname matching, CDN churn, DNS, tamper-resistance, cross-provider
uniformity, per-experiment config) collapses to a solved problem.

---

## Part 3 — The design

### 3.1 One policy object, provider-agnostic

Add a network policy to the sandbox request (`SandboxSpec`,
`ale_run/base_interface/sandbox.py`):

```python
@dataclass(frozen=True)
class NetworkPolicy:
    mode: Literal["open", "allowlist", "off"] = "open"
    #   open      → today's behaviour (no restriction)
    #   allowlist → block all egress except `allow`
    #   off       → full air-gap (allow == [])
    allow: tuple[str, ...] = ()      # extra hostnames beyond the auto-derived model host
```

- `SandboxSpec` gains `network: NetworkPolicy = NetworkPolicy()`.
- **Default allow set is auto-derived**, not written by hand: the resolved agent
  config's `base_url` host is unioned into the allow set before enforcement
  (`NetworkPolicy.effective_allow` / `model_host_from_env`), so `allowlist` mode
  "just works" for any experiment without naming the endpoint.

### 3.1a Where the policy is read from — **per task card** (decided)

The policy is declared **per task**, in each task's `task_card.json`, under the
existing `vm` block (alongside `snapshot` / `machineType` / `timeout`). This is
the same place every other per-task sandbox knob already lives, and it flows
through the same three-hop chain:

```
task_card.json  "vm": { …, "network": { "mode": …, "allow": [ … ] } }
      │  ale_run/tasks/loader.py       _enrich_with_task_card()  → NetworkPolicy.from_card
      ▼
task_meta["network"]
      │  ale_run/orchestration/lifecycle.py  _build_env_spec()
      ▼
SandboxSpec.network
      │  ale_run/environments/env.py   reset_async()  → provider.assert_network_supported (fail-closed)
      ▼
Provider.acquire(spec)   ← enforcement point (egress cut-off + proxy)
```

Field standard (mirrors the existing `vm` fields):

```jsonc
"vm": {
  "snapshot": "cpu-free-ubuntu",
  "machineType": "c4-standard-4",
  "timeout": 7200,
  "network": {
    "mode": "allowlist",              // "open" (default) | "allowlist" | "off"
    "allow": ["pypi.org"]              // optional; extra hosts beyond the auto-derived base_url host
  }
}
```

- **Omitting `network` = `open`** → every existing task card is unchanged and
  keeps today's behaviour (backward compatible).
- `allow` never needs to list the model endpoint — it rides in from the injected
  `base_url`.
- `mode: "off"` is full air-gap; the model host is *not* auto-added (suited to
  host-side agents that don't call the model from inside the VM).
- Malformed `vm.network` fails **at task load**, not mid-run.

### 3.2 Components

```
                 ┌────────────────────── host / control plane ──────────────────────┐
   experiment    │                                                                    │
   config  ──────┼──►  resolve NetworkPolicy ──► allow = { host(base_url) } ∪ extra   │
                 │                     │                                              │
                 │                     ▼                                              │
                 │         egress proxy (allow by CONNECT/SNI hostname)  ◄────────────┼─┐
                 └────────────────────────────────────────────────────────────────────┘ │ only
                                                                                          │ permitted
   ┌──────────────────────────── sandbox VM (agent runs here) ───────────────────────┐   │ pinhole
   │  agent CLI  ──HTTPS──►  (HTTPS_PROXY=host:proxy_port)  ──────────────────────────┼───┘
   │  everything else ─────►  DROP  (provider egress firewall: no direct internet)    │
   │  cua-server :5000  ◄── ingress from host (control channel, unaffected)           │
   └─────────────────────────────────────────────────────────────────────────────────┘
```

Three moving parts:

1. **Egress proxy** (host-side or a sidecar). Accepts `CONNECT host:port`,
   allows iff `host ∈ allow`, otherwise 403 + logs. Does its own DNS. Candidate
   implementations to evaluate in exploration: `tinyproxy` (tiny, allowlist via
   `Allow`/filter), `squid` (heavier, ACL by dstdomain), or a ~100-line Go/Python
   `CONNECT` proxy we own (most control over logging + allow logic). No TLS
   interception in any case.

2. **Provider egress cut-off** — each provider drops all direct VM egress and
   leaves exactly one route: to the proxy.

3. **Env injection** — set `HTTPS_PROXY` / `HTTP_PROXY` (and `NO_PROXY` for
   loopback/cua) in the agent's environment, alongside the existing `base_url` /
   key injection. In-sandbox: into the VM env. Host-side agents: not needed
   (they can be `off`/air-gapped).

### 3.3 Per-provider enforcement (the thin adapters)

| Provider | Cut direct egress | Route to proxy | Notes |
|----------|-------------------|----------------|-------|
| **qemu** | `iptables` in the runner container (already has `--cap-add NET_ADMIN` + hairpin NAT, `qemu.py`) — default-DROP the FORWARD/egress chain | allow VM → proxy addr; VM reaches host proxy via the container gateway `172.30.0.1` | **cheapest** — the container is already the network boundary |
| **gcloud** | VPC **egress** firewall on tag `ale-run` (VMs are already tagged, `gcloud.py:_build_create_args`): low-priority `deny all egress` + higher-priority `allow → proxy IP` | proxy runs on a VM-reachable host/IP (e.g. a small always-on proxy VM in the same VPC, or the orchestrator if routable) | GCP firewall is **stateful** → the cua **ingress** channel and its replies are unaffected; no `--no-address` needed |
| **docker** | container network / `iptables` allowlist | proxy on host, reachable via the docker gateway | mirrors qemu |
| **static** | no lifecycle control → fall back to an **in-guest** firewall baked into the image (weaker tamper story; acceptable for dev/debug substrate) | env injection only | documented limitation |

Key property: **the cua `/cmd` control channel (port 5000) is ingress**,
host→VM. Egress rules don't touch it, and stateful firewalls auto-allow its
return traffic. So isolating the VM never severs orchestration.

### 3.4 Config wiring & auto-derivation

1. Config loader reads `network:` (mode + extra allow) into `SandboxSpec`.
2. At acquire time the provider resolves the **effective allow set** =
   `{ host(base_url) } ∪ policy.allow`, where `base_url` comes from the same
   resolved agent config already used for env injection.
3. Provider applies its egress cut-off; proxy is configured/handed the allow set;
   proxy env is injected into the agent.

No experiment ever writes the model hostname explicitly — it rides along with the
`base_url` it already sets.

---

## Part 4 — Open questions (to resolve during exploration)

- **Proxy implementation choice** — tinyproxy vs squid vs a small owned proxy.
  Leaning to an owned `CONNECT` proxy for clean allowlist + denied-attempt logs.
- **Proxy topology on gcloud** — cheapest reachable place to run one proxy for
  many concurrent task VMs (shared proxy VM in-VPC? per-run? orchestrator-hosted
  if routable to the VMs?).
- **Windows guests** — proxy env var honoring (`HTTPS_PROXY` is respected by most
  CLIs; WinHTTP/system proxy may need separate setting); validate per harness.
- **Legitimately-needed extra endpoints** — telemetry (OTEL exporters, cf. the
  claude_code/codex telemetry PRs), any runtime package fetches. Images are
  pre-baked so runtime installs *should* be unnecessary; confirm per agent and
  add to `allow` only where required.
- **`plain HTTP` / non-proxy-aware clients** — anything that ignores
  `HTTPS_PROXY` still can't escape (direct egress is dropped), it just fails
  instead of being proxied; decide whether that's acceptable or needs transparent
  redirection.
- **Validation plan** — a task that (a) confirms the model call still succeeds,
  and (b) asserts a deliberate `curl https://example.com` is refused, run on
  gcloud + qemu.

## Part 5 — Implementation: Google Cloud (`aleguard`)

The enforcement that ships in this PR is **Google Cloud only**. A design
refinement vs Part 3: rather than a *host-side* proxy plus a VPC egress
firewall, gcloud enforces **in-guest** with a tiny transparent proxy
(`aleguard`) + nftables. This is simpler (no extra proxy VM to run per-run, no
VPC firewall bookkeeping), needs zero agent cooperation (no `HTTPS_PROXY`), and
keeps every change inside the gcloud provider. Trade-off: enforcement lives
inside the guest, so it defends against an agent using its normal tools — the
realistic cheat — not against an actively adversarial root that flushes
nftables. (A future host/VPC-layer variant can close that gap; the policy
surface is unchanged.)

**How it works** (`ale_run/environments/providers/aleguard.py`, applied by
`GcloudProvider._apply_network_policy` at `acquire`, Linux only):

1. `aleguard` — a stdlib-only transparent proxy — runs in the VM as an
   unprivileged `aleproxy` user.
2. nftables `nat/output` **REDIRECTs** all `:80/:443` (except `aleproxy`'s own
   egress) to it; `filter/output` is **default-drop**, permitting only
   loopback, established (so the cua control channel's replies survive), DNS,
   and `aleproxy`.
3. For each connection `aleguard` reads the **original destination**
   (`SO_ORIGINAL_DST`) and the target **hostname** from the plaintext TLS SNI /
   HTTP Host — no decryption — and forwards iff the host matches the allow-list,
   else drops and logs `DENY`.
4. `off` mode installs a drop-all ruleset with no proxy (true air-gap; cua still
   works because it is ingress).

The allow-list is the model endpoint host, **auto-derived** from the agent
config by `resolve_model_host` (explicit `base_url`, else the provider's
default) and folded in by `_build_env_spec` — so a task card never names it and
it tracks endpoint swaps. `GcloudProvider.enforces_network_policy = True`.

Three kernel details this required (all discovered/fixed via e2e, see the fix
commit): disable IPv6 on the guest (dual-stack clients prefer a CDN's AAAA and
bypass the v4 redirect), `sysctl route_localnet=1` (or the REDIRECT-to-127.0.0.1
packet is dropped as martian), and accept `ip daddr 127.0.0.0/8` in the filter
chain (a redirected packet's `oif` is not yet `lo` when the filter chain sees
it).

### What shipped in this PR

- `NetworkPolicy` + `SandboxSpec.network`; `vm.network` parsed/validated at load
  and threaded to the spec; `resolve_model_host` auto-derivation.
- **Fail-closed guard** `Provider.assert_network_supported` (in
  `ALEEnv.reset_async`): a non-`open` policy on a provider without
  `enforces_network_policy` **raises** rather than running unisolated. `open`
  changes nothing → every existing task card is unaffected.
- `aleguard` + `GcloudProvider` enforcement (this section).
- Anti-cheat demo task `demo/netprobe`.
- Unit tests: `tests/test_network_policy.py`, `tests/test_aleguard.py`
  (real-ClientHello SNI parsing, nft/setup builders).

### Validated end-to-end on Google Cloud (`cpu-free-ubuntu`)

- **Core allowlist:** model hosts reachable through the proxy
  (`openrouter.ai` 200, `api.anthropic.com` 404, `yunwu.ai` 200); every other
  host (`google.com` / `example.com` / `github.com`) blocked (`000`); aleguard
  log shows `ALLOW` model-hosts + `DENY` the rest.
- **Endpoint-agnostic:** the three distinct endpoints above all pass — swapping
  `base_url` later needs no code change.
- **Off mode:** model host *and* everything else blocked; cua control channel
  still alive (air-gap with orchestration intact).
- **Concurrency ×4:** four independent VMs each enforce correctly; proxied
  model-call latency ~0.12 s per VM (per-VM proxy, no shared bottleneck).
- **Pipeline integration:** a real `ale_run` run applied
  `mode=allowlist allow=['openrouter.ai']`, auto-derived from the claude_code
  config.
- **Agent-runtime model path:** Node.js (v24, the claude/codex CLI runtime)
  `https.get` reached `openrouter.ai/api/v1/models` (200) through the proxy and
  was reset on `google.com` — the real client stack, not just curl.

### Not yet done (follow-ups)

- Enforcement for the other providers (`qemu` cheapest — the VM already runs in
  a `NET_ADMIN` container; then `docker`/`static`), each flipping its
  `enforces_network_policy` flag.
- Optional host/VPC-layer enforcement for tamper-resistance against a root
  agent.
- Baked task data for `demo/netprobe` so the anti-cheat A/B runs through the
  full pipeline (its enforcement path is already validated; only the task's
  input data isn't baked into the image).
- The open questions in Part 4.
