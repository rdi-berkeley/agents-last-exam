# Network-off tasks

A task card can require that the agent works without internet access:

```json
"vm": { "snapshot": "cpu-free-ubuntu", "network": { "mode": "off" } }
```

`mode` is `"off"` or `"on"`. A card without `vm.network` runs with normal
network access. Any other value fails task loading.

## Experiment settings

```yaml
network_off_policy: enforce          # default: allow_unenforced
network_allowlist: [openrouter.ai]   # optional extra hosts
```

| Task card | `network_off_policy` | Result |
|---|---|---|
| no `vm.network`, or `mode: on` | any | Unchanged |
| `mode: off` | `allow_unenforced` (default) | Normal networking; run.json records `environment.network.enforced: false` |
| `mode: off` | `enforce`, QEMU provider, `sandbox` executor | Sealed run (below) |
| `mode: off` | `enforce`, any other provider or executor | Fails before a VM is created |

Under `enforce`, the allowlist is the host of the agent config's `base_url`
plus `network_allowlist`. Entries are exact DNS names; only port 443 is relayed.
A config without `base_url` must list its model host in `network_allowlist`.

## How a sealed QEMU run works

1. The VM boots with normal networking, and the deployer's `install()` runs
   (agent CLIs are installed from npm, PyPI or GitHub at this point).
2. The in-guest entry asks the host to seal the sandbox and waits.
3. The host starts an egress gateway for this run on the Docker bridge
   address, then adds rules in the QEMU runner container's network namespace.
   Inside the guest, `172.30.0.1:3128` is the gateway. Every other packet the
   guest sends is dropped, IPv4 and IPv6, including DNS. Replies on connections
   made into the guest (CUA, VNC) still flow. The agent cannot change these
   rules because they live outside the VM.
4. A probe inside the guest checks that direct TCP and DNS fail, that the
   gateway refuses an unlisted host, and that every allowlisted host is
   reachable. Any failure fails the run before the agent starts.
5. The agent starts with `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`, `NO_PROXY`
   and `NODE_USE_ENV_PROXY=1` set.
6. The sandbox stays sealed through evaluation. The gateway is closed when the
   VM is released, so a kept VM has no egress at all.

The gateway accepts only `CONNECT <allowlisted-host>:443` and relays TLS
without decrypting it. run.json records the allowlist, the probe report and the
gateway's allowed and denied connection counts under `environment.network`.

## Limitations

- Only the QEMU provider can seal a sandbox.
- An agent CLI that ignores the proxy variables cannot reach its model, and the
  run fails. Check each harness with a smoke run before using it.
- Provider-side tools that reach the web through the model API (for example
  server-side web search) are not blocked, because they travel over the
  allowlisted model connection.
- An evaluator that needs the network inside the guest fails, because the
  sandbox stays sealed.
- The CUA server in the v1.1 images does not start listening while its
  telemetry host (PostHog) is unreachable. Sealing happens after CUA is up, so
  runs are unaffected, but a CUA restart inside a sealed VM does not recover.
  Rebuilt images should set `CUA_TELEMETRY_ENABLED=false` and
  `CUA_TELEMETRY_DISABLED=true` for the CUA service. Do not allowlist the
  telemetry host: it accepts arbitrary payloads.
