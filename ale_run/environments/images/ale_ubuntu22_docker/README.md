# ale-ubuntu22-docker — container build & key decisions

The container form of the `ale-ubuntu22` Linux sandbox, so the ~110 `cpu-free-ubuntu`
(no-GPU, no-license) tasks can run under the **docker provider** on a single host
instead of one GCE VM per task. Published at
**`agentslastexam/ale-ubuntu22-docker:latest`** — pull it, point the docker
provider at it, and it runs; nothing else to set up.

- `Image` registry entry: `../ale_ubuntu22_docker.py`
- Provider + config: `ale_run/environments/providers/docker.py`, `configs/environments/docker.yaml`
- This directory: build/bake tooling only (mirrors `../ale_kasm/`).

## How the base is built — rootfs export, not a rebuild

A container shares the host kernel, so we don't rebuild packages: `build.sh` **exports
the `ale-ubuntu22` VM's root filesystem** (dropping kernel/boot/init/logs) and
`docker import`s it. End to end ~2–3 h, ~146 GB image.

```bash
./build.sh           # export → import → finalize → smoke; resumable
```

| env | default | meaning |
|-----|---------|---------|
| `ALE_BUILD_VM` / `ALE_BUILD_ZONE` | `dev-ubuntu22` / `us-west2-a` | source VM |
| `ALE_BUILD_SSH_USER` / `ALE_BUILD_SSH_KEY` | `weichenzhang` / `~/.ssh/google_compute_engine` | SSH identity |
| `ALE_BUILD_IMAGE` | `ale-ubuntu22-docker:latest` | output tag |
| `ALE_BUILD_WORKDIR` | `~/.cache/ale-docker-build` | rootfs-tar scratch (~100 GB free) |
| `ALE_FORCE_EXPORT=1` / `ALE_KEEP_VM=1` | — | re-export / don't stop the VM |

Rootless docker maps uids through a 65536-wide range, so out-of-range uids/gids in
the VM tar are clamped to 0 (VM-side `chown`, or `remap_uids.py` as a fallback)
before import, preserving uid 1000 = `user`.

## Key decision 1 — nested Docker (DinD), because some evals run `docker` *inside* the sandbox

A handful of tasks need a Docker daemon **inside** the sandbox: their agent work
and/or scoring run `docker run` / `docker compose` / `minikube --driver=docker`:

| task | needs | nested image |
|------|-------|--------------|
| `engineering/openroad_sky130_ibex_pnr_signoff` | eval reruns the pinned OpenROAD flow | `openroad/orfs@sha256:fd77…` |
| `computing_math/k8s_migration_1` | `minikube start --driver=docker` (K8s node = a container) | `gcr.io/k8s-minikube/kicbase:v0.0.42` |
| `business_finance/bpmn_*_l3` (×2) | `docker compose` up Flowable | `flowable/all-in-one:6.5.0` |

On a GCE VM this is just the VM's own dockerd. In a **container** it becomes
Docker-in-Docker, which hits two walls:

1. **Privileged.** A nested dockerd needs `--privileged` to set up its cgroups/mounts.
   Under **rootless** host docker this is user-namespace-bounded (container root maps
   to the unprivileged host user) — it does not grant host root.
2. **overlay2-on-overlay2 is unsupported.** The container's rootfs is already an
   overlay2 mount, and the kernel won't stack overlayfs on overlayfs. So the nested
   daemon runs on **`fuse-overlayfs`** (a userspace overlay without that limit) with a
   fresh data-root at `/var/lib/dind`.

Why not just mount the host's `docker.sock` (DooD)? Because the host daemon resolves
`-v` bind-mounts and `localhost` in the **host's** namespaces, not the sandbox's — so
workspace mounts and `localhost:8080` break — and N parallel sandboxes collide on one
daemon (container names, ports). DinD gives each sandbox its own daemon over its own
fs + network.

### Deferred: DinD tasks (currently OFF)

DinD is **gated off by default** (`ALE_ENABLE_DIND != 1`) and those **4 tasks are
excluded** — the project doesn't run them on Docker yet. Reasons: starting a
fuse-overlayfs dockerd in *every* container is pure overhead for the ~101 tasks that
don't need it, and at concurrency it's an I/O storm that starves cua-server startup.
The nested images stay **baked into the image** (dormant), so re-enabling is cheap:
set `enable_dind: true` **and** `privileged: true` in `docker.yaml`, drop the 4 tasks
back into the task list, and (ideally) cap concurrency for them. Revisit later.

## Key decision 2 — nested images are *baked in* (commit-bake), so there's zero per-start load

The nested image store is **pre-populated into `/var/lib/dind` at build time** and
shipped inside the image. A pulled container starts its nested dockerd and the images
(incl. orfs' `@sha256` RepoDigest) are already there — **no load, no network, no
per-start overhead**. Validated to survive `docker commit` → push → pull intact.

Done by `bake_nested_images.sh`: boot a container, load/pull the nested images into a
fuse-overlayfs `/var/lib/dind`, `docker commit`. The entrypoint is **self-adapting**:
if `/var/lib/dind` is already populated it does nothing; if it's empty but tarballs
exist at `/opt/ale-docker-images/`, it background-loads them (so the build/bake step
itself can populate the store). The load is always **backgrounded** — loading 5+ GB
before `exec cua-server` would blow the provider's 120 s cua-ready timeout and fail
every acquire.

## Key decision 3 — container size/display mirror the task's GCE machine

The provider reads the task card's `vm.machineType` (e.g. `c4-standard-4`) and
translates it to `--cpus`/`--memory` (4 CPU / 15 GB) so a Docker sandbox gets the same
shape gcloud would give — reusing gcloud's own parser. `resolution` (default
`[1024,768]`) sets the in-container Xvfb size via `ALE_SCREEN_RESOLUTION`. Both are
overridable in `docker.yaml` (`cpus`/`memory` as a host cap).

## Entrypoint — `/dockerstartup/vnc_startup.sh`

Invoked by the provider (`docker run --entrypoint /dockerstartup/vnc_startup.sh`).
A container has no init, so it reproduces what the VM's systemd units did:
1. start the nested dockerd on fuse-overlayfs **only if `ALE_ENABLE_DIND=1`** (off by
   default — see "Deferred: DinD tasks"; best-effort, see decision 2);
2. `Xvfb :0` at `ALE_SCREEN_RESOLUTION` (cua-server drives X via pynput/python-xlib);
3. `exec` the cua computer-server on `:5000` as PID 1.

It also exports `CUA_TELEMETRY_ENABLED=false`: cua's PostHog init makes a network
call that, at concurrency, blocks the server from binding its port past the provider's
120 s ready timeout (every acquire then fails). Off → cua comes up in seconds at any
fan-out. A benchmark sandbox shouldn't phone home anyway.

## Files

- `build.sh` — base build orchestrator (VM lifecycle, export→import→finalize→smoke).
- `export_rootfs.sh` — VM-side: zstd tar of `/` minus container-irrelevant paths.
- `remap_uids.py` — fallback uid/gid clamp for rootless `docker import`.
- `bake_nested_images.sh` — populate `/var/lib/dind` with the nested images + commit.
- `vnc_startup.sh` — baked self-adapting entrypoint (DinD + Xvfb + cua-server).
- `cleanup.sh` — in-container: strip VM identity/credentials, verify promised paths.

## Notes

- **Rootless host docker** is assumed (the uid remap + the privileged/userns reasoning
  rely on it). Under rootful docker the remap is a no-op; `--privileged` then *is* real
  host root — only run untrusted agents rootless.
- A container is not a VM: task coverage is only what's validated for this image.
