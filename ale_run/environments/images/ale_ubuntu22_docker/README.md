# ale-ubuntu22-docker — image build

Builds the `ale-ubuntu22-docker` container image from the `ale-ubuntu22` GCE
sandbox VM by **exporting the VM's root filesystem** (no package rebuild). A
container shares the host kernel, so only the userspace rootfs is needed; the
VM's kernel, bootloader, init, logs and desktop bits are dropped.

The `Image` entry for this family lives in `../ale_ubuntu22_docker.py`; this
directory holds only the build tooling (mirroring `../ale_kasm/`).

## Run

```bash
./build.sh
```

Defaults target the `dev-ubuntu22` box in `us-west2-a`; override via env:

| env | default | meaning |
|-----|---------|---------|
| `ALE_BUILD_VM` / `ALE_BUILD_ZONE` | `dev-ubuntu22` / `us-west2-a` | source VM |
| `ALE_BUILD_SSH_USER` / `ALE_BUILD_SSH_KEY` | `weichenzhang` / `~/.ssh/google_compute_engine` | SSH identity |
| `ALE_BUILD_IMAGE` | `ale-ubuntu22-docker:latest` | output tag |
| `ALE_BUILD_WORKDIR` | `~/.cache/ale-docker-build` | scratch for the rootfs tar (~100 GB free) |
| `ALE_FORCE_EXPORT=1` | — | re-export even if the tar exists |
| `ALE_KEEP_VM=1` | — | don't stop the VM afterwards |

`build.sh` starts the VM if stopped (and stops it again on exit), then runs four
phases — **export → import → finalize → smoke**. It is resumable: a present
rootfs tar is reused, so a failed import/finalize can be retried without the
~1 h re-export. End to end is ~2–3 h (the uid-remap pass over ~270 GB dominates)
and produces a ~275 GB image.

## Files

- `build.sh` — orchestrator (VM lifecycle, phases, smoke).
- `export_rootfs.sh` — runs on the VM via SSH; streams a zstd tar of `/` minus
  container-irrelevant paths; records tar's real exit code.
- `remap_uids.py` — streaming tar rewrite clamping out-of-range uids/gids to 0
  so rootless `docker import` succeeds (preserves uid 1000 = `user`).
- `vnc_startup.sh` — baked entrypoint: brings up `Xvfb :0`, then cua-server on
  `:5000` (the docker provider invokes `/dockerstartup/vnc_startup.sh`).
- `cleanup.sh` — in-container: bakes the entrypoint, fixes `/tmp` perms, strips
  VM-host identity and baked credentials, verifies promised paths.

## Notes

- **Rootless Docker** is assumed (the uid remap exists for it). Under rootful
  Docker the remap is a no-op and harmless.
- The image is **not** shipped in-repo (275 GB); it is built locally per host.
- The build does not rebuild packages, so the container's task coverage is only
  as validated as the [Local Docker](../../../../docs/ale-docs-site/pages/local-docker.html)
  page states — a container is not a VM.
