# ale-ubuntu22-docker image build

The container form of the `ale-ubuntu22` Linux sandbox, so the `cpu-free-ubuntu`
(no-GPU, no-license) tasks run under the **docker provider** on one host instead
of one GCE VM each. The **data-less** v1.1 release target is
`agentslastexam/ale-ubuntu22-docker:v1.1`. Pair it with the pinned v1.1 archive,
not an old `task-data` directory. Publication status and exact pins are in
`releases/v1.1/assets.json` and `docs/releases/v1.1.md`.

The v1.1 Docker Hub upload is currently blocked by missing push permission.
Do not treat the prepared local image as a published registry release.

To *run* it you need none of this — pull the image and fetch the data
(`scripts/fetch_task_data.sh`); see the **Local Docker** docs page. This
directory is how the maintainers rebuild the image.

## Local QCOW2 build

`build.sh` requires an explicit, sanitized, sealed **standalone local QCOW2**.
There is no GCloud prerequisite, default dev VM, default output tag, push, or
image pruning. The historical cloud-only orchestration is replaced, while the
original userspace export, Docker import, cleanup, commit and desktop entrypoint
approach remains. This is not a new Ubuntu Dockerfile or a scientific-runtime
reinstallation. Task data is staged separately at runtime by the `local:` source.

Requirements: the repository's uv environment, Linux with `/dev/kvm` available
to the existing Docker daemon, `qemu-img` for source inspection, OpenSSH,
GNU tar/findmnt/df and zstd, and a local Linux Docker daemon using a Unix socket.
Default `--builder-backend provider` uses the existing `QemuProvider` and its
already installed `agentslastexam/ale-qemu:0.2.0` runner. It pins the local image
ID, sets pull policy `never`, and binds the explicit QCOW2 read-only. It needs
neither native host QEMU nor user membership in the KVM group. The provider
uses its matching UEFI firmware and a fresh variable store in the owned slot.
The guest needs Python 3, GNU tar, zstd,
OpenSSH server, systemd and a working CUA command endpoint with passwordless
sudo. No SSH identity or `.ssh` directory is required in the sealed source.
Once CUA on the disposable clone is ready, the builder generates a unique
Ed25519 identity in the persistent build directory and sends only its public
key through the existing CUA `/cmd` protocol. The guest installs that key with
restricted forwarding, generates any missing SSH host keys and starts sshd.
The host pins the returned host public key before using strict SSH validation.
SSH uses `ProxyCommand=docker exec -i <owned-container> nc 172.30.0.2 22` and a
unique HostKeyAlias; there is no additional published SSH port. Provider CUA
and noVNC ports remain loopback-bound. No GCloud credentials are configured.
Alternatively, `--ssh-key /path/to/private-key` derives and provisions the
matching public key of a supplied unencrypted identity. Private key bytes
never enter CUA, the guest, the archive, or logs. Guest `.ssh` and SSH host keys
are excluded before import. Neither mode changes the source disk. Actual
CUA/sshd/sudo and QEMU BIOS/UEFI compatibility remain integration prerequisites.

The runtime manifest is an audited JSON contract. Copy the **structure**, not
the illustrative paths/hashes, from `runtime_manifest.example.json`. Bind its
`source_sha256`, `release`, and `data_manifest_sha256` to the actual sealed disk
and data manifest. `required_paths` must include the installed scientific
environment roots, R libraries, Python environments and model assets. `probes`
are trusted argv arrays executed in the guest and as `user` in the finalized
container; include actual package imports, not just version commands. Both
Python and R probes are required. The manifest is code-equivalent trusted input.
Optional `source_probes` use the same absolute-argv format, execute first on the
disposable source clone, and fail before archive allocation on any error. Use
them for source task-data symlink checks that cannot run in a data-less container.
Required interpreter targets still belong in `required_paths` and ordinary
`probes` so their retention and container startup are independently checked.

From the authoritative repository:

```bash
bash ale_run/environments/images/ale_ubuntu22_docker/build.sh \
  --source-qcow2 "$SEALED_QCOW2" --source-sha256 "$SEALED_SHA256" \
  --release v1.1 --data-manifest "$DATA_MANIFEST" \
  --runtime-manifest "$RUNTIME_MANIFEST" \
  --workdir "$HOME/ale-overall/docker-build-v1.1" \
  --image agentslastexam/ale-ubuntu22-docker:candidate-v1.1-local-001 \
  --ssh-user user --export-only \
  --preflight-only
```

Export preflight hashes the source and writes a receipt but creates no disk,
VM or image. Only after the source is sanitized and allocation is authorized,
run the same invocation without `--preflight-only`. It creates and audits the
compressed export, records compressed/logical sizes and an estimated Docker
budget, stops the owned clone through `QemuProvider.release(mode="stop")`, and
returns WITHOUT importing any image. Logs and the stopped slot remain for audit.

Then replace `--export-only` with `--finalize-only --preflight-only` in the same
invocation. This requires the verified source/script-bound export, audits it
and checks current Docker capacity without starting a VM or importing an image.
Only after separate authorization remove `--preflight-only` to finalize. The
default `--stage all` performs both phases but still rechecks measured capacity
between them. It never reserves both phases' full budgets at once.

The optional `--builder-backend native` retains direct host-QEMU operation.
That explicit mode needs host QEMU and KVM access. For UEFI guests also provide
`--uefi-code /path/to/OVMF_CODE.fd --uefi-vars /path/to/OVMF_VARS.fd` with matching
firmware files. A disposable variables copy is made alongside the overlay;
neither the source QCOW2 nor the firmware template is modified. UEFI uses
`q35,smm=off`; inspect the source runner's actual code/template and boot mode
before choosing firmware. Without these options the generic builder uses BIOS;
do not omit them for a verified UEFI source. Use a fresh `candidate-*` tag;
existing tags are rejected. Promotion
to a release tag is a separate, explicitly approved operation.

### Storage and lifecycle

The work directory must resolve beneath `$HOME/ale-overall`, or the explicitly
configured `ALE_IMAGE_BUILD_ROOT`, on a verified persistent filesystem.
`findmnt -T` and `df` also check the Docker
data-root. Unknown, overlay, network, memory-backed and read-only filesystems
are rejected rather than assumed safe. Same-device budgets are added together.
Default export-phase budgets are 16 GiB overlay, 120 GiB export, 1 GiB for the
provider runner's small writable layer, plus 10 GiB free disk headroom. They
do not include a Docker import/commit reservation. Set `--overlay-budget-gib`,
`--export-budget-gib`, and `--docker-budget-gib` based on the measured source;
these are capacity reservations, not filesystem quotas. Finalization's default
additional Docker budget is twice the inspected uncompressed rootfs, rounded
up to GiB, plus 20 GiB adaptation/transient allowance. An explicit Docker budget
must cover at least twice the inspected rootfs. Finalization reserves no second
archive or new VM disk: existing archives and stopped overlays already reduce
measured free space. Existing archives,
images and retained overlays already consume free space and are not deleted.

RAM checks include live QEMU processes identified by process name or executable
argv, including runners using `-name ...,process=windows`, running and paused
retained guests, their unresident configured memory, the builder/container
limit, compressor overhead and `--headroom-gib` (default 16, minimum 8).
Stopped/exited audit containers have no live QEMU process and reserve no guest
RAM. A paused or SIGSTOP QEMU process still holds RAM; it is not equivalent to
a stopped container. Exited/zombie processes are skipped. Unknown live VM
memory shapes fail closed. Capacity is rechecked before VM creation, export
and import; there is at most one pending builder for this work directory.
Coordinate concurrent allocations with the parent operator: a preflight is not
a host-wide reservation or an OOM recovery clearance.

Each export boots one disposable overlay with loopback-only CUA; provider-mode
SSH is tunneled through its owned runner. Native mode uses restricted user
networking and loopback SSH forwarding. It never boots the sealed disk writable, starts a
solver, mounts a host data tree, or downloads an image. Guest scripts and the
runtime contract are small files under guest `/tmp`; the file list is on the
guest disk under `/var/lib`. The streamed archive and VM files are on persistent
host storage, never host `/tmp` or `/dev/shm`. Only that owned builder and this
build's containers are stopped on errors/signals. The provider handles acquire
failure cleanup; after acquisition the adapter preserves runner logs and calls
the provider's stop mode, never touching parent sandboxes. Overlay, serial/export logs
and failed artifacts remain for review; no backing images or Docker images are
automatically removed. A hard host kill still requires operator reconciliation.
The generated host private key remains in the private build directory for
diagnostics, not in any image. Remove it only after the associated builder is
stopped and required audit evidence has been preserved.

### Export, cache and runtime gates

`rootfs_policy.py` prunes task data, known injected keys, cloud authentication,
SSH keys, solver/harness sessions, histories, browser profiles, unrelated homes,
Docker stores and build state **before the first Docker layer**. `exclude_paths`
adds source-audit-specific exclusions. A NUL-delimited file list and tar
`--no-recursion` prevent directories from reintroducing excluded descendants.
Oversized owners are normalized only on the disposable clone. Every tar,
compression and SSH failure is fatal; partially written exports are not cached.

There is no blanket filesystem-boundary filter. Required paths and their
symlink targets must survive. Audited runtime assets under uv or Hugging Face
caches can be retained through narrowly scoped `preserve_cache_paths`; secrets
cannot be allowed back through that list. Missing/filtered targets fail before
export/import. Cleanup does not delete those preserved caches. Inventory all
required Python/R roots so recursive link checks cover their packages too.

Some installed runtimes live under the otherwise excluded data volume.
`preserve_runtime_paths` permits only explicitly audited individual roots at
`/media/user/data/opt/<name>` or `/media/user/data/toolchains/<name>`. Each must
also be in `required_paths`; missing targets still fail. Only those roots and
their ancestor directories are retained, not sibling tools, installers, or
`/media/user/data/agenthle`. Credential/state exclusions and explicit
`exclude_paths` still take precedence. Audit actual contents before approval;
this exception is not permission to keep task inputs or references. Cleanup
does not recursively change ownership of these retained runtimes.

`runtime_audit.py CONTRACT --max-seconds 180 --max-entries 2000000` inspects
metadata without owner normalization, archive creation, or guest writes. It
reports retained-size upper bounds (hard links counted repeatedly), required
targets, bounded findings and a manifest digest only for completed scans.
An incomplete scan or any unresolved finding exits nonzero; partial byte
totals are not whole-image capacity estimates. Run against an authorized clone,
not a mounted image that might be modified concurrently. A read-only inventory
of an unsealed builder is provisional, not a source-content approval.

The cache receipt binds the exact source hash/path, release, data/runtime
manifest hashes, script contents, package inventory and archive hash/size.
Provider exports also bind the runner's immutable image ID and provider code hash.
Mismatched, incomplete or damaged caches fail, rather than silently re-exporting
over evidence. Use a new work directory after correcting a source or script.
Compressed integrity, tar termination, forbidden members, owners and required
runtime links are checked again before every import, including cache hits.

Cleanup retains the existing XFCE/Xvfb/D-Bus/Chrome adaptations. Missing desktop
packages or required runtimes fail the build. Source package removals/upgrades
are rejected; newly installed desktop dependencies are recorded for review.
The smoke requires two HTTP 200 JSON `status: ok` responses plus an X socket,
window manager, panel and display probe. Containers are bounded, nonprivileged,
with DinD disabled; smoke success is not release acceptance.

### Remaining release gates

The builder emits a candidate image ID and receipts, **not** a release-ready
claim. Before publication, audit actual source content and every final layer
(including metadata) for secrets/data, review desktop package additions, check
scientific inventories and execute official rootless-provider GUI/tool/agent
smokes, cold starts, data/reference staging and evaluator fixtures. Complete all
99 supported task runs and their same-release VM comparisons as specified in
the release validation plan. A path policy cannot identify arbitrary secrets
or task outputs deliberately stored under unexpected runtime filenames. A
later deletion cannot repair sensitive bytes already imported into a layer.

Unit tests use small fixtures and simulated QEMU/Docker/SSH interfaces:

```bash
uv run --no-sync pytest tests/ale_run/test_docker_image_build.py -q
```

## Scripts

| script | runs on | does |
|--------|---------|------|
| `build.sh`, `local_build.py` | your host | explicit local-source orchestration, capacity/cache/archive checks and candidate receipts |
| `bootstrap_ssh.py` | disposable VM, through CUA | installs only the export public key and returns the guest SSH host public key |
| `export_rootfs.sh`, `rootfs_policy.py` | disposable VM / host auditor | filtered userspace export, runtime-link checks and oversized-owner normalization |
| `cleanup.sh` | container, at build | bakes the entrypoint, scrubs VM identity/credentials, makes the empty `/media/user/data/agenthle` task directory |
| `entrypoint.sh` | container, at runtime | `Xvfb :0` + cua-server on `:5000` (+ optional DinD) |
| `bake_nested_images.sh` | container, at build (optional) | pre-populates `/var/lib/dind` with the DinD task images |

## Optional DinD hooks

A few task evals run `docker` **inside** the sandbox (openroad, k8s_migration,
bpmn ×2). In a container that's Docker-in-Docker: the nested daemon needs
`--privileged` and can't stack overlay2 on overlay2, so it runs on
`fuse-overlayfs` at `/var/lib/dind`, with its images baked in
(`bake_nested_images.sh`) so a pulled image needs no per-start load.

The supported local-container profile intentionally remains a simple rootless
Docker path, so tasks that require Docker, Apptainer, or Singularity inside the
sandbox are excluded from `docker_support.txt`. Run those tasks with the
QEMU/KVM provider, where the nested runtime operates inside a complete Ubuntu
guest.

The DinD path remains available for custom development through
`ALE_ENABLE_DIND=1`, `enable_dind: true`, and `privileged: true`. It is not part
of the supported default task list. At high concurrency, its fuse-overlayfs
daemon can also add significant I/O and delay cua-server readiness.
