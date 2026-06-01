# Adding your own environment

An environment in ALE is anything that can produce a live `cua-server`
endpoint the framework can drive. A new environment provider needs two
things:

1. A `Provider` subclass that knows how to acquire and release sandboxes.
2. (Optional) An image-family registration that declares the VM-side
   path conventions.

If your target environment can boot the existing
[`ale-unified-v1`](https://console.cloud.google.com/compute/imagesDetail/projects/agenthle-488519/global/images/ale-unified-v1)
image (or an equivalent), step 2 is free.

---

## The `Provider` contract

Defined in [`ale_run/base_interface/sandbox.py`](../ale_run/base_interface/sandbox.py):

```python
class Provider(abc.ABC):
    @abc.abstractmethod
    async def acquire(self, spec: SandboxSpec) -> SandboxHandle: ...

    @abc.abstractmethod
    async def release(
        self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete",
    ) -> None: ...

    @abc.abstractmethod
    def open_session(self, sandbox: SandboxHandle) -> Any:
        """Return a cua-bench DesktopSession talking to `sandbox`."""

    async def heartbeat(self, sandbox: SandboxHandle) -> None: ...
    async def cancel_external(self, sandbox: SandboxHandle) -> None: ...
```

### `acquire(spec)`

Bring up a sandbox matching `spec` and return a populated
`SandboxHandle`. `spec` carries:

| Field | Meaning |
|---|---|
| `snapshot` | Logical image tag — `cpu-free-ubuntu`, `cpu-free`, `gpu-license`, ... |
| `os` | `"linux"` or `"windows"` |
| `vcpus` / `memory_gb` / `disk_gb` | Sizing hints |
| `gpu` | GPU accelerator name or `None` |
| `task_id` / `harness` / `model_tag` | Tags for logging / labels |

The returned `SandboxHandle` must populate:

- `id`, `endpoint`, `os`
- `work_dir_base`, `task_data_root`, `node`, `python`, `mcp_server_dir`
  — the VM-side paths the framework reads. Use `images.get(snapshot)`
  + `image.sandbox_paths()` to inherit defaults.

### `release(sandbox, mode)`

`mode` ∈ `{"delete", "stop", "keep"}`. Default is `"delete"`. Honor it —
tests rely on `"keep"` to preserve a VM for inspection.

### `open_session(sandbox)`

Return a `cb.RemoteDesktopSession` wired to `sandbox.endpoint`. The
framework calls this once per run and threads it through to
`task.start()` / `task.evaluate()`. See how `GcloudProvider` and
`StaticProvider` do it — both wrap the session with
`_init_computer_skip_wait()` to avoid duplicate readiness probes.

### Optional: `heartbeat` and `cancel_external`

`heartbeat()` is called periodically while a run is in flight — use it
to defer instance termination or refresh leases. `cancel_external()` is
called when the framework receives SIGINT mid-run — use it to terminate
in-flight cloud operations rather than waiting on their natural
timeouts.

---

## Reference implementations

### Full lifecycle — `GcloudProvider`

[`ale_run/environments/providers/gcloud.py`](../ale_run/environments/providers/gcloud.py).

- Creates a fresh GCE VM via `gcloud compute instances create`.
- Walks the capacity-pool fallback ladder
  ([`configs/environments/gcloud_ubuntu.yaml`](../configs/environments/gcloud_ubuntu.yaml))
  on capacity errors.
- Waits for `cua-server` to answer `/status` on TCP 5000.
- `release(mode="delete")` runs `gcloud compute instances delete`.

Copy this pattern for any cloud provider with a comparable API
(AWS EC2, Azure VM, Oracle OCI, etc.).

### Wrap-an-endpoint — `StaticProvider`

[`ale_run/environments/providers/static.py`](../ale_run/environments/providers/static.py).

- No VM lifecycle. `acquire` just wraps the configured `endpoint`.
- Optional `cleanup_script` runs on `release` to scrub state.

Copy this pattern for "bring-your-own-VM" providers (VMware, Proxmox,
bare-metal, dev laptops with cua-server running locally).

---

## Registering a new provider

Add an entry to the provider factory in
[`ale_run/orchestration/factory.py`](../ale_run/orchestration/factory.py)
(the `build_provider` dispatch). Then ship a config file for it under
`configs/environments/`. The file holds `provider:` PLUS that provider's
knobs:

```yaml
# configs/environments/my_provider_default.yaml
provider: my_provider
region: us-west2
...
```

An experiment wires it in by path (one env config per experiment):

```yaml
environment: configs/environments/my_provider_default.yaml
```

The env config lets users share defaults across experiments — see
[`configs/environments/gcloud_ubuntu.yaml`](../configs/environments/gcloud_ubuntu.yaml)
for the shape (`provider:`, a `snapshot:`, and the per-snapshot image map
+ capacity pools under `snapshots:`).

---

## Image families

Image conventions are registered in
[`ale_run/environments/images/__init__.py`](../ale_run/environments/images/__init__.py).
Each entry maps an image-family name (`"ale-ubuntu22"`, `"ale-win10"`)
to a struct declaring:

- `os` — `"linux"` or `"windows"`
- `work_dir_base` — per-run scratch root (e.g. `/home/user/.ale`)
- `task_data_root` — staged-data root (e.g. `/media/user/data/ale-data`)
- `node` — absolute path to the `node` binary (for MCP-server agents)
- `python` — absolute path to the Python interpreter
- `mcp_server_dir` — where the cua MCP server is installed

If your image bakes things at the same paths, you can reuse an existing
family. Otherwise add a new family and reference it from your provider's
`acquire()`.

### For GCP users — swap the image, not the code

If you're staying on GCP but want a different OS, kernel, or pre-installed
toolset, you don't need a new provider. Just edit the `snapshots:` block of
a gcloud env config such as
[`configs/environments/gcloud_ubuntu.yaml`](../configs/environments/gcloud_ubuntu.yaml):

```yaml
snapshots:
  cpu-free-ubuntu:
    image: my-custom-ubuntu-image        # ← only this line changes
    zones:
    - us-central1-a
```

Tasks with `vm.snapshot: cpu-free-ubuntu` will boot your image instead.

### For Docker (TODO)

Docker support is a stub today ([`ale_run/executors/docker.py`](../ale_run/executors/docker.py)).
When implemented, a Docker "provider" would be unusual — instead, the
executor itself runs the agent in a host-side container, with the
sandbox endpoint still living wherever the provider put it (cloud, VM,
host). PRs welcome to flesh this out.

---

## Testing your provider

1. Implement a minimal `acquire` / `release` that returns a working
   `SandboxHandle`.
2. Smoke-test with the in-repo demo task on a hello-world experiment.
   Point `environment:` at your new env config (which carries
   `provider: my_provider`) and pick a simple agent preset:
   ```yaml
   environment: configs/environments/my_provider_default.yaml
   tasks: selected_tasks/helloworld.txt
   agents:
     - configs/agents/openclaw_sonnet_or.yaml   # host-side agent — easy to debug
   ```
3. Verify the run completes through Phase 0 → 6 (see
   [`ale_run/orchestration/lifecycle.py`](../ale_run/orchestration/lifecycle.py)).
4. Add an integration test under `tests/integration/` and document the
   credentials it needs in the test docstring.

---

## Contributing

ALE is designed to grow horizontally — new clouds and on-prem
hypervisors are first-class extensions, not patches. Open an issue
describing the target environment before sending a PR so we can align
on the provider shape and any new image families it needs.
