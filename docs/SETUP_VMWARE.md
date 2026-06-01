# Setup — VMWare (TODO)

> **Status: not yet implemented.** No `vmware` provider exists in
> [`ale_run/environments/providers/`](../ale_run/environments/providers/).
> The framework currently ships `gcloud` and `static` only. This page
> describes the intended shape and the interim workaround.

VMWare support is intended for small-scale runs and single-task
debugging on your own hardware — for example, iterating on a task
author's `start()` / `evaluate()` hooks against a local snapshot
without paying GCP boot time on every run.

---

## Intended shape

A `VmwareProvider` (TODO) would implement the `Provider` ABC from
[`ale_run/base_interface/sandbox.py`](../ale_run/base_interface/sandbox.py)
with three methods:

- `acquire(spec) → SandboxHandle` — clone from a base snapshot via
  `vmrun clone`, power on, wait for `cua-server` on TCP 5000.
- `release(sandbox, mode)` — `vmrun stop` + `vmrun deleteVM`, or revert
  the snapshot for fast reuse.
- `open_session(sandbox)` — wrap `cb.RemoteDesktopSession` at the VM's
  IP, same as the other providers.

Expected yaml shape (TODO). An experiment points `environment:` at a
single env config under `configs/environments/`; the provider and its
knobs live inside that file:

```yaml
# experiment.yaml
environment: configs/environments/vmware_default.yaml   # not yet shipped
```

```yaml
# configs/environments/vmware_default.yaml  (not yet shipped)
provider: vmware
vmrun_path: /usr/bin/vmrun                              # or vmware-workstation
base_snapshot: ~/vmware/ale-unified-v1.vmx
network: nat
```

---

## Interim workaround — use the `static` provider

If you bring up a VMware VM by hand today, you can drive it through the
`static` provider:

1. **Build a VM** from the published Windows image. Easiest path: export
   `ale-unified-v1` from GCP, convert to VMDK with `qemu-img`,
   import into VMware. (Detailed export recipe is TODO.)
2. **Boot the VM** and confirm `cua-server` is listening on TCP 5000:
   ```bash
   curl http://<vm-ip>:5000/status
   # → {"status":"ok", ...}
   ```
3. **Point the experiment at it** with the `static` provider. Copy an
   existing static env config (e.g.
   [`configs/environments/static_win10.yaml`](../configs/environments/static_win10.yaml))
   and set its `endpoint:`/`image:` for your VM:
   ```yaml
   # configs/environments/static_myvmware.yaml
   provider: static
   endpoint: http://192.168.1.42:5000
   image: ale-win10                      # or ale-ubuntu22
   vm_id: my-vmware-box
   ```
   ```yaml
   # experiment.yaml
   environment: configs/environments/static_myvmware.yaml
   ```
4. **Run the experiment normally.** Because the VM persists across runs,
   set `cleanup_on_release: true` plus a `cleanup_script` if you need
   between-run scrub.

See [`ale_run/environments/providers/static.py`](../ale_run/environments/providers/static.py)
for the full set of config keys.

---

## TODO

- [ ] First-class `VmwareProvider` (clone-from-snapshot lifecycle).
- [ ] Bake an importable VMware image (OVA/OVF) from
      `ale-unified-v1` and publish a download link.
- [ ] Convert this doc from "interim workaround" to "5-minute setup"
      once the above two land.
- [ ] Verify GPU passthrough story for the `gpu-*` task snapshots.

Contributions welcome — file an issue first to align on the provider
shape.
