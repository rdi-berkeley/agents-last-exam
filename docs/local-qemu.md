# Local QEMU provider

The `qemu` provider runs a complete Ubuntu or Windows guest with QEMU inside a
Docker container. It provisions one VM for each ALE run and deletes it during
normal cleanup.

## Host requirements

- Linux with hardware virtualization or nested virtualization enabled
- Docker daemon available to the current user
- `/dev/kvm` present
- Sufficient RAM for the task-card machine shape
- Sufficient disk for the cached base images
- `gcloud` or `gsutil` when `qemu.qcow2` is a `gs://` URI

The public base images are currently:

- `gs://ale-data-public/images/ale-ubuntu22.qcow2`
- `gs://ale-data-public/images/ale-win10.qcow2`

Use `configs/environments/qemu.yaml` as the starting configuration.

## Runner image

The provider defaults to `agentslastexam/ale-qemu:0.1.0`. Its complete build
definition is in `ale_run/environments/images/ale_qemu/`. The runner image
contains QEMU, networking, noVNC, and lifecycle supervision, but not either
guest disk.

Build and publish it from the repository root:

```bash
docker build \
  -f ale_run/environments/images/ale_qemu/Dockerfile \
  -t agentslastexam/ale-qemu:0.1.0 \
  -t agentslastexam/ale-qemu:latest \
  .
docker push agentslastexam/ale-qemu:0.1.0
docker push agentslastexam/ale-qemu:latest
```

## Storage model

The first run downloads each base qcow2 to
`~/.cache/ale/qemu/images/`. Every run then creates a small qcow2 overlay under
`~/.cache/ale/qemu/runtime/slots/`. The base image is mounted read-only into the
QEMU container, so concurrent runs do not modify it or copy its full contents.

## Initial limitations

- GPU tasks are not supported. PCIe passthrough requires IOMMU/VFIO host setup,
  a dedicated GPU, QEMU `vfio-pci` configuration, and matching guest drivers.
- `task_data_source: local:...` is not supported. The initial provider expects
  input, software, and encrypted reference data to be baked into the qcow2.
- The host does not enforce a global CPU or memory admission policy. Set ALE
  `concurrency` conservatively for the available machine resources.
