# ALE QEMU runner image

`agentslastexam/ale-qemu` is the container-side runtime used by the ALE `qemu`
provider. It packages QEMU, KVM integration, NAT networking, noVNC, and process
supervision. The Ubuntu or Windows guest itself is supplied separately as
`/storage/data.qcow2`.

The image is based on a digest-pinned `trycua/cua-qemu-windows` release. ALE
replaces its inherited entrypoint because that script can remain alive forever
after QEMU exits. The ALE entrypoint validates the mounted disk and executes
the upstream VM process under `tini`, so Docker observes VM failures and
signals correctly.

## Build

From the repository root:

```bash
docker build \
  -f ale_run/environments/images/ale_qemu/Dockerfile \
  -t agentslastexam/ale-qemu:0.1.0 \
  -t agentslastexam/ale-qemu:latest \
  .
```

## Publish

```bash
docker login
docker push agentslastexam/ale-qemu:0.1.0
docker push agentslastexam/ale-qemu:latest
```

## Runtime contract

- `/storage/data.qcow2` must be a non-empty pre-baked guest disk.
- `/dev/kvm` must be passed through.
- `NET_ADMIN` is required for the guest bridge and NAT rules.
- Container ports `5000` and `8006` expose CUA and noVNC.
- `VM_NET_IP` defaults to `172.30.0.2`.
- Docker health becomes healthy when the guest CUA `/status` endpoint responds.
