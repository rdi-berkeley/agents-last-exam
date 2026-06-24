# ALE QEMU runner image

`agentslastexam/ale-qemu` is the container-side runtime used by the ALE `qemu`
provider. It packages QEMU, KVM integration, NAT networking, noVNC, and process
supervision. The Ubuntu or Windows guest is supplied separately as
`/storage/data.qcow2`.

Docker is the container runtime. Dockur is the upstream QEMU-in-Docker project
whose startup and networking stack this image inherits. ALE adds a stable
runner contract around that upstream image.

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
  -t agentslastexam/ale-qemu:0.2.0 \
  -t agentslastexam/ale-qemu:latest \
  .
```

## Publish

```bash
docker login
docker push agentslastexam/ale-qemu:0.2.0
docker push agentslastexam/ale-qemu:latest
```

## Runtime contract

- `/storage/data.qcow2` must be a non-empty pre-baked guest disk.
- The disk and every backing file must be readable when the container starts.
- `/dev/kvm` must be passed through.
- `NET_ADMIN` is required for the guest bridge and NAT rules.
- `/shared` may be bind-mounted to expose a per-run host exchange directory
  through Dockur's guest-only Samba share.
- Container ports `5000` and `8006` expose CUA and noVNC.
- `VM_NET_IP` defaults to `172.30.0.2`.
- Docker health becomes healthy when the guest CUA `/status` endpoint responds.

The runner never downloads guest disks. Docker bind mounts are fixed when a
container is created, so the host-side provider resolves and caches the qcow2,
creates the per-run overlay, and only then invokes `docker run`.

## Guest disk dataset

The pre-baked guest disks are published in the Hugging Face dataset
`agents-last-exam/ale-images-qcow2`.

| File | Guest |
|---|---|
| `ale-win10.qcow2.manifest.json` | Windows 10 qcow2 manifest |
| `ale-win10.qcow2.parts/*` | Verified 10 GB disk parts |
| `ale-ubuntu22.qcow2.manifest.json` | Ubuntu 22.04 qcow2 manifest |
| `ale-ubuntu22.qcow2.parts/*` | Verified 10 GB disk parts |

Configure the provider with the logical qcow2 path:

```yaml
snapshots:
  cpu-free:
    qemu:
      disk_source: hf://agents-last-exam/ale-images-qcow2/ale-win10.qcow2
  cpu-free-ubuntu:
    qemu:
      disk_source: hf://agents-last-exam/ale-images-qcow2/ale-ubuntu22.qcow2
```

The provider automatically discovers the multipart manifest, downloads and
verifies each part, reconstructs the selected qcow2 in the host cache, verifies
the complete disk SHA-256, and removes the temporary part files.

## Ubuntu disk maintenance

The Ubuntu qcow2 is derived from the GCE image, but its QEMU source image is a
separate artifact. Before creating that source image, mount the guest root
filesystem offline. Generate the validation manifests from the current task
allowlist and canonical GCS inventory:

```bash
gcloud storage ls \
  --recursive \
  --billing-project PROJECT \
  gs://ale-data-public \
  > /tmp/ale-data-public-list.txt

python scripts/prepare_qemu_ubuntu22_image.py manifests \
  --task-list selected_tasks/qemu_support.txt \
  --gcs-listing /tmp/ale-data-public-list.txt \
  --extra-variant demo/hello/base \
  --extra-variant demo/seecheck/base \
  --extra-variant demo/tool_smoke/base \
  --output-dir /tmp/ale-ubuntu22-manifests
```

Then clean and validate the offline filesystem:

```bash
python scripts/prepare_qemu_ubuntu22_image.py clean \
  --root /mnt/ale-ubuntu22 \
  --expected-variants /tmp/ale-ubuntu22-manifests/expected-ubuntu-all-variants.txt \
  --hg002-reference /tmp/hg002-reference \
  --disable-gce-services

python scripts/prepare_qemu_ubuntu22_image.py validate \
  --root /mnt/ale-ubuntu22 \
  --expected-variants /tmp/ale-ubuntu22-manifests/expected-ubuntu-all-variants.txt \
  --expected-visible-files /tmp/ale-ubuntu22-manifests/expected-ubuntu-visible-files.txt \
  --expected-reference-variants /tmp/ale-ubuntu22-manifests/expected-reference-variants.txt \
  --expected-reference-files /tmp/ale-ubuntu22-manifests/expected-ubuntu-reference-files.txt \
  --verify-reference-archives \
  --expect-gce-services-disabled
```

Run the cleanup without `--disable-gce-services` on the canonical GCE source
disk. Clone that cleaned disk for QEMU export, then run the commands above on
the clone. Never mask the GCE services in the canonical GCE image.

The cleanup removes task outputs, plaintext top-level references, stale agent
state, credentials, histories, system logs, GCE runtime state, and old machine
identity. It restores the HG002 input reference directory from the canonical
task-data source, regenerates SSH host keys, and masks GCE-only services in the
QEMU derivative.

After offline validation and a successful boot test, label the versioned GCE
image:

```bash
gcloud compute images add-labels IMAGE \
  --project PROJECT \
  --labels=ale-image-role=qemu-guest,ale-validation=passed
```

Export only to a versioned object. The export script verifies the labels and
refuses to overwrite either an existing object or the canonical path:

```bash
scripts/export_qemu_gce_image.sh \
  IMAGE \
  gs://BUCKET/images/ale-ubuntu22-YYYYMMDD.qcow2 \
  PROJECT \
  gs://NON_REQUESTER_PAYS_STAGING_BUCKET
```

Run `qemu-img check` and a local QEMU cold-boot task against the downloaded
versioned object before copying it to the canonical GCS path or publishing it
to Hugging Face.
