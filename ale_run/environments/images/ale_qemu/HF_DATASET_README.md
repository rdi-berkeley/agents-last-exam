---
pretty_name: ALE QEMU guest images
---

# ALE QEMU guest images

This dataset distributes the pre-baked guest disks used by the
`agents-last-exam` local `qemu` provider.

| File | Guest |
|---|---|
| `ale-win10.qcow2.manifest.json` | Windows 10 qcow2 manifest |
| `ale-win10.qcow2.parts/*` | Verified 10 GB disk parts |

The Ubuntu guest image is being revised and is not published in this dataset
revision.

The guest disks do not contain the QEMU runtime. The provider downloads a disk
to the host cache, creates a disposable qcow2 overlay for each task run, and
boots that overlay with `agentslastexam/ale-qemu`.

Example source:

```yaml
qemu:
  disk_source: hf://agents-last-exam/ale-images-qcow2/ale-win10.qcow2.manifest.json
```

The provider downloads each part, verifies its SHA-256, reconstructs
`ale-win10.qcow2` in the host cache, verifies the complete disk SHA-256, and
then deletes the temporary part files.
