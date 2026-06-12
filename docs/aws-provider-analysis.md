# AWS Provider — Preliminary Feasibility Analysis

Branch `feat/aws-provider` (worktree off `main`, base 379a35f).

Goal: add **AWS (EC2 + S3)** as a first-class alternative to the existing
gcloud (GCE + GCS) backend, so a deployment can run ALE on AWS instead of GCP.

## TL;DR

The framework is already provider-agnostic where it matters. There are exactly
**three integration surfaces** in code and **two artifact-migration chores**.

| Surface | Effort | Notes |
|---|---|---|
| Provider class (`providers/aws.py`) | Medium | Mirror `gcloud.py`; `acquire`/`release`/`open_session`. The cua-server I/O, readiness poll, Windows-resolution and session code are all reusable as-is. |
| Data backend (`task_data/s3bucket.py` + `s3://` in `output_pull.py`) | Low | Mirror `gsbucket.py`; `aws s3 sync` instead of `gsutil rsync`. IAM instance profile removes the credential-injection dance entirely. |
| Config plumbing (`factory`, `config_loader`, `providers/__init__`) | Low | Add an `aws` branch alongside `gcloud`/`docker`. |
| **VM images → AMIs** | **High** | The Linux/Windows VM images are external GCE images; must be rebuilt or imported. Windows is the hard case. |
| **Task data → S3** | Low | One-time bucket copy; the 7z format is unchanged. |

## How the abstraction is layered (why this is tractable)

* **`Provider` ABC** (`base_interface/sandbox.py:223`) is tiny: `acquire(spec) ->
  SandboxHandle`, `release`, `open_session`. Everything else the framework does
  to a sandbox goes through **`SandboxHandle`'s HTTP I/O to the cua-server**
  (`/cmd`, `/status`) — completely cloud-agnostic. A new provider only has to
  stand up a box that runs cua-server on a reachable IP:port and hand back a
  handle.
* **`Image`** (`environments/images/__init__.py`) is pure metadata — name, os,
  baked paths (`work_dir_base`, `task_data_root`, `node`, `python`,
  `mcp_server_dir`), `cua_server_port`. It is **not** tied to GCE; the same
  registry entries describe an AMI just as well, because they only assert the
  filesystem layout *inside* the box. **No change to the Image abstraction.**
* **Provider selection is per-snapshot** (`factory.EnvironmentRouter`,
  `config_loader._build_environment`). `build_provider` dispatches on
  `spec.kind` (`gcloud`/`static`/`docker`) — add `"aws"`.
* **Task-data source is scheme-dispatched** (`task_data/__init__.py:select`):
  `baked_in_sandbox` / `gs://` / `hf://`. Add `s3://`. Output pull-back
  (`output_pull.py`) dispatches on `local` / `gs://` — add `s3://`.

## 1. Provider class — `environments/providers/aws.py`

Mirror `GcloudProvider`. The GCP→AWS mapping:

| gcloud | aws |
|---|---|
| `gcloud compute instances create --image --image-project` | `aws ec2 run-instances --image-id <ami>` |
| `--zone` + zone fallback list | `--availability-zone` (via subnet) + AZ/subnet fallback list |
| `--machine-type` + C→N2 fallback | `--instance-type` + a fallback chain (e.g. `c7i`→`m7i`) |
| `--boot-disk-type` (hyperdisk/pd-ssd by family) | `--block-device-mappings` EBS `gp3`/`io2` |
| `--network`/`--subnet` (must expose :5000) | `--subnet-id` + **security group** allowing tcp:5000 (and 3389 for Win) |
| `--accelerator type=…,count=1` (GPU) | GPU instance family (`g5`/`g6`/`g4dn`) — GPU is the instance type, not a flag |
| external IP from `networkInterfaces[].accessConfigs[].natIP` | public IP from `Reservations[].Instances[].PublicIpAddress` (or assign EIP) |
| poll `instances describe` for IP | poll `describe-instances` for `PublicIpAddress` + `running` |
| labels | tags (`Name`, `purpose=ale-run`, `snapshot=…`) |
| SA-key file (`service_account_key`) | IAM: instance profile / role + region + credentials chain |

**Reusable verbatim** from `gcloud.py`: `wait_cua_ready` / `_probe_cua` /
SSE parsing, `_set_windows_resolution` (`_SET_RES_PY`), `_init_computer_skip_wait`,
`open_session` (RemoteDesktopSession), the post-create delete-on-failure guard,
the machine×zone retry ordering (becomes instance-type×AZ).

**`AwsProviderConfig`** (parallels `GcloudProviderConfig`): `region`,
`subnets`/`availability_zones`, `security_group_id`, `key_name` (optional),
`iam_instance_profile`, `instance_prefix`, and `snapshots: {tag -> {image(AMI),
gpu, zones(AZs), resolution}}`. The `SnapshotConfig` shape carries straight over;
`image` now holds an **AMI id/name** instead of a GCE image name.

**Simplification vs GCP:** GCP injects an SA key into every VM because the baked
`gsutil` is anonymous. On EC2 you attach an **IAM instance profile** at launch
and the instance's `aws` CLI is authenticated with zero key-pushing. So
`_inject_gcs_credentials` has **no AWS counterpart** — drop it.

## 2. Data backend — `environments/task_data/s3bucket.py`

Mirror `gsbucket.py` 1:1:

* `stage_input`: skip if baked; else `aws s3 sync s3://<prefix>/{input,software} <dst>`.
* `stage_reference`: wipe + `aws s3 sync s3://<prefix>/reference <dst>`, then chmod.
* `_gcs_exists` → `aws s3 ls`; `_rsync_cmd` → `aws s3 sync`.
* No `-u <project>`/requester-pays and no `-o gs_service_key_file`: with an IAM
  instance profile the in-box CLI is already authed. Requester-pays has an S3
  analogue (`--request-payer requester`) only if you choose to enable it.
* Register `s3://` in `task_data/__init__.py:select` and in `output_pull.py`.

The on-disk data contract is identical: `<prefix>/<domain>/<task>/<variant>/{input,
software,reference}`, reference still a flat 7z. **Format unchanged.**

## 3. The two big questions

### Q1: Does the VM image need to be exported from GCE and re-imported to AWS?

The **abstraction** ports for free; the **bits** do not. The Linux/Windows
images (`ale-ubuntu22`, `ale-win10`) are external GCE images in project
`agenthle-488519` with **no bake source in this repo** (only `ale-kasm` has a
Dockerfile). Two routes:

* **(a) Lift-and-shift via VM Import/Export.** `gcloud compute images export`
  → raw disk tar.gz in GCS → copy to S3 → `aws ec2 import-image`. *Feasible for
  Linux but fiddly:* GCE images carry the google-guest-agent + virtio drivers;
  EC2 wants ENA/NVMe drivers and cloud-init. Expect driver/boot fixups. **Windows
  is the hard case** — cross-cloud Windows import drags BYOL licensing, sysprep,
  and driver swaps; AWS strongly prefers you *start from an AWS-provided Windows
  base AMI* (license-included) rather than importing a GCE Windows disk.
* **(b) Rebuild with Packer (recommended).** Since there's no bake source
  anyway, author Packer templates: start from a stock AWS base AMI (Ubuntu 22.04
  / Windows Server license-included), run the same provisioning (cua-server,
  node, python venv, 7z, **`aws` CLI in place of the gcloud SDK**, baked task
  data), tag the result. Reproducible, no import/driver/licensing pain, and it
  finally gives the project a checked-in image recipe. Windows resolution control
  already exists (the `_SET_RES_PY` PowerShell), so that carries over.
* **Docker families are already portable.** `ale-kasm` (Dockerfile) and
  `ale-ubuntu22-docker` (rootfs export) push to **ECR** and run via the existing
  `DockerProvider` on an EC2 Docker host — **zero provider work** if you only
  need Linux and are willing to run Docker on EC2 rather than raw instances.

**Recommendation:** start with **(b) Packer for Ubuntu** + **Docker/ECR for the
Linux-only fast path**; treat **Windows + GPU AMIs as a separate, later phase**
(they carry the licensing and driver risk).

### Q2: Can task data be exported and imported into AWS object storage (S3)?

**Yes, cleanly.** Data is either baked into the image (no action — it rides the
AMI) or pulled from a `gs://` bucket. To move the buckets:

```
# one-time, host-side
gcloud storage cp -r gs://ale-data-public/** s3://<your-bucket>/...   # via rclone or stream
# or rclone gcs:ale-data-public s3:<bucket>
```

The 7z-encrypted reference + flat input/software layout is byte-identical on S3;
only the `task_data_source` scheme in the env yaml changes (`gs://…` → `s3://…`),
which the new `s3bucket.py` backend consumes. Requester-pays is GCP-specific and
simply not needed on the S3 side.

## 4. Operator setup (AWS equivalent of quickstart §3e–3f)

* VPC + subnet(s) across the chosen AZs; **security group** ingress tcp:5000
  (cua) and tcp:3389 (Windows RDP), mirroring `ale-allow-cua`/`ale-allow-rdp`.
* **IAM role / instance profile** with `s3:GetObject`/`PutObject` on the data
  bucket (replaces the SA + key.json + `roles/storage.objectAdmin`).
* AMIs registered in the target region (built per Q1).
* An S3 bucket holding the migrated task data (Q2).

## 5. Suggested phasing

1. **Plumbing + data (low risk):** `s3bucket.py`, `s3://` in `select`/`output_pull`,
   `aws` branch in `factory`/`config_loader`/`providers.__init__`,
   `AwsProviderConfig`. Copy one domain's data to S3 to test staging.
2. **Linux EC2 provider:** `AwsProvider.acquire/release/open_session` against a
   Packer-built `ale-ubuntu22` AMI; reuse `wait_cua_ready`/session code. Validate
   demo/hello end-to-end.
3. **Windows + GPU (separate phase):** Windows base-AMI Packer build, license
   handling, GPU instance families + NVIDIA drivers, resolution control.

## Touch list (files)

* New: `ale_run/environments/providers/aws.py`,
  `ale_run/environments/task_data/s3bucket.py`,
  `configs/environments/environment_aws.yaml`, Packer template(s).
* Edit: `providers/__init__.py`, `orchestration/factory.py:build_provider`,
  `orchestration/config_loader.py` (`aws` branch + `_validate_provider_required`),
  `task_data/__init__.py:select`, `environments/output_pull.py`,
  `docs/quickstart.md` (AWS setup section).
