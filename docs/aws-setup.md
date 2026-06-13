# Running ALE on AWS (EC2 + S3)

AWS counterpart of `docs/quickstart.md` §3. Gets `ale-ubuntu22` running on EC2
from the GCE-exported disk. Windows/GPU is a later phase.

## Verified status (us-east-1, account 670060057810)

Both images imported and run on EC2 (cua-server reachable on :5000, commands
execute, baked data present):

- **Linux** `ale-ubuntu22` → AMI via **import-snapshot + register-image**
  (import-image rejects kernel 6.5.0-15). `AwsProvider` acquire→run→release
  acceptance test passes (`tools/aws/provider_accept_test.py`).
- **Windows** `ale-win10` → AMI via **import-image** (BYOL; passes AWS boot
  validation). Boots to desktop, cua up, E: data drive present. Launched on
  **shared tenancy for validation only** — license-compliant runs need
  **dedicated tenancy**, which is gated on a new-account quota increase (AWS
  Support case; the Dedicated-Instances vCPU limit is not API-adjustable).

## 0. Prerequisites (one-time, needs a human)

- An **AWS account** (registration needs a payment method + verification —
  can't be automated).
- Programmatic credentials for the automation principal. Either:
  - `aws configure` with an IAM user's access key + secret (simplest for setup),
    or export `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_DEFAULT_REGION`.
  - For setup the principal needs IAM (create role/policy), EC2 (VPC, SG,
    run/terminate/describe, **ImportImage**), S3 (the import bucket), and
    `iam:PassRole` on `role/vmimport`. `AdministratorAccess` on a dedicated
    automation user is the path of least resistance; scope down later (see
    `docs/aws-provider-analysis.md` for the minimal policy).
- `aws` CLI v2 on PATH (installed at `~/.local/bin/aws`).
- `gsutil` authenticated for `gs://ale-data-public` (already set up here).

## 1. Bootstrap the account/region

```bash
export AWS_REGION=us-east-1                 # STS-enabled by default; large capacity
AWS_REGION=$AWS_REGION ./tools/aws/bootstrap_account.sh
```

Creates: the S3 import bucket, the `vmimport` service role (trust policy with
the mandatory `sts:Externalid=vmimport` condition + S3/EC2 permissions), a VPC +
public subnet + internet gateway + route, and the `ale-sandbox` security group
(ingress tcp:5000 for cua-server, tcp:3389 for Windows RDP). Writes the resolved
ids to `tools/aws/.aws-env`.

## 2. Import the ale-ubuntu22 AMI

```bash
source tools/aws/.aws-env
./tools/aws/import_ubuntu_ami.sh
```

Streams `gs://ale-data-public/images/ale-ubuntu22.tar.gz` (GCE raw export) →
`gunzip | tar -xO` → S3 (no local staging), then `aws ec2 import-image
--format raw` and polls to completion (~1.5–4 h for ~300 GB). Prints the AMI id
and saves it to `tools/aws/.ubuntu-ami`.

> **Format note:** the exported `.vmdk` is `monolithicSparse` and the `.qcow2`
> is unsupported by AWS VM Import — only **raw** (from the tarball),
> stream-optimized VMDK, VHD(X), or OVA work. Hence the raw path.

## 3. First-boot reachability (the main risk)

We reach the box over cua-server on tcp:5000 — not SSH — so EC2 key injection /
cloud-init datasource don't matter. What MUST work on the imported GCE image:

1. **Networking comes up via DHCP on the EC2 NIC.** GCE netplan may pin a
   specific interface name; if the box launches but `:5000` never answers,
   open the **EC2 Serial Console** and set a DHCP-all netplan
   (`network: {version: 2, ethernets: {all: {match: {name: "en*"}, dhcp4: true}}}`),
   then `netplan apply`.
2. **cua-server autostarts** on boot, bound to `0.0.0.0:5000`.
3. **ENA + NVMe drivers** present (Ubuntu 22.04 generic kernel has them) so a
   Nitro instance (default `m6i.2xlarge`) gets disk + network.

If first launch is unreachable, fix on the instance via Serial Console, then
`aws ec2 create-image` to bake a corrected AMI and use that id.

## 4. Point the env config at the AMI and run

```bash
source tools/aws/.aws-env
export ALE_AWS_UBUNTU_AMI=$(cat tools/aws/.ubuntu-ami)

python -m ale_run ... \
  --environment configs/environments/environment_aws.yaml \
  ...   # task data is baked into the AMI -> task_data_source: baked_in_sandbox
```

`environment_aws.yaml` reads `AWS_REGION`, `ALE_AWS_UBUNTU_AMI`, `ALE_AWS_SG`,
`ALE_AWS_SUBNET` from the environment. Start with a single Linux task (e.g.
demo/hello) and `task_data_source: baked_in_sandbox` to validate the provider
without needing the S3 data migration.

## 5. (Later) Task data on S3 + Windows/GPU

- To stage data from S3 instead of baked-in: copy the `gs://` task-data buckets
  to S3, attach an `iam_instance_profile` with `s3:GetObject` to the snapshot,
  and set `task_data_source: s3://<bucket>` — the `s3bucket` backend handles it.
- Windows (`ale-win10`) + GPU: import the win AMI once exported, add snapshot
  blocks with `resolution:` and a GPU instance family (`g5`/`g6`).
