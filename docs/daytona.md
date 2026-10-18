# Daytona Linux container provider

The Daytona provider is intended to run ALE Linux Docker images in private
Daytona sandboxes, for tasks in the Docker support list. Tasks that require a
full VM must use a VM provider.

The published v1.1 Ubuntu Docker image is currently too large for the tested
Daytona account's 20 GiB snapshot import limit (34.82 GiB compressed). A
compatible image or a higher limit is required before running v1.1 tasks. The
provider has not yet been validated against the original image or for task
score parity with a VM.

Install the optional SDK with `uv sync --extra daytona` and set
`DAYTONA_API_KEY` in the environment or the experiment's secret file. Use
`configs/environments/daytona.yaml` as the environment configuration.

CPU and memory default to the task card's VM shape, with `cpu` and `memory_gb`
as explicit overrides. Set `disk_gb` high enough for the image and task data.
The Docker image is pinned by digest. Sandboxes are private and ephemeral;
`cleanup_mode: delete` removes them, and idle auto-stop also deletes them.
`cleanup_mode: stop` therefore deletes an ephemeral sandbox. `keep` leaves it
running until auto-stop. Configure `auto_stop_minutes` and `ttl_minutes` for
long jobs. Jobs running longer than the idle timeout should increase it, as
CUA requests do not necessarily count as Daytona toolbox activity.

The image does not contain task data. `task_data_source: local:<path>` stages
input and software from the host, preserving modes and symlinks. Reference
files or a `reference.7z` archive are transferred only after the solver has
finished; encrypted archives additionally require
`ALE_REFERENCE_ARCHIVE_PASSWORD` on the host. The existing `gs://` source can
also be used with `gcs_sa_key` configured at the top level of the environment.

To reuse a prebuilt Daytona snapshot, set `daytona.snapshot` to its name.
The provider checks that its image reference and resource shape match the
configured image and task before starting it. A shared snapshot has a fixed
shape, so tasks with different CPU or memory requirements need matching
snapshots or separate image-based sandboxes.

The CUA server is reached through an authenticated private preview and a
loopback proxy; no public sandbox endpoint is created. Agents running inside
the sandbox use CUA on loopback. An agent's model endpoint must be reachable
from Daytona. The provider records image and allocated-resource provenance,
but using the same image does not establish VM equivalence: validate task
inputs, GUI behavior, evaluator results and agent scores separately.

Daytona also has Linux VM and Windows sandbox classes. A Linux VM sandbox with
`kvm: true` exposes `/dev/kvm` and could in principle run QEMU with an ALE VM
disk inside it, including a Windows guest. This requires the organization's
`sandbox_kvm` feature and a Linux VM snapshot, plus adequate CPU, memory and disk.
See [Daytona's nested virtualization documentation](https://www.daytona.io/docs/en/sandboxes/#nested-virtualization)
and [CLI options](https://www.daytona.io/docs/en/tools/cli/).
The tested organization rejected `daytona-vm-small` with `kvm: true` because
KVM sandboxes are not enabled, so guest boot and performance remain untested.
The current provider supports Docker sandboxes only; it does not yet orchestrate
nested QEMU, transfer VM images or validate Windows/Linux guest workflows.
