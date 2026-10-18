"""Provider implementations for VM lifecycle.

``Provider`` ABC + ``SandboxSpec`` + ``SandboxHandle`` + ``ReleaseMode`` live in
:mod:`ale_run.base_interface`; this package only holds the backends:

  - :class:`GcloudProvider` (``gcloud.py``): ephemeral GCE VMs.
  - :class:`AwsProvider` (``aws.py``): ephemeral EC2 instances.
  - :class:`AliyunProvider` (``aliyun.py``): ephemeral Alibaba Cloud ECS instances.
  - :class:`StaticProvider` (``static.py``): a pre-existing VM endpoint.
  - :class:`DockerProvider` (``docker.py``): ephemeral Docker containers.
  - :class:`DaytonaProvider` (``daytona.py``): Daytona Docker sandboxes.
  - :class:`QemuProvider` (``qemu.py``): ephemeral local QEMU VMs in Docker.
"""

from ...base_interface import SandboxSpec, Provider, ReleaseMode, SandboxHandle
from .aws import AwsProvider, AwsProviderConfig
from .aliyun import AliyunProvider, AliyunProviderConfig
from .docker import DockerProvider, DockerProviderConfig
from .daytona import DaytonaProvider, DaytonaProviderConfig
from .gcloud import GcloudProvider, GcloudProviderConfig
from .qemu import QemuProvider, QemuProviderConfig
from .static import StaticProvider, StaticProviderConfig

__all__ = [
    "SandboxSpec",
    "AwsProvider",
    "AwsProviderConfig",
    "AliyunProvider",
    "AliyunProviderConfig",
    "DockerProvider",
    "DockerProviderConfig",
    "DaytonaProvider",
    "DaytonaProviderConfig",
    "GcloudProvider",
    "GcloudProviderConfig",
    "Provider",
    "QemuProvider",
    "QemuProviderConfig",
    "ReleaseMode",
    "StaticProvider",
    "StaticProviderConfig",
    "SandboxHandle",
]
