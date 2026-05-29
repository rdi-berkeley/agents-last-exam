"""Shared constants for pauli_channel_cpu_instance_1."""

from dataclasses import dataclass

DOMAIN_NAME = "computing_math"
TASK_NAME = "pauli_channel_cpu_instance_1"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"


@dataclass(frozen=True)
class VariantSpec:
    variant_name: str
    n_qubits: int
    num_random_states: int
    state_support_size: int
    num_trajectories: int
    state_seed: int
    trajectory_seed: int
    channel_seed: int

    @property
    def matrix_dim(self) -> int:
        return 1 << self.n_qubits

    @property
    def density_filenames(self) -> tuple[str, ...]:
        return tuple(f"state_{idx:03d}.npz" for idx in range(self.num_random_states))


VARIANT = VariantSpec(
    variant_name="n12_pair_noise",
    n_qubits=12,
    num_random_states=100,
    state_support_size=100,
    num_trajectories=10000,
    state_seed=1121,
    trajectory_seed=1221,
    channel_seed=3012,
)
