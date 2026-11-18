"""Per-env circular replay buffer, ported from SuperMimic's ReplayBuffer
(HybridRobotics/supermimic @7e94b07, src/supermimic/replay_buffer.py),
with plain dicts of tensors instead of TensorDict.

Row convention (same as the official runner's _step_and_store): row t of
env e holds the PRE-step state s_t, the action a_t taken from it, and the
flags of that step. terminated/truncated at row t mean the transition out
of row t ended the episode, so row t+1 is the next episode's reset state.

sample(L, B) returns windows of L consecutive rows that lie in one episode:
rows [a, a+L-2] carry no done flag (row a+L-1 may). That gives L-1 real
transitions (s_i, a_i) -> s_{i+1} per window.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FieldSpec:
    dim: int = 0  # 0 = scalar per row, >0 = vector
    dtype: torch.dtype = torch.float32

    def shape(self, num_envs, buffer_size):
        return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
    def __init__(self, schema, buffer_size, num_envs, device="cpu"):
        if not {"terminated", "truncated"} <= schema.keys():
            raise ValueError("schema needs 'terminated' and 'truncated'")
        self.schema = dict(schema)
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.num_transitions = 0
        self._step = 0
        self._storage = {
            k: torch.zeros(s.shape(num_envs, buffer_size), dtype=s.dtype, device=device)
            for k, s in self.schema.items()
        }

    def __getitem__(self, key):
        return self._storage[key]

    def insert(self, fields):
        if set(fields.keys()) != set(self._storage.keys()):
            raise ValueError(f"schema mismatch: {sorted(fields)} vs {sorted(self._storage)}")
        for k, v in fields.items():
            if v.shape[0] != self.num_envs:
                raise ValueError(f"{k}: leading dim {v.shape[0]} != num_envs {self.num_envs}")
            self._storage[k][:, self._step] = v.to(self.device, self._storage[k].dtype)
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def _base(self):
        return self._step if self.num_transitions == self.buffer_size else 0

    def valid_starts(self, L):
        """(E, T) bool over ages: True where a window of L rows can start."""
        E, T, n = self.num_envs, self.buffer_size, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}")
        ages = torch.arange(T, device=self.device)
        ended = (self._storage["terminated"] | self._storage["truncated"])[:, (self._base() + ages) % T].long()
        # interior [a, a+L-2] is done-free iff c[a+L-1] == c[a]
        c = torch.cat([ended.new_zeros(E, 1), ended.cumsum(1), ended.new_zeros(E, L - 1)], dim=1)
        return (c[:, L - 1:L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)

    def sample_indices(self, L, B, generator=None):
        """(env_idx (B,), storage_idx (B, L)), uniform over all valid windows
        of envs that have any (env drawn uniformly, then a start inside it,
        matching the official code)."""
        T = self.buffer_size
        valid = self.valid_starts(L)
        csum = valid.cumsum(dim=1, dtype=torch.long)
        total = csum[:, -1]
        if not (total > 0).any():
            raise RuntimeError(f"no valid sequences L={L} n={self.num_transitions}")
        env_idx = torch.multinomial((total > 0).float(), B, replacement=True, generator=generator)
        u = (torch.rand(B, device=self.device, generator=generator) * total[env_idx].float()).long()
        age_start = torch.searchsorted(csum[env_idx], u[:, None], right=True).squeeze(-1)
        storage_idx = (self._base() + age_start[:, None] + torch.arange(L, device=self.device)) % T
        return env_idx, storage_idx

    def sample(self, L, B, generator=None):
        env_idx, idx = self.sample_indices(L, B, generator)
        return {k: v[env_idx[:, None], idx] for k, v in self._storage.items()}

    def sample_starts(self, B, generator=None):
        """B single rows, for imagined-rollout start states."""
        return {k: v[:, 0] for k, v in self.sample(1, B, generator).items()}

    # ---- persistence (collection script writes, offline fit reads) ----
    def state_dict(self):
        return {
            "schema": {k: (s.dim, s.dtype) for k, s in self.schema.items()},
            "buffer_size": self.buffer_size,
            "num_envs": self.num_envs,
            "num_transitions": self.num_transitions,
            "step": self._step,
            "storage": {k: v.cpu() for k, v in self._storage.items()},
        }

    @classmethod
    def from_state_dict(cls, sd, device="cpu"):
        schema = {k: FieldSpec(d, t) for k, (d, t) in sd["schema"].items()}
        buf = cls(schema, sd["buffer_size"], sd["num_envs"], device)
        buf.num_transitions = sd["num_transitions"]
        buf._step = sd["step"]
        buf._storage = {k: v.to(device) for k, v in sd["storage"].items()}
        return buf
