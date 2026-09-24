"""``RoadmapGraph``: an append-only, edge-recorded roadmap over task-space states.

Nodes are replayable states admitted by xyz radius, so node ids are stable. Directed edges are SEED (teleport seed
to first post-teleport frame) or TEMPORAL (consecutive subsamples of one walk). Free of ``isaaclab`` imports.
"""

from __future__ import annotations

import heapq

import numpy as np
import torch

SEED = 0
TEMPORAL = 1

_EDGE_KIND_NAMES = {SEED: "seed", TEMPORAL: "temporal"}


class RoadmapGraph:
    """Append-only SoA roadmap: nodes are task-space states keyed by xyz, edges are directed.

    Args:
        capacity: Maximum number of nodes; later inserts resolve to the nearest node and count in ``dropped_nodes``.
        state_dim: Shape of one node's state payload.
        dedup_dist: Admission radius in metres (xyz) within which a candidate is absorbed into an existing node.
        device: Torch device for ``states`` and ``xyz``; FAISS always runs on CPU.
    """

    def __init__(
        self,
        capacity: int,
        state_dim: tuple[int, ...],
        dedup_dist: float,
        device: str | torch.device = "cpu",
    ) -> None:
        try:
            import faiss
        except ImportError as e:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu") from e

        self._faiss = faiss
        self.capacity = capacity
        self.state_dim = tuple(state_dim)
        self.dedup_dist = dedup_dist
        self.device = device

        self.states = torch.zeros(capacity, *self.state_dim, device=device)
        self.xyz = torch.zeros(capacity, 3, device=device)
        self.n = 0
        self.dropped_nodes = 0

        self.index = faiss.IndexFlatL2(3)

        self._edge_u: list[int] = []
        self._edge_v: list[int] = []
        self._edge_kind: list[int] = []
        self._edge_hop: list[int] = []
        self._edge_set: set[int] = set()

    @property
    def num_nodes(self) -> int:
        return self.n

    @property
    def num_edges(self) -> int:
        return len(self._edge_u)

    def add_or_get(self, states: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """Insert or absorb a batch of candidate nodes; return one int64 node id per input row.

        Rows are absorbed into existing nodes within ``dedup_dist``, then deduped against each other on a voxel
        grid. Rows past ``capacity`` resolve to the nearest existing node.
        """
        states = states.to(self.device)
        xyz = xyz.to(self.device)
        M = states.shape[0]
        ids = torch.full((M,), -1, dtype=torch.long, device=self.device)

        # Phase 1: absorb into the existing graph.
        if self.n > 0:
            dist, idx = self.nearest(xyz, k=1)
            absorbed_mask = dist[:, 0] <= self.dedup_dist
            ids[absorbed_mask] = idx[absorbed_mask, 0]
        else:
            absorbed_mask = torch.zeros(M, dtype=torch.bool, device=self.device)

        remaining_mask = ~absorbed_mask
        if remaining_mask.any():
            rem_idx = remaining_mask.nonzero(as_tuple=True)[0]  # ascending local->original indices
            rem_xyz = xyz[rem_idx]

            # Phase 2: intra-batch voxel dedup; the lowest-index row per voxel is its representative.
            voxel = torch.floor(rem_xyz / self.dedup_dist).long()  # (R, 3)
            _, inverse = torch.unique(voxel, dim=0, return_inverse=True)
            R = rem_idx.shape[0]
            order = torch.arange(R, device=self.device)
            first_local = torch.full((inverse.max().item() + 1,), R, dtype=torch.long, device=self.device)
            first_local.scatter_reduce_(0, inverse, order, reduce="amin", include_self=True)
            rep_of = first_local[inverse]  # (R,) local index of each row's representative
            is_rep = rep_of == order
            rep_local_indices = order[is_rep]  # ascending
            num_reps = rep_local_indices.shape[0]

            can_insert = max(0, self.capacity - self.n)
            num_fit = min(num_reps, can_insert)

            local_to_new_id = torch.full((R,), -1, dtype=torch.long, device=self.device)

            if num_fit > 0:
                fit_local = rep_local_indices[:num_fit]
                fit_orig = rem_idx[fit_local]
                new_ids = torch.arange(self.n, self.n + num_fit, device=self.device)
                self.states[self.n : self.n + num_fit] = states[fit_orig]
                self.xyz[self.n : self.n + num_fit] = xyz[fit_orig]
                self.index.add(xyz[fit_orig].detach().cpu().float().numpy())
                self.n += num_fit
                local_to_new_id[fit_local] = new_ids

            if num_fit < num_reps:
                overflow_local = rep_local_indices[num_fit:]
                self.dropped_nodes += overflow_local.shape[0]
                if self.n > 0:
                    overflow_orig = rem_idx[overflow_local]
                    _, overflow_ids = self.nearest(xyz[overflow_orig], k=1)
                    local_to_new_id[overflow_local] = overflow_ids[:, 0]
                # else: capacity == 0, degenerate; leave unresolved (-1).

            ids[rem_idx] = local_to_new_id[rep_of]

        return ids

    def add_edges(self, u: torch.Tensor, v: torch.Tensor, kind: int, hop: int = 0) -> int:
        """Insert directed edges ``u -> v``, silently dropping self-loops and existing edges of any kind.

        Returns:
            Number of edges actually inserted.
        """
        inserted = 0
        for uu, vv in zip(u.tolist(), v.tolist()):
            if uu == vv:
                continue
            key = uu * self.capacity + vv
            if key in self._edge_set:
                continue
            self._edge_set.add(key)
            self._edge_u.append(uu)
            self._edge_v.append(vv)
            self._edge_kind.append(kind)
            self._edge_hop.append(hop)
            inserted += 1
        return inserted

    def nearest(self, query_xyz: torch.Tensor, k: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the ``k`` nearest nodes for each query point.

        Returns:
            ``((Q, k) float32 L2 distances, (Q, k) int64 node ids)``.

        Raises:
            ValueError: If the graph is empty.
        """
        if self.n == 0:
            raise ValueError("nearest() called on an empty RoadmapGraph")
        q = query_xyz.detach().cpu().float().numpy()
        dist_sq, idx = self.index.search(q, k)
        dist = np.sqrt(np.maximum(dist_sq, 0.0))
        return torch.from_numpy(dist).to(self.device), torch.from_numpy(idx).long().to(self.device)

    def gap_dist(self, goal_xyz: torch.Tensor) -> tuple[float, int]:
        """Return the distance from ``goal_xyz`` to the nearest node and its id, or ``(inf, -1)`` if empty."""
        if self.n == 0:
            return float("inf"), -1
        dist, idx = self.nearest(goal_xyz.reshape(1, 3), k=1)
        return float(dist[0, 0].item()), int(idx[0, 0].item())

    def shortest_path(self, src: int, dst: int) -> list[int]:
        """Return the directed Dijkstra shortest path from ``src`` to ``dst`` by xyz edge length.

        The path is a node list, empty when unreachable.
        """
        if src == dst:
            return [src]

        adj: list[list[tuple[int, float]]] = [[] for _ in range(self.n)]
        if self._edge_u:
            us = torch.tensor(self._edge_u, dtype=torch.long, device=self.device)
            vs = torch.tensor(self._edge_v, dtype=torch.long, device=self.device)
            weights = (self.xyz[us] - self.xyz[vs]).norm(dim=-1).cpu().tolist()
            for uu, vv, ww in zip(self._edge_u, self._edge_v, weights):
                adj[uu].append((vv, ww))

        dist = [float("inf")] * self.n
        prev = [-1] * self.n
        visited = [False] * self.n
        dist[src] = 0.0
        pq: list[tuple[float, int]] = [(0.0, src)]
        while pq:
            d, node = heapq.heappop(pq)
            if visited[node]:
                continue
            visited[node] = True
            if node == dst:
                break
            for nbr, w in adj[node]:
                nd = d + w
                if nd < dist[nbr]:
                    dist[nbr] = nd
                    prev[nbr] = node
                    heapq.heappush(pq, (nd, nbr))

        if dist[dst] == float("inf"):
            return []
        path = []
        node = dst
        while node != -1:
            path.append(node)
            node = prev[node]
        path.reverse()
        return path

    def edge_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``((E, 2) int64 endpoints, (E,) int8 kinds, (E,) int16 hops)``, on CPU."""
        if self._edge_u:
            edges = torch.tensor(list(zip(self._edge_u, self._edge_v)), dtype=torch.long)
            kinds = torch.tensor(self._edge_kind, dtype=torch.int8)
            hops = torch.tensor(self._edge_hop, dtype=torch.int16)
        else:
            edges = torch.zeros((0, 2), dtype=torch.long)
            kinds = torch.zeros((0,), dtype=torch.int8)
            hops = torch.zeros((0,), dtype=torch.int16)
        return edges, kinds, hops

    def to_dict(self) -> dict:
        """The roadmap.pkl core payload: nodes + edges + admission config, all on CPU."""
        edges, kinds, hops = self.edge_tensors()
        return {
            "states": self.states[: self.n].cpu(),
            "xyz": self.xyz[: self.n].cpu(),
            "edges": edges,
            "kinds": kinds,
            "hops": hops,
            "dedup_dist": self.dedup_dist,
            "capacity": self.capacity,
            "dropped_nodes": self.dropped_nodes,
        }
