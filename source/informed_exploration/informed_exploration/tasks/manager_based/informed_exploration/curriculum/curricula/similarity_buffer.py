from __future__ import annotations

import numpy as np
import torch
import faiss

class SimilarityBuffer:
    """FIFO buffer with FAISS nearest-neighbour search and an optional separate metric for dedup.

    With ``novelty_projection``, a second index tracks the novelty metric and is read only by
    :meth:`add_novel`; :meth:`get_similar` always searches the primary index.
    """

    def __init__(
        self,
        dim: int,
        max_size: int,
        item_shape: tuple[int, ...],
        projection: callable | None = None,
        scale: float = 0.7,
        novelty_dim: int | None = None,
        novelty_projection: callable | None = None,
    ):
        assert (novelty_dim is None) == (novelty_projection is None), (
            "novelty_dim and novelty_projection must be set together"
        )

        self.max_size = max_size
        self.scale = scale
        self.dim = dim
        self.item_ndim = len(item_shape)
        self.projection = projection
        self.novelty_dim = novelty_dim
        self.novelty_projection = novelty_projection

        self.buffer: list[torch.Tensor] = []
        self.similarity_index = faiss.IndexFlatL2(dim)  # type: ignore
        self.novelty_index = faiss.IndexFlatL2(novelty_dim) if novelty_projection is not None else None  # type: ignore

    def __iter__(self):
        return iter(self.buffer)

    def __len__(self):
        return len(self.buffer)

    def _to_index_vecs(self, item: torch.Tensor) -> np.ndarray:
        """Project, if configured, and flatten ``item`` to a ``(rows, dim)`` float32 array."""
        vec = self.projection(item) if self.projection else item
        return vec.reshape(-1, self.dim).cpu().numpy().astype(np.float32)

    def _to_novelty_vecs(self, item: torch.Tensor) -> np.ndarray:
        """Like :meth:`_to_index_vecs` but through ``novelty_projection``/``novelty_dim``."""
        vec = self.novelty_projection(item) if self.novelty_projection else item
        return vec.reshape(-1, self.novelty_dim).cpu().numpy().astype(np.float32)

    def _rebuild_indices(self):
        """Rebuild both FAISS indices from the current :attr:`buffer` in one batched insertion each."""
        self.similarity_index = faiss.IndexFlatL2(self.dim)
        if self.novelty_index is not None:
            self.novelty_index = faiss.IndexFlatL2(self.novelty_dim)
        if self.buffer:
            stacked = torch.stack(self.buffer)
            self.similarity_index.add(self._to_index_vecs(stacked))
            if self.novelty_index is not None:
                self.novelty_index.add(self._to_novelty_vecs(stacked))

    def _trim_buffer(self):
        excess = len(self.buffer) - int(self.max_size * self.scale)
        self.buffer = self.buffer[excess:]
        self._rebuild_indices()

    def add(self, item: torch.Tensor):
        """Add a single task ``(*task_dim)`` or a batch ``(n, *task_dim)``.

        Batching is detected from the rank against ``item_shape``, since a batch of one flattens to a single row.
        """
        vecs = self._to_index_vecs(item)  # (rows, dim)
        novelty_vecs = self._to_novelty_vecs(item) if self.novelty_index is not None else None
        if item.dim() > self.item_ndim:
            self.buffer.extend(item)  # batch: each row becomes its own buffer entry
        else:
            self.buffer.append(item)
        self.similarity_index.add(vecs)
        if novelty_vecs is not None:
            self.novelty_index.add(novelty_vecs)

        if len(self.buffer) >= self.max_size:
            self._trim_buffer()

    def extend(self, items: list[torch.Tensor]):
        for item in items:
            self.add(item)

    def add_novel(self, item: torch.Tensor, min_sq_dist: float) -> int:
        """Add only rows of ``item`` at least ``sqrt(min_sq_dist)`` from their nearest neighbour in the dedup metric.

        Rows are checked against the buffer and earlier accepted rows of the same call. A non-positive
        ``min_sq_dist`` falls back to :meth:`add`.

        Returns:
            The number of rows inserted.
        """
        vecs = self._to_index_vecs(item)  # (rows, dim), the selection metric
        if self.novelty_index is not None:
            novelty_vecs = self._to_novelty_vecs(item)  # (rows, novelty_dim), the dedup metric
            dedup_index = self.novelty_index
        else:
            novelty_vecs = vecs
            dedup_index = self.similarity_index
        n = vecs.shape[0]
        if min_sq_dist <= 0.0:
            self.add(item)
            return n

        # same rank-based batch check as add()
        rows = list(item) if item.dim() > self.item_ndim else [item]

        # phase 1 (batched): reject rows too close to the existing buffer, in the dedup metric
        keep = np.ones(n, dtype=bool)
        if dedup_index.ntotal > 0:
            dist, _ = dedup_index.search(novelty_vecs, k=1)
            keep = dist[:, 0] >= min_sq_dist

        # phase 2 (greedy): dedup the survivors against each other so a close cluster admits one row
        work_index = faiss.IndexFlatL2(dedup_index.d)
        final_rows: list[torch.Tensor] = []
        final_vecs: list[np.ndarray] = []
        final_novelty_vecs: list[np.ndarray] = []
        for r in np.nonzero(keep)[0]:
            v = novelty_vecs[r : r + 1]
            if work_index.ntotal > 0:
                dist, _ = work_index.search(v, k=1)
                if dist[0, 0] < min_sq_dist:
                    continue
            work_index.add(v)
            final_rows.append(rows[int(r)])
            final_vecs.append(vecs[r : r + 1])
            final_novelty_vecs.append(v)

        if not final_rows:
            return 0

        # store and index the survivors in one batched insertion, as add()'s batch path does
        self.buffer.extend(final_rows)
        self.similarity_index.add(np.concatenate(final_vecs, axis=0))
        if self.novelty_index is not None:
            self.novelty_index.add(np.concatenate(final_novelty_vecs, axis=0))
        if len(self.buffer) >= self.max_size:
            self._trim_buffer()
        return len(final_rows)

    def get_random(self, num_samples: int, replace: bool = False) -> torch.Tensor:
        """Uniform draw of ``num_samples`` items, capped at the buffer length unless ``replace`` is set."""
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        if replace:
            indices = np.random.randint(0, len(self.buffer), size=num_samples)
        else:
            indices = np.random.choice(len(self.buffer), min(num_samples, len(self.buffer)), replace=False)
        return torch.stack([self.buffer[i] for i in indices], dim=0)

    def get_similar(self, item: torch.Tensor, knn: int) -> torch.Tensor:
        """Return one uniformly chosen top-``knn`` neighbour per query row, in the primary metric."""
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        query = self._to_index_vecs(item)
        k = min(knn, len(self.buffer))  # avoid faiss padding indices with -1 when k > ntotal
        _, indices = self.similarity_index.search(query, k=k)

        # one neighbour per query, uniform over the top-k, so the row count is unchanged
        choice = np.random.randint(0, indices.shape[1], size=indices.shape[0])
        picked = indices[np.arange(indices.shape[0]), choice]
        return torch.stack([self.buffer[int(i)] for i in picked], dim=0)
