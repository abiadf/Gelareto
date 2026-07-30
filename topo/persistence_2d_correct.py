"""Stream 2D row chunks and return exact prefix H0/H1 after each update."""

from __future__ import annotations
import heapq
from dataclasses import dataclass, field
import numpy as np


EXTERIOR_SENTINEL = -1


def count_grid_work(shape: tuple[int, int], chunk_rows: int | None = None) -> dict[str, int]:
    """Count grid cells processed by the exact solver."""
    R, C = shape
    if R < 1 or C < 1:
        return {
            "pixels": 0,
            "edges": 0,
            "faces": 0,
            "h0_sort_items": 0,
            "h1_sort_items": 0,
            "dsu_find_calls_approx": 0,
            "old_cells_rebuilt": 0,
            "new_cells_built": 0,
            "boundary_edges": 0,}

    pixels = R * C
    edges = R * max(C - 1, 0) + max(R - 1, 0) * C
    faces = max(R - 1, 0) * max(C - 1, 0)

    if chunk_rows is None or chunk_rows >= R:
        new_cells_built = pixels + edges + faces
        old_cells_rebuilt = new_cells_built
        boundary_edges = 0
    else:
        boundary_edges = max(0, ((R - 1) // chunk_rows)) * C
        new_cells_built = pixels + edges + faces
        old_cells_rebuilt = 0

    return {
        "pixels": pixels,
        "edges": edges,
        "faces": faces,
        "h0_sort_items": edges,
        "h1_sort_items": edges,
        "dsu_find_calls_approx": 4 * edges,
        "old_cells_rebuilt": old_cells_rebuilt,
        "new_cells_built": new_cells_built,
        "boundary_edges": boundary_edges,
    }


def print_grid_work(label: str, shape: tuple[int, int], chunk_rows: int | None = None) -> None:
    counts = count_grid_work(shape, chunk_rows=chunk_rows)
    print(f"\n{label} work units")
    print(f"pixels: {counts['pixels']}")
    print(f"edges processed: {counts['edges']}")
    print(f"faces: {counts['faces']}")
    print(f"H0 sort items: {counts['h0_sort_items']}")
    print(f"H1 sort items: {counts['h1_sort_items']}")
    print(f"DSU find calls approx: {counts['dsu_find_calls_approx']}")
    print(f"old cells rebuilt: {counts['old_cells_rebuilt']}")
    print(f"new/boundary cells built: {counts['new_cells_built']}")
    print(f"boundary edges: {counts['boundary_edges']}")


def find_critical_pixels_2d(grid: np.ndarray) -> dict[str, np.ndarray]:
    """Mark minima, maxima, and saddle-like pixels."""
    grid = np.asarray(grid, dtype=np.float32)
    minima = np.zeros(grid.shape, dtype=bool)
    maxima = np.zeros(grid.shape, dtype=bool)
    saddles = np.zeros(grid.shape, dtype=bool)
    R, C = grid.shape

    for r in range(1, R - 1):
        for c in range(1, C - 1):
            center = grid[r, c]
            ring = np.array(
                [
                    grid[r - 1, c - 1],
                    grid[r - 1, c],
                    grid[r - 1, c + 1],
                    grid[r, c + 1],
                    grid[r + 1, c + 1],
                    grid[r + 1, c],
                    grid[r + 1, c - 1],
                    grid[r, c - 1],
                ],
                dtype=np.float32,
            )
            any_gt = np.any(ring > center)
            any_lt = np.any(ring < center)
            if np.all(ring >= center) and any_gt:
                minima[r, c] = True
            elif np.all(ring <= center) and any_lt:
                maxima[r, c] = True
            elif any_gt and any_lt:
                not_lower = ring >= center
                transitions = np.count_nonzero(not_lower != np.roll(not_lower, -1))
                saddles[r, c] = transitions // 2 >= 2

    return {"minima": minima, "maxima": maxima, "saddles": saddles}


def count_adaptive_work(grid: np.ndarray, chunk_rows: int | None = None) -> dict[str, int]:
    """Count critical pixels plus streaming frontier pixels."""
    grid = np.asarray(grid, dtype=np.float32)
    R, C = grid.shape
    critical = find_critical_pixels_2d(grid)
    n_min = int(np.count_nonzero(critical["minima"]))
    n_max = int(np.count_nonzero(critical["maxima"]))
    n_saddle = int(np.count_nonzero(critical["saddles"]))
    n_critical = n_min + n_max + n_saddle

    if chunk_rows is None or chunk_rows >= R:
        frontier_pixels = 0
        chunks = 1
    else:
        chunks = int(np.ceil(R / chunk_rows))
        frontier_pixels = max(0, chunks - 1) * C

    return {
        "chunks": chunks,
        "minima": n_min,
        "maxima": n_max,
        "saddles": n_saddle,
        "critical_pixels": n_critical,
        "frontier_pixels": frontier_pixels,
        "adaptive_events": n_critical + frontier_pixels,
    }


def print_adaptive_work(label: str, grid: np.ndarray, chunk_rows: int | None = None) -> None:
    counts = count_adaptive_work(grid, chunk_rows=chunk_rows)
    print(f"\n{label} adaptive work units")
    print(f"chunks: {counts['chunks']}")
    print(f"minima: {counts['minima']}")
    print(f"maxima: {counts['maxima']}")
    print(f"saddles: {counts['saddles']}")
    print(f"critical pixels: {counts['critical_pixels']}")
    print(f"frontier pixels: {counts['frontier_pixels']}")
    print(f"adaptive events approx: {counts['adaptive_events']}")


@dataclass
class _DSU:
    parent: np.ndarray
    birth: np.ndarray

    @classmethod
    def pixels(cls, values: np.ndarray) -> "_DSU":
        n = len(values)
        return cls(np.arange(n, dtype=np.int64), values.astype(np.float32, copy=False))

    @classmethod
    def faces(cls, values: np.ndarray) -> "_DSU":
        n = len(values)
        parent = np.arange(n + 1, dtype=np.int64)
        birth = np.empty(n + 1, dtype=np.float32)
        birth[:n] = values.astype(np.float32, copy=False)
        birth[n] = np.inf
        return cls(parent, birth)

    def root(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x


@dataclass
class IncrementalH0DSUState:
    """Reusable H0 DSU processed below safe watermarks."""

    parent: list[int] = field(default_factory=list)
    birth: list[float] = field(default_factory=list)
    h0_birth_death_pairs: list[tuple[float, float]] = field(default_factory=list)
    pending_edges: list[tuple[float, int, int]] = field(default_factory=list)
    edges_processed: int = 0
    last_watermark: float = -np.inf

    def add_vertex_births(self, values: np.ndarray) -> None:
        """Add new pixels as live H0 components."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        start = len(self.parent)
        self.parent.extend(range(start, start + len(values)))
        self.birth.extend(float(v) for v in values)

    def queue_h0_edges(self, component_edges: np.ndarray, watermark: float | None = None) -> None:
        """Queue H0 edges and process safe ones."""
        if len(component_edges) == 0:
            if watermark is not None:
                self.process_edges_until(watermark)
            return
        for val, u_float, v_float in component_edges:
            val = float(val)
            if val < self.last_watermark:
                raise ValueError("new edge is below the processed H0 watermark")
            heapq.heappush(self.pending_edges, (val, int(u_float), int(v_float)))
        if watermark is not None:
            self.process_edges_until(watermark)

    def process_edges_until(self, watermark: float) -> None:
        """Process queued H0 edges up to watermark."""
        if watermark < self.last_watermark:
            raise ValueError("H0 watermark cannot move backward")
        while self.pending_edges and self.pending_edges[0][0] <= watermark:
            val, u, v = heapq.heappop(self.pending_edges)
            self._merge_components(u, v, val)
            self.edges_processed += 1
        self.last_watermark = float(watermark)

    def diagram(self) -> np.ndarray:
        """Return processed H0 pairs plus survivor."""
        pairs = list(self.h0_birth_death_pairs)
        if self.birth:
            pairs.append((min(self.birth), np.inf))
        return np.asarray(pairs, dtype=np.float32)

    def summary(self) -> dict[str, int]:
        """Count online H0 state."""
        return {
            "vertices_seen": len(self.parent),
            "edges_processed_online": self.edges_processed,
            "edges_waiting_for_watermark": len(self.pending_edges),
            "h0_pairs_recorded_online": len(self.h0_birth_death_pairs),
        }

    def _root(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _merge_components(self, u: int, v: int, death_value: float) -> None:
        ru, rv = self._root(u), self._root(v)
        if ru == rv:
            return
        if self.birth[ru] <= self.birth[rv]:
            survivor, victim = ru, rv
        else:
            survivor, victim = rv, ru
        if death_value > self.birth[victim]:
            self.h0_birth_death_pairs.append((self.birth[victim], death_value))
        self.parent[victim] = survivor


@dataclass
class IncrementalStreamingPersistence2D:
    """Update H0 per chunk; finalize exact H0/H1."""

    width: int | None = None
    rows: int = 0
    vertex_birth_values: list[np.ndarray] = field(default_factory=list)
    face_birth_values: list[np.ndarray] = field(default_factory=list)
    component_merge_edges: list[np.ndarray] = field(default_factory=list)
    loop_dual_edges: list[np.ndarray] = field(default_factory=list)
    critical_counts: list[dict[str, int]] = field(default_factory=list)
    incremental_h0_state: IncrementalH0DSUState = field(
        default_factory=IncrementalH0DSUState
    )
    _bottom_h1_finalized: bool = False

    def update_chunk(self, chunk: np.ndarray, h0_watermark: float | None = None) -> None:
        """Add one chunk and update safe H0 merges."""
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError("chunk must be a 2D array")
        if chunk.shape[0] == 0:
            return
        if self.width is None:
            self.width = int(chunk.shape[1])
        if chunk.shape[1] != self.width:
            raise ValueError(f"expected width {self.width}, got {chunk.shape[1]}")
        if self._bottom_h1_finalized:
            raise RuntimeError("cannot update after finalize()")

        old_rows = self.rows
        context = self._chunk_with_boundary_context(chunk)
        self.critical_counts.append(count_adaptive_work(context))
        self.vertex_birth_values.append(chunk.reshape(-1))
        self.incremental_h0_state.add_vertex_births(chunk)
        full = self.values()
        self.rows += int(chunk.shape[0])

        component_edges = self._make_h0_component_edges(full, old_rows, self.rows)
        if len(component_edges) > 0:
            self.component_merge_edges.append(component_edges)
            self.incremental_h0_state.queue_h0_edges(component_edges, watermark=h0_watermark)
        self._store_face_birth_values(full, old_rows, self.rows)
        self._store_h1_dual_edges_except_bottom(full, old_rows, self.rows)

    def stored_state_summary(self) -> dict[str, int]:
        """Count stored streamed state."""
        summary = {
            "rows_seen": self.rows,
            "vertex_birth_blocks": len(self.vertex_birth_values),
            "face_birth_blocks": len(self.face_birth_values),
            "h0_component_edge_blocks": len(self.component_merge_edges),
            "h1_loop_edge_blocks": len(self.loop_dual_edges),
            "critical_count_blocks": len(self.critical_counts),
        }
        summary.update(self.incremental_h0_state.summary())
        return summary

    def values(self) -> np.ndarray:
        """Return streamed values as one grid."""
        if not self.vertex_birth_values:
            return np.empty((0, 0 if self.width is None else self.width), dtype=np.float32)
        return np.concatenate(self.vertex_birth_values).reshape(-1, self.width)

    def current_exact_prefix_diagrams(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute exact H0/H1 for chunks seen so far."""
        if self.width is None or self.rows == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty

        full = self.values()

        pix_vals = full.reshape(-1).astype(np.float32, copy=False)
        face_vals = (
            np.concatenate(self.face_birth_values).astype(np.float32, copy=False)
            if self.face_birth_values
            else np.empty(0, dtype=np.float32)
        )
        component_edges = (
            np.vstack(self.component_merge_edges).astype(np.float32, copy=False)
            if self.component_merge_edges
            else np.empty((0, 3), dtype=np.float32)
        )
        loop_edges = (
            np.vstack(self.loop_dual_edges).astype(np.float32, copy=False)
            if self.loop_dual_edges
            else np.empty((0, 3), dtype=np.float32)
        )
        bottom_edges = self._make_current_bottom_h1_dual_edges(full)
        if len(bottom_edges) > 0:
            loop_edges = np.vstack([loop_edges, bottom_edges]).astype(np.float32, copy=False)

        return compute_h0_component_pairs(component_edges, pix_vals), compute_h1_loop_pairs(
            loop_edges, face_vals
        )

    def finalize_exact(self) -> tuple[np.ndarray, np.ndarray]:
        """Return exact H0/H1 for the final streamed prefix."""
        self.incremental_h0_state.process_edges_until(np.inf)
        return self.current_exact_prefix_diagrams()

    def _pid(self, row: int, col: int) -> int:
        return row * self.width + col

    def _fid(self, row: int, col: int) -> int:
        return row * (self.width - 1) + col

    def _chunk_with_boundary_context(self, chunk: np.ndarray) -> np.ndarray:
        if not self.vertex_birth_values:
            return chunk
        previous_last_row = self.values()[-1:, :]
        return np.vstack([previous_last_row, chunk])

    def _make_h0_component_edges(self, full: np.ndarray, old_rows: int, new_rows: int) -> np.ndarray:
        """Create H0 edges for new rows plus boundary."""
        C = self.width
        records = []

        for r in range(old_rows, new_rows):
            for c in range(C - 1):
                u, v = self._pid(r, c), self._pid(r, c + 1)
                records.append((max(full[r, c], full[r, c + 1]), u, v))

        for r in range(max(0, old_rows - 1), new_rows - 1):
            for c in range(C):
                u, v = self._pid(r, c), self._pid(r + 1, c)
                records.append((max(full[r, c], full[r + 1, c]), u, v))

        if not records:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(records, dtype=np.float32)

    def _store_face_birth_values(self, full: np.ndarray, old_rows: int, new_rows: int) -> None:
        """Store H1 face birth values."""
        C = self.width
        if C < 2:
            return
        records = []
        for r in range(max(0, old_rows - 1), new_rows - 1):
            for c in range(C - 1):
                records.append(
                    max(full[r, c], full[r, c + 1], full[r + 1, c], full[r + 1, c + 1])
                )
        if records:
            self.face_birth_values.append(np.asarray(records, dtype=np.float32))

    def _store_h1_dual_edges_except_bottom(
        self, full: np.ndarray, old_rows: int, new_rows: int
    ) -> None:
        """Store H1 dual edges except final bottom exterior."""
        C = self.width
        if C < 2 or new_rows < 2:
            return
        records = []

        for r in range(max(0, old_rows - 1), new_rows - 1):
            for c in range(C - 1):
                f1 = self._fid(r - 1, c) if r > 0 else EXTERIOR_SENTINEL
                f2 = self._fid(r, c)
                records.append((-max(full[r, c], full[r, c + 1]), f1, f2))

        for r in range(max(0, old_rows - 1), new_rows - 1):
            for c in range(C):
                f1 = self._fid(r, c - 1) if c > 0 else EXTERIOR_SENTINEL
                f2 = self._fid(r, c) if c < C - 1 else EXTERIOR_SENTINEL
                records.append((-max(full[r, c], full[r + 1, c]), f1, f2))

        if records:
            self.loop_dual_edges.append(np.asarray(records, dtype=np.float32))

    def _store_final_bottom_h1_dual_edges(self, full: np.ndarray) -> None:
        """Close bottom exterior for H1 loops."""
        if self._bottom_h1_finalized or self.width < 2 or self.rows < 2:
            return
        C = self.width
        r = self.rows - 1
        records = []
        for c in range(C - 1):
            records.append(
                (-max(full[r, c], full[r, c + 1]), self._fid(r - 1, c), EXTERIOR_SENTINEL)
            )
        self.loop_dual_edges.append(np.asarray(records, dtype=np.float32))
        self._bottom_h1_finalized = True

    def _make_current_bottom_h1_dual_edges(self, full: np.ndarray) -> np.ndarray:
        """Create current-prefix bottom exterior H1 edges."""
        if self.width < 2 or self.rows < 2:
            return np.empty((0, 3), dtype=np.float32)
        C = self.width
        r = self.rows - 1
        records = []
        for c in range(C - 1):
            records.append(
                (-max(full[r, c], full[r, c + 1]), self._fid(r - 1, c), EXTERIOR_SENTINEL)
            )
        return np.asarray(records, dtype=np.float32)


def compute_h0_component_pairs(component_edges: np.ndarray, vertex_births: np.ndarray) -> np.ndarray:
    """Compute H0 birth-death pairs from component edges."""
    if len(vertex_births) == 0:
        return np.empty((0, 2), dtype=np.float32)

    dsu = _DSU.pixels(vertex_births)
    pairs = []
    for val, u_float, v_float in component_edges[np.argsort(component_edges[:, 0])]:
        u, v = int(u_float), int(v_float)
        ru, rv = dsu.root(u), dsu.root(v)
        if ru == rv:
            continue
        if dsu.birth[ru] <= dsu.birth[rv]:
            survivor, victim = ru, rv
        else:
            survivor, victim = rv, ru
        if val > dsu.birth[victim]:
            pairs.append((dsu.birth[victim], val))
        dsu.parent[victim] = survivor

    pairs.append((float(np.min(vertex_births)), np.inf))
    return np.asarray(pairs, dtype=np.float32)


def compute_h1_loop_pairs(loop_dual_edges: np.ndarray, face_births: np.ndarray) -> np.ndarray:
    """Compute H1 birth-death pairs from dual-face edges."""
    if len(face_births) == 0 or len(loop_dual_edges) == 0:
        return np.empty((0, 2), dtype=np.float32)

    exterior = len(face_births)
    edges = loop_dual_edges.copy()
    edges[:, 1][edges[:, 1] == EXTERIOR_SENTINEL] = exterior
    edges[:, 2][edges[:, 2] == EXTERIOR_SENTINEL] = exterior
    dsu = _DSU.faces(face_births)
    pairs = []
    for neg_val, f1_float, f2_float in edges[np.argsort(edges[:, 0])]:
        val = -neg_val
        f1, f2 = int(f1_float), int(f2_float)
        r1, r2 = dsu.root(f1), dsu.root(f2)
        if r1 == r2:
            continue
        if dsu.birth[r1] >= dsu.birth[r2]:
            survivor, victim = r1, r2
        else:
            survivor, victim = r2, r1
        if victim != exterior and dsu.birth[victim] > val:
            pairs.append((val, dsu.birth[victim]))
        dsu.parent[victim] = survivor

    return np.asarray(pairs, dtype=np.float32)


def compute_streamed_by_rows(grid: np.ndarray, chunk_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact pairs from row-streamed chunks."""
    engine = IncrementalStreamingPersistence2D()
    grid = np.asarray(grid, dtype=np.float32)
    for start in range(0, grid.shape[0], chunk_rows):
        engine.update_chunk(grid[start : start + chunk_rows])
    return engine.finalize_exact()


def compute_exact_prefix_diagrams_by_rows(
    grid: np.ndarray, chunk_rows: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return exact H0/H1 after every streamed row chunk."""
    engine = IncrementalStreamingPersistence2D()
    grid = np.asarray(grid, dtype=np.float32)
    diagrams = []
    for start in range(0, grid.shape[0], chunk_rows):
        engine.update_chunk(grid[start : start + chunk_rows])
        diagrams.append(engine.current_exact_prefix_diagrams())
    return diagrams


def compute_2d_prefix_diagrams_by_full_recompute(
    grid: np.ndarray, chunk_rows: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return prefix H0/H1 by rebuilding each prefix from scratch."""
    grid = np.asarray(grid, dtype=np.float32)
    diagrams = []
    for end in range(chunk_rows, grid.shape[0] + chunk_rows, chunk_rows):
        end = min(end, grid.shape[0])
        engine = IncrementalStreamingPersistence2D()
        engine.update_chunk(grid[:end])
        diagrams.append(engine.current_exact_prefix_diagrams())
        if end == grid.shape[0]:
            break
    return diagrams
