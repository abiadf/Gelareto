"[old and slow] main persistence code"
import gc
from time import perf_counter
import torch
import psutil
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_memory():
    """Forces garbage collection to erase backend residuals between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_vms_mib():
    return psutil.Process(os.getpid()).memory_info().vms / (1024 * 1024)


"main persistence code"

# to remove, edge_index is more efficient
def adjacency_from_distance_matrix(D: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Given distance matrix D and threshold epsilon, find which node pairs are connected
    Output: binary (bool) adjacency matrix from distance matrix. No self-edges"""
    A = (D < epsilon) & (D > 0.0) # keep distances < epsilon, and remove self-edges (D=0)
    A = torch.triu(A, diagonal=1) # take upper triangle + remove diagonal
    A = A | A.T                   # applies upper triangle to lower part to make symmetric (without diagonal)
    return A

def count_edges(A: torch.Tensor) -> int:
    """Count undirected edges from upper (or lower) triangle"""
    return int(torch.triu(A, diagonal=1).sum().item())

def count_triangles_from_adjacency(A: torch.Tensor) -> int:
    """Count undirected triangles using spectral triangle counting.
    uses matrix trace identity: # triangles = trace(A^3)/6
    - each triangle appears 6 times"""
    Af = A.float()
    A3 = Af @ Af @ Af # compute A^3 (counts length-3 walks between nodes)
    return int(torch.trace(A3).item() / 6) # Trace gives total closed 3-walks; divide by 6 to correct overcounting

class UnionFind:
    """Union-find / disjoint set."""
    def __init__(self, n: int):
        self.parent = torch.arange(n, device=device)
        self.rank = torch.zeros(n, dtype=torch.int32, device=device)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return

        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1

    def num_components(self) -> int:
        # roots = torch.unique(self.parent)
        roots = set(self.find(int(i)) for i in range(len(self.parent)))
        return len(roots)

class BettiComputer:
    def __init__(self, n_triangles, n_edges, n_vertices, device=device):
        self.n_triangles = n_triangles
        self.n_edges     = n_edges
        self.n_vertices  = n_vertices
        self.device      = device

    def compute_betti_0(self, adj: torch.Tensor, tolerance: float = 1e-5) -> int:
        """Connected components via Laplacian spectrum. Betti-0 via graph Laplacian.. B0 = vertices - rank."""
        A       = adj.float()
        degree  = torch.diag(A.sum(dim=1))
        L       = degree - A
        eigvals = torch.linalg.eigvalsh(L)
        return int((eigvals < tolerance).sum().item())

    def compute_betti_0_edge_index(self, edge_index: torch.Tensor, n_vertices: int) -> int:
        """Connected components via union-find."""
        uf       = UnionFind(n_vertices)
        src, dst = edge_index
        for i in range(src.numel()):
            uf.union(int(src[i]), int(dst[i]))
        self.B0 = uf.num_components()
        return self.B0

    def compute_betti_1(self, V: int, E: int, B0: int) -> int:
        """Betti B1 = edges + connected components - vertices
        connected components C = B0, which are the # of unique clusters of points"""
        return max(E + B0 - V, 0)



# ========== another old and super slow algo ==========
# NOTE: we currently treat min-boundary as boundary, need to treat as min
# NOTE: when there are many equal g-min, gudhi makes just 1 go to inf, the other one dies at g-max
# (can be seen as a 'bug')
# TODO: when we increment the timeseries, the boundary min/gmin need to be removed (since the bonudary is gone)

MIN, MAX, GMIN, GMAX = 1, -1, 2, -2 # int encodings for keypoints


class Sublevel1D:
    """Implementation for 1D timeseries sublevel β₀ persistence. We keep track of keypoints (min/max/gmin/gmax) and their indices, and then compute persistence pairs via a pairing algorithm
    We also use vectors (no arrays/dicts) to make it parallelizable"""
    def __init__(self, x: torch.Tensor = None):
        """x = 1D timeseries"""
        self.x      = x # timeseries signal
        self.device = x.device

        # keypoints
        self.keypoint_idx   = None
        self.keypoint_types = None
        # persistence arrays (one entry per minimum pt)
        self.persistence_idx    = None # indices of each MINIMUM
        self.persistence_births = None # = y-val
        self.persistence_deaths = None # = y=val
        # active 
        self.active_min_idx = None  # minima idx, consumed during pairing
        self.active_min_y   = None  # minima y-val, consumed during pairing
        self.active_max_idx = None  # maxima idx, consumed during pairing
        self.active_max_y   = None  # maxima y-val, consumed during pairing
        # globals
        self.gmax_idx  = None
        self.gmax_yval = None
        self.gmin_idx  = None
        self.gmin_yval = None

    def find_extrema_in_timeseries(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (indices, types) of minima and maxima, sorted by index.
        Types: MIN=1, MAX=-1, GMIN=2, GMAX=-2.
        - Having many gmin/gmax values is correctly labeled
        - Interior gmin/gmax are upgraded from min/max via isin check
        - We assess whether boundary points are min/max based on their neighbor (if < neighbor they its a min, elts its a max);
        - boundary maxima are discarded (they mess up persistence)
        - If gmax only appears at boundaries, we ignore it and promote next highest interior max to GMAX
        NOTE: keeping only keypoints indirectly de-duplicates the dataset, solving the 'plateaus' problem"""

        # Vectorized comparison with neighbors
        is_min = (self.x[1:-1] < self.x[:-2]) & (self.x[1:-1] < self.x[2:])
        is_max = (self.x[1:-1] > self.x[:-2]) & (self.x[1:-1] > self.x[2:])
        
        # Adjust indices because we sliced off the boundaries
        min_idx  = torch.nonzero(is_min).flatten() + 1
        max_idx  = torch.nonzero(is_max).flatten() + 1

        gmin_val = self.x.min()
        gmax_val = self.x.max()
        gmin_idx = torch.nonzero(self.x == gmin_val).flatten()
        gmax_idx = torch.nonzero(self.x == gmax_val).flatten()

        # upgrade min/max to gmin/gmax before concat
        min_types = torch.where(torch.isin(min_idx, gmin_idx),
                                torch.tensor(GMIN, device=self.device),
                                torch.tensor(MIN,  device=self.device))
        max_types = torch.where(torch.isin(max_idx, gmax_idx),
                                torch.tensor(GMAX, device=self.device),
                                torch.tensor(MAX,  device=self.device))

        # boundaries: keep only if min/gmin, discard if max/gmax
        boundary_indices, boundary_types_list = [], []
        for idx, neighbor in [(0, 1), (len(self.x) - 1, len(self.x) - 2)]:
            val = self.x[idx]
            if val == gmax_val:
                pass  # discard
            elif val < self.x[neighbor]:
                boundary_indices.append(idx)
                boundary_types_list.append(GMIN if val == gmin_val else MIN)
            # else: boundary is a non-global max → discard

        # promote next interior max to GMAX if boundary was gmax and no interior gmax exists
        left_was_gmax  = self.x[0]   == gmax_val
        right_was_gmax = self.x[-1]  == gmax_val
        boundary_stole_gmax = (left_was_gmax or right_was_gmax) and not torch.isin(max_idx, gmax_idx).any()
        if boundary_stole_gmax:
            interior_max_val = self.x[1:-1].max()
            promoted_idx     = torch.nonzero(self.x[1:-1] == interior_max_val).flatten() + 1
            max_types        = torch.where(torch.isin(max_idx, promoted_idx),
                                           torch.tensor(GMAX, device=self.device), max_types)

        if boundary_indices:
            b_idx     = torch.tensor(boundary_indices,    device=self.device)
            b_types   = torch.tensor(boundary_types_list, device=self.device)
            all_idx   = torch.cat([min_idx, max_idx, b_idx])
            all_types = torch.cat([min_types, max_types, b_types])
        else:
            all_idx   = torch.cat([min_idx, max_idx])
            all_types = torch.cat([min_types, max_types])

        sort_key  = all_idx * 10 - all_types.abs()  # GMIN/GMAX before MIN/MAX at same index. *10 > max type abs val (2), so index dominates
        order     = torch.argsort(sort_key)
        all_idx   = all_idx[order]
        all_types = all_types[order]

        mask = torch.cat([torch.tensor([True], device=self.device),
                        all_idx[1:] != all_idx[:-1]])

        self.keypoint_idx, self.keypoint_types = all_idx[mask], all_types[mask]
        return self.keypoint_idx, self.keypoint_types

    def init_persistence_tensors(self):
        """Function that initializes the persistence process. Creates persistence arrays for [GLOBAL]-MINIMA ONLY (maxima dont die). birth/death values are usually y-vals (can be idx)
        - persistence_idx:   original series index of each minimum
        - persistence_births: y-val at birth (known immediately)
        - persistence_deaths: y-val at death (inf until pairing fills it; stays inf for gmin)
        NOTE: we set all min points' persistence to ∞ (only gmin is correctly set to die at ∞), which need to be updated later"""

        # minima (they are born and die)
        min_mask                = (self.keypoint_types == MIN) | (self.keypoint_types == GMIN)
        min_idx                 = self.keypoint_idx[min_mask] #which indices are [g]-minima
        self.persistence_idx    = min_idx # consists of minima only
        self.persistence_births = self.x[min_idx] # birth y-val is known immediately for minima
        self.persistence_deaths = torch.full((min_mask.sum(),), float('inf'), device=self.device) # pre-fill
        self.active_min_idx     = self.persistence_idx # minima start as active (not yet paired)
        self.active_min_y       = self.persistence_births

        # maxima; they dont have persistence since they dont die, but kill minima
        max_mask                = (self.keypoint_types == MAX)
        max_idx                 = self.keypoint_idx[max_mask] #which indices are [g]-maxima
        self.max_yvals          = self.x[max_idx]
        self.active_max_idx     = max_idx.clone()
        self.active_max_y       = self.max_yvals.clone()

        # globals
        self.gmax_idx           = self.keypoint_idx[self.keypoint_types == GMAX]
        self.gmin_idx           = self.keypoint_idx[self.keypoint_types == GMIN]
        self.gmax_yval          = self.x[self.gmax_idx]
        self.gmin_yval          = self.x[self.gmin_idx]

    def compute_persistence_for_gmin_and_opposite_gmin(self):
        """1D sublevel β₀ persistence algo for 1d timeseries
        the # components = # min or max indices
        Step 1: find the lowest min on the opposite side of gmax from gmin → dies at gmax.
        Step 2: remove it and gmin from active arrays (gmin dies at inf, already set)."""

        # find active mins on OPPOSITE side of gmin (wrt to gmax)
        if self.gmin_idx < self.gmax_idx:
            opposite_mask = self.active_min_idx > self.gmax_idx  # gmin on left → look right
        else:
            opposite_mask = self.active_min_idx < self.gmax_idx  # gmin on right → look left

        opposite_idx   = self.active_min_idx[opposite_mask]  # series positions of opposite-side mins
        opposite_yvals = self.active_min_y[opposite_mask]    # their y-vals

        # lowest opposite-side min = other side's gmin → dies at gmax
        local_argmin      = torch.argmin(opposite_yvals)    # local position within opposite arrays
        opposite_gmin_idx = opposite_idx[local_argmin]      # its position in original series

        # assign death
        death_loc = (self.persistence_idx == opposite_gmin_idx).nonzero().flatten()[0]
        self.persistence_deaths[death_loc] = self.gmax_yval

        # remove both gmin and other_gmin from active arrays
        remove_mask         = ~torch.isin(self.active_min_idx,
                                          torch.tensor([opposite_gmin_idx, self.gmin_idx], device=self.device))
        self.active_min_idx = self.active_min_idx[remove_mask]
        self.active_min_y   = self.active_min_y[remove_mask]

    def _get_neighbors_for_1_max(self):
        argmin         = torch.argmin(self.active_max_y)  # local position within active arrays
        max_series_idx = self.active_max_idx[argmin]      # position in original series
        max_yval       = self.active_max_y[argmin]        # its yval

        # find adjacent mins: one to the left, one to the right of this max
        left_mask  = self.active_min_idx < max_series_idx
        right_mask = self.active_min_idx > max_series_idx
        left_idx   = self.active_min_idx[left_mask]
        right_idx  = self.active_min_idx[right_mask]

        # nearest neighbor on each side: gmax acts as natural barrier, so cross-side mins are never nearest
        left_neighbor  = left_idx[-1] if len(left_idx)  > 0 else None
        right_neighbor = right_idx[0] if len(right_idx) > 0 else None

        # pick higher min (lower persistence = dies first)
        if left_neighbor is None:
            victim_idx = right_neighbor
        elif right_neighbor is None:
            victim_idx = left_neighbor
        else:
            left_y     = self.x[left_neighbor]
            right_y    = self.x[right_neighbor]
            victim_idx = left_neighbor if left_y > right_y else right_neighbor

        # assign death
        death_loc = (self.persistence_idx == victim_idx).nonzero().flatten()[0]
        self.persistence_deaths[death_loc] = max_yval

        # consume victim min AND this max from active arrays
        min_remove = (self.active_min_idx != victim_idx)
        max_remove = (self.active_max_idx != max_series_idx)
        # min_remove = ~torch.isin(self.active_min_idx, torch.tensor([victim_idx], device=self.device))
        # max_remove = ~torch.isin(self.active_max_idx, torch.tensor([max_series_idx], device=self.device))
        self.active_min_idx = self.active_min_idx[min_remove]
        self.active_min_y   = self.active_min_y[min_remove]
        self.active_max_idx = self.active_max_idx[max_remove]
        self.active_max_y   = self.active_max_y[max_remove]


    def compute_persistence_for_other_minima(self):
        """Pair remaining mins with their lowest adjacent max (elder rule).
        Each iteration: start with lowest active max point, find 2 adjacent mins on either side (wrt gmax),
        kill the min point closest to it (in terms of y value), then remove both points from active arrays"""

        while len(self.active_max_idx) > 0: # keep going until we have no more active maxima
            # lowest active max is the next kill event
            argmin         = torch.argmin(self.active_max_y)  # local position within active arrays
            max_series_idx = self.active_max_idx[argmin]      # position in original series
            max_yval       = self.active_max_y[argmin]        # its yval

            # find adjacent mins: one to the left, one to the right of this max
            left_mask  = self.active_min_idx < max_series_idx
            right_mask = self.active_min_idx > max_series_idx
            left_idx   = self.active_min_idx[left_mask]
            right_idx  = self.active_min_idx[right_mask]

            # nearest neighbor on each side: gmax acts as natural barrier, so cross-side mins are never nearest
            left_neighbor  = left_idx[-1] if len(left_idx)  > 0 else None
            right_neighbor = right_idx[0] if len(right_idx) > 0 else None

            # pick higher min (lower persistence = dies first)
            if left_neighbor is None:
                victim_idx = right_neighbor
            elif right_neighbor is None:
                victim_idx = left_neighbor
            else:
                left_y     = self.x[left_neighbor]
                right_y    = self.x[right_neighbor]
                victim_idx = left_neighbor if left_y > right_y else right_neighbor

            # assign death
            death_loc = (self.persistence_idx == victim_idx).nonzero().flatten()[0]
            self.persistence_deaths[death_loc] = max_yval

            # consume victim min AND this max from active arrays
            min_remove = (self.active_min_idx != victim_idx)
            max_remove = (self.active_max_idx != max_series_idx)
            # min_remove = ~torch.isin(self.active_min_idx, torch.tensor([victim_idx], device=self.device))
            # max_remove = ~torch.isin(self.active_max_idx, torch.tensor([max_series_idx], device=self.device))
            self.active_min_idx = self.active_min_idx[min_remove]
            self.active_min_y   = self.active_min_y[min_remove]
            self.active_max_idx = self.active_max_idx[max_remove]
            self.active_max_y   = self.active_max_y[max_remove]


class IncrementalSublevel(Sublevel1D):
    """This class inherits from the sublevel class, and is used to handle incremented timeseries chunks"""
    def __init__(self, x_new, x: torch.Tensor = None):
        super().__init__(x)
        self.x_new = x_new

        self.inc_keypoint_idx   = None
        self.inc_keypoint_types = None
        # persistence arrays (one entry per minimum pt)
        self.inc_persistence_idx    = None # indices of each MINIMUM
        self.inc_persistence_births = None # = y-val
        self.inc_persistence_deaths = None # = y=val
        # active 
        self.inc_active_min_idx = None  # minima idx, consumed during pairing
        self.inc_active_min_y   = None  # minima y-val, consumed during pairing
        self.inc_active_max_idx = None  # maxima idx, consumed during pairing
        self.inc_active_max_y   = None  # maxima y-val, consumed during pairing
        # globals
        self.inc_gmax_idx  = None
        self.inc_gmax_yval = None
        self.inc_gmin_idx  = None
        self.inc_gmin_yval = None
        # i want to have attributes for the incremented timeseries!!

    def add_timeseries(self):
        """function to add new chunk to our timeseries"""
        self.x = torch.cat([self.x, self.x_new], dim=0)

    def update_persistence_with_new_series(self):
        """ok"""

    def handle_no_gmin_or_gmax():
        """Case for no new g-min or g-max."""
        self.inc_keypoint_idx, self.inc_keypoint_types = self.find_extrema_in_timeseries()

        self.inc_gmax_idx  = self.inc_keypoint_idx[self.inc_keypoint_types == GMAX]
        self.inc_gmin_idx  = self.inc_keypoint_idx[self.inc_keypoint_types == GMIN]
        self.inc_gmax_yval = self.x[self.inc_gmax_idx]
        self.inc_gmin_yval = self.x[self.inc_gmin_idx]

        if (self.inc_gmax_yval > self.gmax_yval):
            if (self.inc_gmin_yval < self.gmin_yval):
                print("new gmax and new gmin")
            else:
                print("new gmax but no new gmin")
        elif (self.inc_gmax_yval > self.gmax_yval):
            if (self.inc_gmin_yval < self.gmin_yval):
                print("new gmin but no new gmax")
            else:
                print("no new gmax or gmin")

        # apply persistence
        self.init_persistence_tensors()
        self.compute_persistence_for_gmin_and_opposite_gmin()
        self.compute_persistence_for_other_minima()

        # then concat persistence of old + new series
        # so here i want to concat 2 arrays, but how to get the new persistence as array? should the functions in the above lines output something or no (since theyre in self?)?

        # 0) if increment has no gmin or gmax
            # then we compute the increment's persistence and append it to the current one

        # 1) if increment has new gmin
            # 1) if new gmin is on opposite side of old gmin
                # new gmin > inf, old gmin > gmax
            # 2) if new gmin is on same side as old gmin
                # old gmin > nearby mamx [previouslt old gmin > inf]. Careful with this step
                # new gmin > inf
            # 3) if gmin is on boundary, treat it as normal point and proceed. find its position wrt gmax and work accordingly

        # 2) if increment has new gmax
            # 1) if new gmax is on opposite side of old gmin (wrt old gmax)
                # boundary shifted to new gmax
                # old gmin dies at inf [same as before]
                # new opposite lowest min (NOT gmin), dies at new gmax [new g-maximum will create new minima,
                    # unless it is at boundary, in which case it is not a proper g-max and we take the next highest max as g-max]
            # 2) if new gmax is on same side as old gmin (wrt old gmax)
                # boundary shifted to new gmax, so new points become on opposite side as gmin
                # old gmin > inf [same as before]
                # on newly created side of gmax (=right side), lowest min dies at new gmax
            # 3) if new gmax is on boundary
                # boundary max dont kill, so ignore it, treat the 2nd highest max as gmax

        # 3) if increment has new gmin AND new gmax
            # 1) if new gmin is on SAME side of old gmin
                # oldest lowest min op opposite side of old gmin killed by old gmax
                # new gmax kills lowest min on opposite side of new gmin [previously this old gmin > inf]
            # 2) if new gmin is on OPPOSITE side as old gmin
                # old gmin dies at gmax [prev this old gmin > inf]
                # new gmin > inf
