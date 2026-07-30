"[INEFFICIENT CODE] sublevel persistence using water lake analogy, no numba"

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MIN, MAX, GMIN, GMAX = 1, -1, 2, -2 # int encodings for keypoints

class PersistenceLake1D:
    """An optimized Disjoint-Set (Union-Find) structure customized for 
    1D Topological Data Analysis (Sublevel Set Filtration / Elder Rule)."""
    def __init__(self, keypoint_series_idx: torch.Tensor, device: torch.device):
        num_keypoints = len(keypoint_series_idx)
        self.basin_membership_ids = torch.arange(num_keypoints, device=device) 
        self.keypoint_idx_array   = keypoint_series_idx.clone().to(device) 

    def find_root_sequence_rank(self, keypoint_sequence_rank: int) -> int:
        """Finds the root component sequence rank using iterative path compression."""
        cursor_rank = keypoint_sequence_rank
        while cursor_rank != self.basin_membership_ids[cursor_rank]:
            self.basin_membership_ids[cursor_rank] = self.basin_membership_ids[self.basin_membership_ids[cursor_rank]]
            cursor_rank = self.basin_membership_ids[cursor_rank]
        return int(cursor_rank)

    def merge_basins_via_elder_rule(self, left_sequence_rank: int, right_sequence_rank: int, timeseries_values: torch.Tensor):
        """Merges two adjacent components using the TDA Elder Rule.
        Returns the (victim_birth_series_idx) of the feature that dies."""
        left_root_rank  = self.find_root_sequence_rank(left_sequence_rank)
        right_root_rank = self.find_root_sequence_rank(right_sequence_rank)

        if left_root_rank == right_root_rank:
            return None

        left_birth_height  = timeseries_values[self.keypoint_idx_array[left_root_rank]]
        right_birth_height = timeseries_values[self.keypoint_idx_array[right_root_rank]]

        if left_birth_height > right_birth_height:
            victim_birth_series_idx = self.keypoint_idx_array[left_root_rank]
            self.basin_membership_ids[left_root_rank] = right_root_rank
            return victim_birth_series_idx
        else:
            victim_birth_series_idx = self.keypoint_idx_array[right_root_rank]
            self.basin_membership_ids[right_root_rank] = left_root_rank
            return victim_birth_series_idx

class OnlineSublevelPersistenceLake:
    """Maintains a persistent, incrementally updated Lake Registry across data streams."""
    def __init__(self, initial_x: torch.Tensor, initial_keypoint_idx: torch.Tensor, initial_keypoint_types: torch.Tensor):
        self.device = initial_x.device
        self.history_values = initial_x.clone()
        
        # Instantiate and populate historical state in the registry
        self.lake_registry = PersistenceLake1D(initial_keypoint_idx, self.device)
        self.num_keypoints = len(initial_keypoint_idx)
        
        # Persist a clean, running array of keypoint types matching the registry sizes
        self.keypoint_types = initial_keypoint_types.clone().to(self.device)
        
        # Track which keypoint positions are currently active (submerged)
        self.submerged_components = torch.zeros(self.num_keypoints, dtype=torch.bool, device=self.device)
        
        # Run a batch sweep over the initial baseline data to catch up state
        self._sweep_increment(range(self.num_keypoints))

    def _sweep_increment(self, sequence_ranks_to_process: list):
        """Sweeps a specific subset of sorted sequence ranks and emits persistence pairs."""
        new_pairs = []
        
        # Sort incoming sequence ranks by their actual time-series height values
        heights = self.history_values[self.lake_registry.keypoint_idx_array[sequence_ranks_to_process]]
        sorted_indices = torch.argsort(heights)
        
        for idx in sorted_indices.tolist():
            active_sequence_rank = sequence_ranks_to_process[idx]
            # Clean, direct lookup from our tracking tensor
            keypoint_type        = self.keypoint_types[active_sequence_rank].item()
            active_series_idx    = self.lake_registry.keypoint_idx_array[active_sequence_rank]

            if keypoint_type in (MIN, GMIN):
                self.submerged_components[active_sequence_rank] = True

            elif keypoint_type in (MAX, GMAX):
                left_rank  = active_sequence_rank - 1
                right_rank = active_sequence_rank + 1

                has_left  = (left_rank >= 0) and self.submerged_components[self.lake_registry.find_root_sequence_rank(left_rank)]
                has_right = (right_rank < self.num_keypoints) and self.submerged_components[self.lake_registry.find_root_sequence_rank(right_rank)]

                if has_left and has_right:
                    victim_birth_idx = self.lake_registry.merge_basins_via_elder_rule(left_rank, right_rank, self.history_values)
                    if victim_birth_idx is not None:
                        new_pairs.append((int(victim_birth_idx.item()), int(active_series_idx.item())))
                        
                elif has_left:
                    self.lake_registry.basin_membership_ids[active_sequence_rank] = self.lake_registry.find_root_sequence_rank(left_rank)
                    self.submerged_components[active_sequence_rank] = True
                elif has_right:
                    self.lake_registry.basin_membership_ids[active_sequence_rank] = self.lake_registry.find_root_sequence_rank(right_rank)
                    self.submerged_components[active_sequence_rank] = True
                    
        return new_pairs

    def handle_boundary_point(self, new_x_chunk: torch.Tensor):
        """Evaluates and stitches the 4-point stitching window between history and stream."""
        old_len = len(self.history_values)
        if old_len < 2 or len(new_x_chunk) < 2:
            return None, None
            
        pts = [
            self.history_values[-2].item(),
            self.history_values[-1].item(),
            new_x_chunk[0].item(),
            new_x_chunk[1].item()]
        
        old_last_type = None
        if pts[1] < pts[0] and pts[1] < pts[2]:
            old_last_type = MIN
        elif pts[1] > pts[0] and pts[1] > pts[2]:
            old_last_type = MAX
            
        new_first_type = None
        if pts[2] < pts[1] and pts[2] < pts[3]:
            new_first_type = MIN
        elif pts[2] > pts[1] and pts[2] > pts[3]:
            new_first_type = MAX
            
        return old_last_type, new_first_type

    def append_stream_data(self, new_x_chunk: torch.Tensor, stream_keypoint_idx: torch.Tensor, stream_keypoint_types: torch.Tensor):
        """Appends streaming data chunk, updates tracking structures, and emits new pairs."""
        old_history_len = len(self.history_values)
        
        # 1. Update full historical values tracking array immediately
        self.history_values = torch.cat([self.history_values, new_x_chunk])
        
        # Shift incoming stream index markers to align with global time tracker
        adjusted_stream_idx = stream_keypoint_idx + old_history_len
        
        # 2. Update persistent global type tensor
        self.keypoint_types = torch.cat([self.keypoint_types, stream_keypoint_types.to(self.device)])
        
        new_num_keypoints = self.num_keypoints + len(adjusted_stream_idx)
        
        # Resize registry tensors dynamically
        extended_membership = torch.arange(new_num_keypoints, device=self.device)
        extended_membership[:self.num_keypoints] = self.lake_registry.basin_membership_ids
        self.lake_registry.basin_membership_ids = extended_membership
        
        self.lake_registry.keypoint_idx_array = torch.cat([self.lake_registry.keypoint_idx_array, adjusted_stream_idx])
        
        # Expand submerged state buffer
        extended_submerged = torch.zeros(new_num_keypoints, dtype=torch.bool, device=self.device)
        extended_submerged[:self.num_keypoints] = self.submerged_components
        self.submerged_components = extended_submerged
        
        # Determine sequence positions requiring processing
        start_rank = self.num_keypoints
        self.num_keypoints = new_num_keypoints
        ranks_to_process = list(range(start_rank, self.num_keypoints))
        
        # 3. Sweep exclusively over the incremental additions
        new_pairs = self._sweep_increment(ranks_to_process)
        
        return new_pairs


def compute_1d_sublevel_persistence_lake(timeseries_values: torch.Tensor, keypoint_series_idx: torch.Tensor, keypoint_types: torch.Tensor):
    device        = timeseries_values.device
    num_keypoints = len(keypoint_series_idx)

    # Calculate heights and sort them entirely on the current device
    keypoint_heights    = timeseries_values[keypoint_series_idx]
    sweep_line_schedule = torch.argsort(keypoint_heights)

    lake_registry        = PersistenceLake1D(keypoint_series_idx, device)
    submerged_components = torch.zeros(num_keypoints, dtype=torch.bool, device=device)
    
    birth_death_pairs_list = []

    # SPEEDUP FIX 1: Convert type matching to integer operations on the GPU/CPU directly
    # This lets us skip slow Python string/tuple checks inside the loop
    is_min_mask = (keypoint_types == MIN) | (keypoint_types == GMIN)
    is_max_mask = (keypoint_types == MAX) | (keypoint_types == GMAX)

    # SPEEDUP FIX 2: Convert to a list of integers once so the python loop iterates at bare-metal speeds
    schedule_list = sweep_line_schedule.tolist()

    for active_sequence_rank in schedule_list:
        # Quick tensor lookups on the target device
        active_series_idx = keypoint_series_idx[active_sequence_rank]

        if is_min_mask[active_sequence_rank]:  
            submerged_components[active_sequence_rank] = True

        elif is_max_mask[active_sequence_rank]:  
            left_rank  = active_sequence_rank - 1
            right_rank = active_sequence_rank + 1

            # Clean inline evaluations
            has_left  = (left_rank >= 0) and submerged_components[lake_registry.find_root_sequence_rank(left_rank)]
            has_right = (right_rank < num_keypoints) and submerged_components[lake_registry.find_root_sequence_rank(right_rank)]

            if has_left and has_right:
                victim_birth_series_idx = lake_registry.merge_basins_via_elder_rule(
                    left_rank, right_rank, timeseries_values)
                if victim_birth_series_idx is not None:
                    # Append natively as integers directly
                    birth_death_pairs_list.append((int(victim_birth_series_idx), int(active_series_idx))) 

            elif has_left:
                lake_registry.basin_membership_ids[active_sequence_rank] = lake_registry.find_root_sequence_rank(left_rank)
                submerged_components[active_sequence_rank] = True
            
            elif has_right:
                lake_registry.basin_membership_ids[active_sequence_rank] = lake_registry.find_root_sequence_rank(right_rank)
                submerged_components[active_sequence_rank] = True
                
    return birth_death_pairs_list

