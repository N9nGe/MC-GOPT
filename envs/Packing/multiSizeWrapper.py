"""
Wrapper for multi-size bin training.
Enables training on multiple bin sizes by randomly selecting a bin size on each reset.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .container import Container


class MultiSizeWrapper(gym.Wrapper):
    """
    Wrapper that randomly changes bin size on each reset.
    Handles different observation dimensions by padding to maximum size.

    Args:
        env: The base PackingEnv environment
        bin_sizes: List of bin size tuples, e.g., [[10,10,10], [20,20,20], [30,30,30]]
    """

    def __init__(self, env, bin_sizes=None):
        super().__init__(env)
        self.bin_sizes = bin_sizes or [[10, 10, 10], [20, 20, 20], [30, 30, 30]]
        self.current_bin_size = None
        self.max_bin_size = None
        self.max_area = None
        self._compute_max_dimensions()
        self._update_observation_space()

    def _compute_max_dimensions(self):
        """Compute maximum dimensions across all bin sizes"""
        self.max_bin_size = [max(b[i] for b in self.bin_sizes) for i in range(3)]
        self.max_area = self.max_bin_size[0] * self.max_bin_size[1]
        print(f"MultiSizeWrapper initialized with bin_sizes: {self.bin_sizes}")
        print(f"Max bin size: {self.max_bin_size}, max area: {self.max_area}")

    def _update_observation_space(self):
        """Update observation space to accommodate maximum dimensions"""
        # Observation: heightmap + item_size + candidates
        # heightmap: max_area elements
        # item_size: 6 elements (3 for original + 3 for rotated)
        # candidates: k_placement * 6 for EMS scheme (or * 3 for others)
        if self.env.unwrapped.action_scheme == "EMS":
            candidate_size = self.env.unwrapped.k_placement * 6
        else:
            candidate_size = self.env.unwrapped.k_placement * 3

        obs_len = self.max_area + 6 + candidate_size

        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=0,
                high=max(self.max_bin_size),
                shape=(obs_len,),
                dtype=np.float32
            ),
            "mask": spaces.Discrete(self.env.k_placement)
        })

        print(f"Observation space updated: obs_len={obs_len} (area={self.max_area}, "
              f"item=6, candidates={candidate_size})")

    def reset(self, seed=None, **kwargs):
        """
        Reset environment with a randomly selected bin size.
        Pads observation to max size.
        """
        # Randomly select bin size
        idx = np.random.randint(len(self.bin_sizes))
        self.current_bin_size = self.bin_sizes[idx]

        # Update environment's bin size and area
        # Use unwrapped to access the actual PackingEnv instance
        self.env.unwrapped.bin_size = tuple(self.current_bin_size)
        self.env.unwrapped.area = int(self.current_bin_size[0] * self.current_bin_size[1])

        # Call original reset (this creates new Container with updated bin_size)
        obs, info = self.env.reset(seed=seed, **kwargs)

        # Pad observation to max size
        padded_obs = self._pad_observation(obs)

        # Add bin size info to info dict for logging
        info['bin_size'] = self.current_bin_size

        return padded_obs, info

    def _pad_observation(self, obs):
        """
        Pad observation to maximum dimensions.

        Original obs structure:
        - heightmap: area elements (current bin size)
        - item_size: 6 elements
        - candidates: k_placement * 6 (or * 3) elements

        Padded obs structure:
        - heightmap: max_area elements (padded with zeros)
        - item_size: 6 elements (unchanged)
        - candidates: k_placement * 6 (unchanged)
        """
        original_obs = obs["obs"]
        original_mask = obs["mask"]

        # Calculate current area from ACTUAL environment area after reset
        # The env.area should now be correct since we updated it before calling reset
        current_area = self.env.unwrapped.area

        if self.env.unwrapped.action_scheme == "EMS":
            candidate_size = self.env.unwrapped.k_placement * 6
        else:
            candidate_size = self.env.unwrapped.k_placement * 3

        # Debug: check observation size
        expected_obs_len = current_area + 6 + candidate_size
        if len(original_obs) != expected_obs_len:
            print(f"WARNING: Observation size mismatch!")
            print(f"  Expected: {expected_obs_len} (area={current_area} + item=6 + candidates={candidate_size})")
            print(f"  Actual: {len(original_obs)}")
            print(f"  Current bin_size: {self.current_bin_size}")
            print(f"  Env bin_size: {self.env.unwrapped.bin_size}")
            print(f"  Env area: {self.env.unwrapped.area}")
            # Use actual observation length to extract heightmap
            # The observation might not have been reset properly
            actual_area = len(original_obs) - 6 - candidate_size
            if actual_area > 0:
                current_area = actual_area
                print(f"  Using actual_area: {actual_area}")

        # Extract components
        heightmap = original_obs[:current_area]
        item_and_candidates = original_obs[current_area:]  # item (6) + candidates

        # Pad heightmap to max_area
        padded_heightmap = np.zeros(self.max_area, dtype=np.float32)
        padded_heightmap[:len(heightmap)] = heightmap  # Use actual heightmap length

        # Concatenate padded heightmap with item and candidates
        padded_obs = np.concatenate([
            padded_heightmap,
            item_and_candidates
        ])

        return {
            "obs": padded_obs,
            "mask": original_mask
        }

    def step(self, action):
        """
        Step environment and pad observation.
        """
        obs, reward, done, truncated, info = self.env.step(action)
        padded_obs = self._pad_observation(obs)

        # Add bin size to info
        info['bin_size'] = self.current_bin_size

        return padded_obs, reward, done, truncated, info

    def get_current_bin_size(self):
        """Get the current bin size being used"""
        return self.current_bin_size
