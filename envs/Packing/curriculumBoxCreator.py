"""
Curriculum-based box creator for 3D bin packing.
Adjusts difficulty by biasing box size distribution.
"""

import numpy as np
from .binCreator import RandomBoxCreator
from curriculum import difficulty_to_box_size_bias


class CurriculumBoxCreator(RandomBoxCreator):
    """
    Box creator with curriculum learning support.

    Adjusts sampling distribution based on difficulty:
    - Low difficulty (0.0-0.4): Prefer large boxes (easier packing)
    - Medium difficulty (0.4-0.6): Uniform distribution
    - High difficulty (0.6-1.0): Prefer small boxes or uniform (harder packing)
    """

    def __init__(self, box_size_set=None):
        """
        Args:
            box_size_set: List of available box sizes (tuples)
        """
        super().__init__(box_size_set)
        self.difficulty = 0.5  # Default: medium difficulty
        self.weights = None  # Sampling weights
        self._update_weights()

        # Stats for logging
        self.total_boxes_generated = 0
        self.difficulty_history = []

        print(f"CurriculumBoxCreator initialized with {len(self.box_set)} box sizes")

    def set_difficulty(self, difficulty: float):
        """
        Set target difficulty level and update sampling weights.

        Args:
            difficulty: Target difficulty in [0.0, 1.0]
                       0.0 = easiest (large boxes)
                       1.0 = hardest (small boxes)
        """
        self.difficulty = np.clip(difficulty, 0.0, 1.0)
        self._update_weights()
        self.difficulty_history.append(self.difficulty)

    def _update_weights(self):
        """Update sampling weights based on current difficulty"""
        self.weights = difficulty_to_box_size_bias(self.difficulty, self.box_set)

    def generate_box_size(self, **kwargs):
        """
        Generate a box size using weighted sampling based on difficulty.
        """
        # Sample index according to weights
        idx = np.random.choice(len(self.box_set), p=self.weights)
        self.box_list.append(self.box_set[idx])
        self.total_boxes_generated += 1

    def get_stats(self) -> dict:
        """Get statistics about box generation"""
        if self.total_boxes_generated == 0:
            avg_volume = 0
        else:
            volumes = [b[0] * b[1] * b[2] for b in self.box_list]
            avg_volume = np.mean(volumes) if volumes else 0

        return {
            'total_boxes': self.total_boxes_generated,
            'current_difficulty': self.difficulty,
            'avg_difficulty': np.mean(self.difficulty_history) if self.difficulty_history else 0.5,
            'avg_box_volume': avg_volume,
        }

    def reset(self):
        """Reset box list (called at episode start)"""
        super().reset()
        # Note: We don't reset difficulty or weights on episode reset
        # They are controlled by the curriculum scheduler


# Global instance for sharing difficulty across subprocesses
_GLOBAL_DIFFICULTY = 0.5
_DIFFICULTY_LOCK = None


def set_global_difficulty(difficulty: float):
    """
    Set global difficulty target.
    Can be called from training loop to update all environments.

    Args:
        difficulty: Target difficulty [0.0, 1.0]
    """
    global _GLOBAL_DIFFICULTY
    _GLOBAL_DIFFICULTY = np.clip(difficulty, 0.0, 1.0)


def get_global_difficulty() -> float:
    """Get current global difficulty"""
    return _GLOBAL_DIFFICULTY


class GlobalCurriculumBoxCreator(CurriculumBoxCreator):
    """
    Curriculum box creator that reads from global difficulty.
    Useful for parallel environments in SubprocVectorEnv.
    """

    def __init__(self, box_size_set=None):
        super().__init__(box_size_set)
        print("GlobalCurriculumBoxCreator: Will read difficulty from global state")

    def generate_box_size(self, **kwargs):
        """Generate box using global difficulty"""
        # Update difficulty from global state
        global_diff = get_global_difficulty()
        if global_diff != self.difficulty:
            self.set_difficulty(global_diff)

        # Generate box with updated weights
        super().generate_box_size(**kwargs)


if __name__ == "__main__":
    # Test curriculum box creator
    print("=== Testing CurriculumBoxCreator ===\n")

    # Create box set
    box_set = []
    for i in range(1, 10, 2):
        for j in range(1, 10, 2):
            for k in range(1, 10, 2):
                box_set.append((i, j, k))

    print(f"Box set size: {len(box_set)}")
    print(f"Box volumes range: {min(b[0]*b[1]*b[2] for b in box_set)} "
          f"to {max(b[0]*b[1]*b[2] for b in box_set)}\n")

    # Create curriculum box creator
    creator = CurriculumBoxCreator(box_set)

    # Test different difficulty levels
    for difficulty in [0.0, 0.3, 0.5, 0.7, 1.0]:
        print(f"\n--- Difficulty: {difficulty:.1f} ---")
        creator.set_difficulty(difficulty)

        # Generate 20 boxes
        boxes = []
        for _ in range(20):
            creator.generate_box_size()
            boxes.append(creator.box_list[-1])

        # Analyze generated boxes
        volumes = [b[0] * b[1] * b[2] for b in boxes]
        print(f"Generated {len(boxes)} boxes")
        print(f"  Avg volume: {np.mean(volumes):.1f} ± {np.std(volumes):.1f}")
        print(f"  Volume range: [{min(volumes)}, {max(volumes)}]")

        # Count large vs small boxes
        large_boxes = sum(1 for v in volumes if v > 100)
        small_boxes = sum(1 for v in volumes if v < 30)
        print(f"  Large boxes (>100): {large_boxes}/20")
        print(f"  Small boxes (<30): {small_boxes}/20")

        creator.reset()  # Clear for next test

    # Print stats
    print("\n=== Creator Stats ===")
    print(creator.get_stats())
