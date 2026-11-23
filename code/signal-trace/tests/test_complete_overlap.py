import unittest

import numpy as np

from pipeline.overlap.complete_overlap import distance_path, member_means, native_targets, validate_pairs


class CompleteCorrespondenceTests(unittest.TestCase):
    def test_positive_similarity_does_not_reward_extra_path_steps(self):
        likeness = np.full((4, 4), 0.8)
        np.fill_diagonal(likeness, 1.0)
        np.testing.assert_array_equal(distance_path(likeness), np.column_stack((np.arange(4), np.arange(4))))

    def test_all_native_steps_contribute_when_axis_is_shorter(self):
        pairs = distance_path([[1, 0], [1, 0], [0, 1]])
        np.testing.assert_array_equal(pairs, [[0, 0], [1, 0], [2, 1]])
        np.testing.assert_allclose(member_means(np.array([[1.], [3.], [9.]]), pairs, 2), [[2.], [9.]])
        np.testing.assert_allclose(native_targets(np.array([[10.], [20.]]), pairs, 3), [[10.], [10.], [20.]])

    def test_repeated_step_gets_one_target_not_multiple_fit_rows(self):
        pairs = distance_path([[1, 1, 0], [0, 0, 1]])
        np.testing.assert_array_equal(pairs, [[0, 0], [0, 1], [1, 2]])
        targets = native_targets(np.array([[1., 3.], [3., 5.], [7., 9.]]), pairs, 2)
        np.testing.assert_allclose(targets, [[2., 4.], [7., 9.]])
        self.assertEqual(targets.shape[0], 2)

    def test_members_have_equal_weight_despite_different_local_counts(self):
        long = member_means(np.array([[0.], [2.], [4.]]), np.array([[0, 0], [1, 0], [2, 1]]), 2)
        short = member_means(np.array([[9.], [6.]]), np.array([[0, 0], [1, 1]]), 2)
        np.testing.assert_allclose((long + short) / 2, [[5.], [5.]])

    def test_missing_native_step_is_not_a_complete_path(self):
        with self.assertRaises(ValueError):
            validate_pairs(np.array([[0, 0], [2, 1]]), 3, 2)

    def test_the_path_is_the_cell_by_cell_loop_s_path_ties_included(self):
        def by_cells(likeness):   # the loop distance_path was until November 15, 2026: the path's definition, kept here as what it is held to
            cost = 1 - np.clip(np.asarray(likeness, dtype=np.float64), -1, 1)
            n, m = cost.shape
            total = np.full((n + 1, m + 1), np.inf)
            total[0, 0] = 0
            move = np.zeros((n, m), dtype=np.int8)
            for i in range(n):
                for j in range(m):
                    diagonal, vertical, horizontal = total[i, j], total[i, j + 1], total[i + 1, j]
                    if diagonal <= vertical and diagonal <= horizontal:
                        previous, direction = diagonal, 0
                    elif vertical <= horizontal:
                        previous, direction = vertical, 1
                    else:
                        previous, direction = horizontal, 2
                    total[i + 1, j + 1] = previous + cost[i, j]
                    move[i, j] = direction
            pairs, i, j = [], n - 1, m - 1
            while i >= 0 and j >= 0:
                pairs.append((i, j))
                i, j = (i - 1, j - 1) if move[i, j] == 0 else (i - 1, j) if move[i, j] == 1 else (i, j - 1)
            return np.asarray(pairs[::-1], dtype=np.int32)

        rng = np.random.default_rng(0)
        shapes = [(1, 1), (1, 9), (9, 1), (2, 2), (40, 40), (17, 31), (31, 17)] + [tuple(rng.integers(1, 36, size=2)) for _ in range(60)]
        for n, m in shapes:
            smooth = rng.uniform(-1.2, 1.2, size=(n, m))                 # past the clip on both sides
            coarse = rng.integers(-2, 3, size=(n, m)) / 2                # five values: nearly every cell is a tie of two or three ways
            flat = np.full((n, m), 0.5)                                  # every way ties everywhere
            for likeness in (smooth, coarse, flat, smooth.astype(np.float32)):
                ours = distance_path(likeness)
                np.testing.assert_array_equal(ours, by_cells(likeness))
                self.assertEqual(ours.dtype, np.int32)

    def test_invalid_similarity_is_rejected(self):
        for likeness in (np.empty((0, 2)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                distance_path(likeness)


if __name__ == "__main__":
    unittest.main()
