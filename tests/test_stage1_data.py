import numpy as np
from numpy.testing import assert_array_equal

from research.run_stage1 import unique_paired_indices


def test_exact_paired_duplicates_keep_first_original_row_in_original_order():
    x = np.array([[9], [1], [9], [2], [1]])
    y = np.array([[3], [4], [3], [5], [4]])
    assert_array_equal(unique_paired_indices(x, y), [0, 1, 3])


def test_matching_only_one_view_does_not_count_as_identical_paired_object():
    x = np.array([[1], [1], [2]])
    y = np.array([[1], [2], [2]])
    assert_array_equal(unique_paired_indices(x, y), [0, 1, 2])
