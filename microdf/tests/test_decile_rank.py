"""Test script for enhanced decile_rank functionality."""

import numpy as np

from microdf import MicroSeries


def test_decile_rank() -> None:
    """Test the enhanced decile_rank method with assert statements."""
    print("Running enhanced decile_rank tests...")

    # Create test data with some negative values
    test_data = [-5, -2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
    test_weights = [1] * len(test_data)
    test_ms = MicroSeries(test_data, weights=test_weights)

    default_deciles = test_ms.decile_rank()
    assert all(
        1 <= d <= 10 for d in default_deciles.values
    ), "Default deciles should be between 1 and 10"

    negative_indices = [i for i, val in enumerate(test_data) if val < 0]
    negative_deciles = [default_deciles.values[i] for i in negative_indices]
    assert all(
        d >= 1 for d in negative_deciles
    ), "Negative values should be ranked 1-10 in default mode"

    assert isinstance(
        default_deciles, MicroSeries
    ), "decile_rank should return MicroSeries"

    zero_deciles = test_ms.decile_rank(negatives_in_zero=True)
    negative_zero_deciles = [zero_deciles.values[i] for i in negative_indices]
    assert all(
        d == 0 for d in negative_zero_deciles
    ), "Negative values should be in decile 0 when negatives_in_zero=True"

    non_negative_indices = [i for i, val in enumerate(test_data) if val >= 0]
    non_negative_deciles = [
        zero_deciles.values[i] for i in non_negative_indices
    ]
    assert all(
        1 <= d <= 10 for d in non_negative_deciles
    ), "Non-negative values should be ranked 1-10 when negatives_in_zero=True"

    unique_deciles = sorted(zero_deciles.unique())
    assert 0 in unique_deciles, "Decile 0 should exist with negative values"
    assert all(
        d in unique_deciles for d in range(1, 11)
    ), "Deciles 1-10 should exist with non-negative values"

    positive_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    positive_ms = MicroSeries(positive_data, weights=[1] * 10)

    pos_default = positive_ms.decile_rank()
    pos_zero = positive_ms.decile_rank(negatives_in_zero=True)
    assert np.array_equal(
        pos_default.values, pos_zero.values
    ), "Both modes should produce identical results for all positive data"
    assert (
        0 not in pos_zero.unique()
    ), "Decile 0 should not exist when all values are positive"

    negative_data = [-10, -8, -6, -4, -2]
    negative_ms = MicroSeries(negative_data, weights=[1] * 5)

    neg_default = negative_ms.decile_rank()
    neg_zero = negative_ms.decile_rank(negatives_in_zero=True)
    assert all(
        1 <= d <= 10 for d in neg_default.values
    ), "Default mode should rank all negative values 1-10"
    assert all(
        d == 0 for d in neg_zero.values
    ), "All negative values should be in decile 0 when negatives_in_zero=True"

    weighted_data = [-1, 0, 1, 2, 3]
    weighted_weights = [0.5, 1.0, 1.5, 2.0, 2.5]
    weighted_ms = MicroSeries(weighted_data, weights=weighted_weights)

    w_default = weighted_ms.decile_rank()
    w_zero = weighted_ms.decile_rank(negatives_in_zero=True)
    assert np.array_equal(
        w_default.weights.values, weighted_weights
    ), "Weights should be preserved in default mode"
    assert np.array_equal(
        w_zero.weights.values, weighted_weights
    ), "Weights should be preserved in negatives_in_zero mode"


if __name__ == "__main__":
    test_decile_rank()
