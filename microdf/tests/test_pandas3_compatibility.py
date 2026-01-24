"""
Tests for pandas 3.0.0 compatibility in microdf.

These tests verify that microdf works correctly with pandas 3.0.0,
which introduces:
1. PyArrow-backed strings as default (StringDtype)
2. Copy-on-Write by default
3. Changes to how Series subclasses are handled
"""

import numpy as np
import pandas as pd
import pytest

from microdf.microseries import MicroSeries
from microdf.microdataframe import MicroDataFrame


class TestMicroSeriesSubclassPreservation:
    """Test that MicroSeries subclass is preserved across operations."""

    def test_microseries_set_weights_after_creation(self):
        """
        Ensure set_weights works on MicroSeries.
        This is the error reported in pandas 3:
        AttributeError: 'Series' object has no attribute 'set_weights'
        """
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 1.0, 1.0]))
        assert hasattr(ms, "set_weights")
        assert hasattr(ms, "weights")

        # Should be able to call set_weights
        ms.set_weights(np.array([2.0, 2.0, 2.0]))
        assert np.allclose(ms.weights, [2.0, 2.0, 2.0])

    def test_microseries_preserved_after_arithmetic(self):
        """
        Arithmetic operations should return MicroSeries, not plain Series.
        """
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 2.0, 3.0]))

        # Addition
        result = ms + 1
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"
        assert hasattr(result, "weights")
        assert hasattr(result, "set_weights")

        # Multiplication
        result = ms * 2
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"

        # Division
        result = ms / 2
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"

    def test_microseries_preserved_after_comparison(self):
        """
        Comparison operations should return MicroSeries, not plain Series.
        """
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 2.0, 3.0]))

        # Greater than
        result = ms > 1
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"
        assert hasattr(result, "weights")

        # Less than
        result = ms < 3
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"

    def test_microseries_preserved_after_indexing(self):
        """
        Indexing operations should return MicroSeries, not plain Series.
        """
        ms = MicroSeries([1, 2, 3, 4, 5], weights=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        # Boolean indexing
        result = ms[ms > 2]
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"
        assert hasattr(result, "weights")

        # Slice indexing
        result = ms[1:3]
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"


class TestMicroDataFrameSubclassPreservation:
    """Test that MicroDataFrame column access returns MicroSeries."""

    def test_microdataframe_column_returns_microseries(self):
        """
        Accessing a column from MicroDataFrame should return MicroSeries.
        """
        mdf = MicroDataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            weights=np.array([1.0, 2.0, 3.0])
        )

        # Column access
        col = mdf["a"]
        assert isinstance(col, MicroSeries), f"Got {type(col)} instead of MicroSeries"
        assert hasattr(col, "weights")
        assert hasattr(col, "set_weights")

    def test_microdataframe_operations_preserve_type(self):
        """
        Operations on MicroDataFrame columns should preserve MicroSeries type.
        """
        mdf = MicroDataFrame(
            {"a": [1, 2, 3], "b": [4, 5, 6]},
            weights=np.array([1.0, 2.0, 3.0])
        )

        # Column operations
        result = mdf["a"] + mdf["b"]
        assert isinstance(result, MicroSeries), f"Got {type(result)} instead of MicroSeries"
        assert hasattr(result, "weights")


class TestStringDtypeHandling:
    """Test that MicroSeries/MicroDataFrame handle pandas 3 string dtypes."""

    def test_microseries_with_string_data(self):
        """
        MicroSeries should work with string data in pandas 3.
        """
        # Create with string data
        ms = MicroSeries(["a", "b", "c"], weights=np.array([1.0, 2.0, 3.0]))
        assert len(ms) == 3
        assert hasattr(ms, "weights")

    def test_microdataframe_with_string_columns(self):
        """
        MicroDataFrame should work with string columns in pandas 3.
        """
        mdf = MicroDataFrame(
            {"names": ["alice", "bob", "charlie"], "values": [1, 2, 3]},
            weights=np.array([1.0, 2.0, 3.0])
        )
        assert len(mdf) == 3

        # String column access should still work
        names = mdf["names"]
        assert len(names) == 3


class TestWeightedOperationsWithPandas3:
    """Test that weighted operations work correctly with pandas 3."""

    def test_weighted_sum(self):
        """Weighted sum should work correctly."""
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 2.0, 3.0]))
        # Weighted sum: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        assert ms.sum() == 14

    def test_weighted_mean(self):
        """Weighted mean should work correctly."""
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 2.0, 3.0]))
        # Weighted mean: (1*1 + 2*2 + 3*3) / (1 + 2 + 3) = 14 / 6 ≈ 2.333
        assert np.isclose(ms.mean(), 14 / 6)

    def test_weighted_count(self):
        """Weighted count should return sum of weights."""
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 2.0, 3.0]))
        assert ms.count() == 6.0


class TestCopyOnWriteCompatibility:
    """Test compatibility with pandas 3 Copy-on-Write."""

    def test_microseries_copy_independent(self):
        """
        Copying a MicroSeries should create an independent copy.
        """
        ms = MicroSeries([1, 2, 3], weights=np.array([1.0, 2.0, 3.0]))
        ms_copy = ms.copy()

        # Modify original
        ms.set_weights(np.array([4.0, 5.0, 6.0]))

        # Copy should be unchanged
        assert np.allclose(ms_copy.weights, [1.0, 2.0, 3.0])

    def test_microdataframe_copy_independent(self):
        """
        Copying a MicroDataFrame should create an independent copy.
        """
        mdf = MicroDataFrame(
            {"a": [1, 2, 3]},
            weights=np.array([1.0, 2.0, 3.0])
        )
        mdf_copy = mdf.copy()

        # Modify original
        mdf.set_weights(np.array([4.0, 5.0, 6.0]))

        # Copy should be unchanged
        assert np.allclose(mdf_copy.weights, [1.0, 2.0, 3.0])


class TestGroupByWithPandas3:
    """Test groupby operations with pandas 3."""

    def test_microseries_groupby_preserves_weights(self):
        """
        GroupBy operations should preserve weights.
        """
        ms = MicroSeries([1, 2, 3, 4], weights=np.array([1.0, 2.0, 3.0, 4.0]))
        groups = pd.Series(["a", "a", "b", "b"])

        gb = ms.groupby(groups)
        # Should be able to call weighted operations
        result = gb.sum()
        # Group a: 1*1 + 2*2 = 5
        # Group b: 3*3 + 4*4 = 25
        assert result["a"] == 5
        assert result["b"] == 25

    def test_microdataframe_groupby_preserves_weights(self):
        """
        MicroDataFrame groupby should preserve weights on columns.
        """
        mdf = MicroDataFrame(
            {"group": ["a", "a", "b", "b"], "value": [1, 2, 3, 4]},
            weights=np.array([1.0, 2.0, 3.0, 4.0])
        )

        gb = mdf.groupby("group")
        result = gb.sum()

        # Check that weighted sum was computed
        assert "value" in result.columns
