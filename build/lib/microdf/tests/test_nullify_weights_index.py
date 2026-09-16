import microdf as mdf


def test_nullify_weights_non_default_index():
    """nullify_weights must align to the index, not a fresh RangeIndex."""
    s = mdf.MicroSeries([1, 2, 3], index=[10, 11, 12], weights=[1, 2, 3])
    s.nullify_weights()
    assert s.sum() == 6
    assert s.mean() == 2
    assert list(s.weights.index) == [10, 11, 12]
