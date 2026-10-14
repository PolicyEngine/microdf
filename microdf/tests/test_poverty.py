"""Hand-calculated poverty estimators for issue #334."""

import pytest
import microdf as mdf


@pytest.mark.parametrize(
    "method,expected",
    [
        ("poverty_rate", 2 / 5),
        ("deep_poverty_rate", 2 / 5),
        ("poverty_gap", 160),
        ("deep_poverty_gap", 60),
        ("squared_poverty_gap", 12800),
    ],
)
def test_poverty_estimators(method, expected):
    # Only the first row contributes: 2 people at income 20, threshold 100.
    # The 3 people exactly at threshold are not poor; income 0 has weight 0.
    d = mdf.MicroDataFrame(
        {"income": [20.0, 100.0, 0.0], "threshold": [100.0, 100.0, 100.0]},
        weights=[2, 3, 0],
    )
    assert getattr(d, method)("income", "threshold") == pytest.approx(expected)
    assert getattr(d, method)(d.income, d.threshold) == pytest.approx(expected)


@pytest.mark.parametrize(
    "income,rate,deep_rate,gap,deep_gap,squared",
    [
        ([50.0, 100.0, 0.0], 2 / 5, 0, 100, 0, 5000),
        ([100.0, 200.0, 0.0], 0, 0, 0, 0, 0),
    ],
)
def test_poverty_threshold_boundaries(income, rate, deep_rate, gap, deep_gap, squared):
    d = mdf.MicroDataFrame(
        {"income": income, "threshold": [100.0] * 3}, weights=[2, 3, 0]
    )
    for method, expected in zip(
        [
            "poverty_rate",
            "deep_poverty_rate",
            "poverty_gap",
            "deep_poverty_gap",
            "squared_poverty_gap",
        ],
        [rate, deep_rate, gap, deep_gap, squared],
    ):
        assert getattr(d, method)("income", "threshold") == pytest.approx(expected)
