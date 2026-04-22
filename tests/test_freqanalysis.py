import pytest
import numpy as np
import thunderfish.freqanalysis as fa


def test_musical_intervals():
    r = 0.0
    i = -1
    for k in fa.musical_intervals:
        x = fa.musical_intervals[k]
        assert len(x) == 4, 'musical_intervals wrong number of item elements'
        assert x[3] > i, 'musical_intervals wrong index'
        assert x[0] > r, 'musical_intervals ratio too small'
        assert x[1]/x[2] == x[0], 'musical_intervals wrong ratio'
        r = x[0]
        i = x[3]
