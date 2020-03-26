import pytest
from neurokit.utils import mask_to_intervals, intervals_to_mask


data = [
    ([1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1], [(0, 2), (6, 7), (8, 11)], None),
    ([0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0], [(1, 3), (8, 10)], None),
    ([0, 0, 0, 0, 0], [], None),
    ([1, 1, 1, 1, 1], [(0, 5)], None),
]

data_index = [
    ([1, 1, 1, 1, 1], [("a", "e")], ["a", "b", "c", "d", "e"]),
    ([1, 1, 0, 0, 0], [("a", "b")], ["a", "b", "c", "d", "e"]),
    ([0, 0, 0, 0, 1], [("e", "e")], ["a", "b", "c", "d", "e"]),
    ([0, 0, 1, 1, 0], [("c", "d")], ["a", "b", "c", "d", "e"]),
    ([0, 0, 1, 0, 0], [("c", "c")], ["a", "b", "c", "d", "e"]),
]


@pytest.mark.parametrize("the_mask,the_intervals,the_index", data + data_index)
def test_mask_to_intervals(the_mask, the_intervals, the_index):
    intervals = mask_to_intervals(the_mask, the_index)

    assert intervals == the_intervals


@pytest.mark.parametrize("the_mask,the_intervals,the_index", data_index)
def test_intervals_to_mask(the_mask, the_intervals, the_index):
    mask = intervals_to_mask(the_intervals, the_index)

    assert list(mask) == list(the_mask)
