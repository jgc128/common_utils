import numpy as np

from common_utils.image import get_border_size


def test_get_border_size_dividable():
    shape = (256, 256, 3)
    target_multiplier = 64

    border_size = get_border_size(np.random.random(shape), target_multiplier)

    assert border_size == (32, 32, 32, 32)


def test_get_border_size_not_dividable():
    shape = (250, 250, 3)
    target_multiplier = 64

    border_size = get_border_size(np.random.random(shape), target_multiplier)

    assert border_size == (3, 3, 3, 3)


def test_get_border_size_not_square():
    shape = (240, 250, 3)
    target_multiplier = 64

    border_size = get_border_size(np.random.random(shape), target_multiplier)

    assert border_size == (8, 8, 3, 3)


def test_get_border_size_not_even():
    shape = (241, 250, 3)
    target_multiplier = 64

    border_size = get_border_size(np.random.random(shape), target_multiplier)

    assert border_size == (7, 8, 3, 3)
