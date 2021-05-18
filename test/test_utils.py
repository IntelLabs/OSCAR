#
# Copyright (C) 2020 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import logging
import numpy as np
import pytest
import math
import torch

from numpy.testing import assert_array_equal

from oscar.utils.utils import batch_yield
from oscar.utils.utils import bmap
from oscar.utils.layers import Quantize

logger = logging.getLogger(__name__)


@pytest.fixture
def videos():
    # 7 videos each with 19 frames containing 320x240 RGB pixels
    x = np.random.random((7, 19, 320, 240, 3))
    return x


@pytest.fixture
def labels():
    # 7 labels
    y = np.random.randint(low=0, high=101, size=(7,))
    return y

def assert_stopped(x):
    with pytest.raises(StopIteration):
        next(x)


def test_batch_yield(videos):
    x = batch_yield(videos)

    for i in range(videos.shape[0]):
        x_batch, x_selector = next(x)

        assert_array_equal(x_batch, videos[i, :, :, :, :])

    assert_stopped(x)


def test_batch_yield_dim(videos):
    DIM = 1

    x = batch_yield(videos, dim=DIM) #size=0

    for i in range(videos.shape[DIM]):
        x_batch, x_selector = next(x)

        assert_array_equal(x_batch, videos[:, i, :, :, :])

    assert_stopped(x)


def test_batch_yield_baddim(videos):
    x = batch_yield(videos, dim=len(videos.shape)) #size=0

    x_batch, x_selector = next(x)

    assert_array_equal(x_batch, videos)
    assert_stopped(x)


@pytest.mark.parametrize("size", [1, 3, 8])
def test_batch_yield_size(videos, size):
    x = batch_yield(videos, size=size)

    for i in range(0, videos.shape[0], size):
        x_batch, x_selector = next(x)

        assert_array_equal(x_batch, videos[i:i+size, :, :, :, :])

    assert_stopped(x)


def test_batch_yield_many(videos):
    x = batch_yield(videos, dim=(0, 1), size=(0, 0))

    for i in range(videos.shape[0]):
        for j in range(videos.shape[1]):
            x_batch, x_selector = next(x)

            assert_array_equal(x_batch, videos[i, j, :, :, :])

    assert_stopped(x)

@pytest.mark.parametrize("sizes", [(1, 1), (3, 3), (8, 8), (3, 5), (5, 3)])
def test_batch_yield_manysizes(videos, sizes):
    x = batch_yield(videos, dim=[0, 1], size=sizes)

    for i in range(0, videos.shape[0], sizes[0]):
        for j in range(0, videos.shape[1], sizes[1]):
            x_batch, x_selector = next(x)

            assert_array_equal(x_batch, videos[i:i+sizes[0], j:j+sizes[1], :, :, :])

    assert_stopped(x)


def test_batch_yield_manydimfail(videos):
    x = batch_yield(videos, dim=(0, 1))

    with pytest.raises(AssertionError):
        next(x)


def test_batch_yield_manysizefail(videos):
    x = batch_yield(videos, size=(3, 3))

    with pytest.raises(AssertionError):
        next(x)


def map1_mul(x):
    return x*2


def map1_mul_none(x):
    return x*2, None


def map1_mul_const(x):
    return x*2, 0


def map1_mul_mul(x):
    return x*2, x*4


def map2_mul(x, y):
    return x*2


def map2_mul_none(x, y):
    return x*2, None


def map2_none_mul(x, y):
    return None, y*2


def map2_mul_mul(x, y):
    return x*2, y*2


@pytest.mark.parametrize("map_fn", [map1_mul, map1_mul_none])
def test_bmap1(videos, map_fn):
    x = bmap(map_fn, videos)

    assert_array_equal(x, 2*videos)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_bmap1_batchsize(videos, batch_size):
    x = bmap(map1_mul, videos, batch_size=batch_size)

    assert_array_equal(x, 2*videos)


@pytest.mark.parametrize("map_fn", [map1_mul, map1_mul_none])
def test_bmap1_batchdim(videos, map_fn):
    x = bmap(map_fn, videos, batch_dim=1)

    assert_array_equal(x, 2*videos)


def test_bmap1_badmapfn(videos):
    with pytest.raises(TypeError):
        bmap(map2_mul, videos)


def test_bmap2(videos, labels):
    x = bmap(map2_mul, videos, labels)

    assert_array_equal(x, 2*videos)


    x, y = bmap(map2_mul_none, videos, labels)

    assert_array_equal(x, 2*videos)
    assert y is None


    x, y = bmap(map2_none_mul, videos, labels)

    assert x is None
    assert_array_equal(y, 2*labels)


    x, y = bmap(map2_mul_mul, videos, labels)

    assert_array_equal(x, 2*videos)
    assert_array_equal(y, 2*labels)


@pytest.mark.parametrize("batch_size", [1, 3, 8])
def test_bmap2_batchsize(videos, labels, batch_size):
    x = bmap(map2_mul, videos, labels, batch_size=batch_size)

    assert_array_equal(x, 2*videos)


    x, y = bmap(map2_mul_none, videos, labels, batch_size=batch_size)

    assert_array_equal(x, 2*videos)
    assert y is None


    x, y = bmap(map2_none_mul, videos, labels, batch_size=batch_size)

    assert x is None
    assert_array_equal(y, 2*labels)


    x, y = bmap(map2_mul_mul, videos, labels, batch_size=batch_size)

    assert_array_equal(x, 2*videos)
    assert_array_equal(y, 2*labels)


def test_bmap2_batchdim(videos, labels):
    x = bmap(map2_mul, videos, videos, batch_dim=1)

    assert_array_equal(x, 2*videos)


    x, y = bmap(map2_mul_none, videos, videos, batch_dim=1)

    assert_array_equal(x, 2*videos)
    assert y is None


    x, y = bmap(map2_mul_mul, videos, videos, batch_dim=1)

    assert_array_equal(x, 2*videos)
    assert_array_equal(y, 2*videos)


def test_bmap_badbatchdim(videos, labels):
    with pytest.raises(ValueError):
        bmap(map1_mul, videos, batch_dim=len(videos.shape))

    with pytest.raises(ValueError):
        bmap(map2_mul, videos, labels, batch_dim=len(labels.shape))


def test_bmap_badbatchsize(videos, labels):
    with pytest.raises(ValueError):
        bmap(map1_mul, videos, batch_size=0)

    with pytest.raises(ValueError):
        bmap(map1_mul, videos, batch_size=[1, 1, 0])

    with pytest.raises(ValueError):
        bmap(map2_mul, videos, labels, batch_size=[1, 0, 1])

    with pytest.raises(AssertionError):
        bmap(map2_mul, videos, labels, batch_size=[1, 1, 1])


def test_quantize():
    inputs = torch.tensor([0, 255, 0.4, 0.6, 32.5, 32.6, -1, 256, 128, 127])/255

    outputs = Quantize.apply(inputs, 255)
    assert_array_equal(outputs, torch.tensor([0, 255, 0, 1, 32, 33, 0, 255, 128, 127])/255)

    outputs = Quantize.apply(inputs, 1)
    assert_array_equal(outputs, torch.tensor([0, 1, 0, 0, 0, 0, 0, 1, 1, 0], dtype=torch.float))
