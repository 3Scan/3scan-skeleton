import nose.tools
import numpy as np

import kesm.base.tools.io_tools as io_tools


def test_module_dir():
    d = io_tools.module_dir()
    assert d.endswith('io_tools'), d


def test_module_relative_path():
    nose.tools.assert_equals(
        io_tools.module_relative_path('base_tests.py'),
        __file__)


def test_pad_int():
    np.testing.assert_array_equal(io_tools.pad_int(5), "00000005")
