import numpy as np

from topo_descriptors import topo


def test_sx_distance():

    radius = 150.0
    dx = 50.0
    dy = 40.0

    output = topo._sx_distance(radius, dx, dy)
    expected_first_row = np.array(
        [
            256.1249695,
            219.31712199,
            188.67962264,
            167.63054614,
            160.0,
            167.63054614,
            188.67962264,
            219.31712199,
            256.1249695,
        ]
    )

    assert np.all(np.isclose(output[0, :], expected_first_row))
    assert output.dtype == np.float64


def test_sx_bresenhamlines():

    start = np.array([[8, 9], [17, 22]])
    end = np.array([15, 15])
    output = topo._sx_bresenhamlines(start, end)
    expected = np.array(
        [
            [9, 10],
            [10, 11],
            [11, 12],
            [12, 12],
            [13, 13],
            [14, 14],
            [17, 21],
            [16, 20],
            [16, 19],
            [16, 18],
            [16, 17],
            [15, 16],
        ]
    )

    assert np.all(output == expected)
    assert output.dtype == np.int64


def test_sx_source_idx_delta():

    azimuths = np.array([3.0, 4.0, 5.0, 6.0])
    radius = 500
    dx = 20
    dy = 30
    output = topo._sx_source_idx_delta(azimuths, radius, dx, dy)
    expected = np.array([[17, 1], [17, 2], [17, 2], [17, 3]])

    assert np.all(output == expected)
    assert output.dtype == np.int64


def test_create_patches():
    # Dummy DEM with two distinct patches and a pixel touching the second
    # patch only diagonally
    dem = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 3]
    ], dtype=float)

    # Test 8-connectivity: the diagonal pixel joins the second patch
    output_8 = topo._create_patches(dem, neighbors=8)
    expected_8 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 2]
    ])

    # Test 4-connectivity: the diagonal pixel is a separate patch
    output_4 = topo._create_patches(dem, neighbors=4)
    expected_4 = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 2, 2, 0],
        [0, 0, 0, 0, 3]
    ])

    # Test NaN values
    dem_with_nan = dem.copy()
    dem_with_nan[1, 1] = np.nan
    dem_with_nan[2, 2] = np.nan
    output_nan = topo._create_patches(dem_with_nan, neighbors=4, fillna_value=0)
    expected_nan = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 3, 3, 0],
        [0, 0, 3, 3, 0],
        [0, 0, 0, 0, 4]
    ])

    assert np.all(output_8 == expected_8)
    assert np.all(output_4 == expected_4)
    assert np.all(output_nan == expected_nan)
    assert output_8.dtype == np.int32


def test_extract_patches():
    # Dummy DEM with two distinct patches
    dem = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.5, 1.7, 0.0, 0.0],
        [0.0, 1.6, 1.8, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5, 2.7, 0.0],
        [0.0, 0.0, 2.6, 2.8, 0.0]
    ])
    # Both blocks share the value 1 but are labeled as patches 1 and 2
    dem_thresh = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0]
    ])

    # Test extract one patch (index 1)
    output_single = topo._extract_patches(dem_thresh, 1, val=dem)
    expected_single = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.5, 1.7, 0.0, 0.0],
        [0.0, 1.6, 1.8, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    # Test extract multiple patches
    output_multi = topo._extract_patches(dem_thresh, [1, 2], val=dem)
    expected_first, expected_second = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.5, 1.7, 0.0, 0.0],
        [0.0, 1.6, 1.8, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ]), np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 2.5, 2.7, 0.0],
        [0.0, 0.0, 2.6, 2.8, 0.0]
    ])

    assert np.all(output_single == expected_single)
    assert isinstance(output_multi, tuple)
    assert len(output_multi) == 2
    assert np.all(output_multi[0] == expected_first)
    assert np.all(output_multi[1] == expected_second)
    assert output_single.dtype == np.float64
