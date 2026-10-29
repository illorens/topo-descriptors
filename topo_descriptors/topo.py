import logging

import numpy as np
import numpy.ma as ma
import xarray as xr
import dask.array as da
from numba import njit, prange
from scipy import ndimage, signal, stats
from skimage.morphology import medial_axis

import topo_descriptors.helpers as hlp
from topo_descriptors import CFG

logger = logging.getLogger(__name__)


def compute_dem(dem_ds, scales, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'dem' function to launch computations for all scales and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the DEM.
        Corresponds to the diameter of the kernel used to compute it.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    dem
    """

    hlp.check_dem(dem_ds)
    logger.info(f"***Starting dem computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]

    scales_pxl, res_meters = hlp.scale_to_pixel(scales, dem_ds)
    sigmas = scales_pxl / CFG.scale_std
    dem_val = hlp.get_da(dem_ds).values
    units = "m"

    for idx, sigma in enumerate(sigmas):
        logger.info(f"Computing scale {scales[idx]} meters")
        name = _dem_name(scales[idx])
        array = dem(dem_val, sigma)
        array[ind_nans] = np.nan
        hlp.to_netcdf(array, dem_ds, name, crop, outdir, units)
        del array


def dem(dem, sigma):
    """Smoothen digital elevation model with a standard deviation of sigma

    Parameters
    ----------
    dem : array representing the DEM.
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    array with smoothed dem

    See also
    --------
    scipy.ndimage.gaussian_filter
    """
    return ndimage.gaussian_filter(dem, sigma)


def _dem_name(scale):
    """Return names for the arrays in output of the dem function"""
    return f"DEM_{scale}M"


def compute_tpi(dem_ds, scales, smth_factors=None, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'tpi' function to launch computations for all scales and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the TPI.
        Corresponds to the diameter of the kernel used to compute it.
    smth_factors (optional) : scalar or None or list with a combination of both.
        Fraction(s) of the scale(s) at which the DEM is smoothed first (with a
        gaussian filter). If None (default), no prior smoothing is performed.
        If a scalar, the same fraction is used to determine the smoothing scale
        of all specified scales. If a list, must match the length of arg 'scales'.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    tpi, circular_kernel
    """

    hlp.check_dem(dem_ds)
    logger.info(f"***Starting TPI computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(smth_factors, "__iter__"):
        smth_factors = [smth_factors] * len(scales)

    scales_pxl, _ = hlp.scale_to_pixel(scales, dem_ds)
    sigmas = hlp.get_sigmas(smth_factors, scales_pxl)
    dem_val = hlp.get_da(dem_ds).values
    units = "m"

    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(
            f"Computing scale {scales[idx]} meters with smoothing factor"
            f" {smth_factors[idx]} ..."
        )
        name = _tpi_name(scales[idx], smth_factors[idx])
        array = tpi(dem_val, scale_pxl, sigma=sigmas[idx])
        array[ind_nans] = np.nan
        hlp.to_netcdf(array, dem_ds, name, crop, outdir, units)
        del array


@hlp.timer
def tpi(dem, size, sigma=None):
    """Compute the TPI over a digital elevation model. The TPI represents
    the elevation difference of a pixel relative to its neighbors.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the diameter (i.e. scale)
        in pixels at which the TPI is computed.
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    array with TPI values

    See also
    --------
    scipy.signal.convolve, scipy.ndimage.gaussian_filter
    """

    kernel = circular_kernel(size)
    # exclude mid point from the kernel
    kernel[int(size / 2), int(size / 2)] = 0

    if sigma:
        dem = ndimage.gaussian_filter(dem, sigma)

    conv_fn = lambda a: signal.convolve(a, kernel, mode="same")

    if isinstance(dem.data, da.Array):
        conv = da.map_overlap(conv_fn, dem.data, depth=size * 2, boundary="none")
    else:
        conv = conv_fn(dem)
    return dem - conv / np.sum(kernel)


def _tpi_name(scale, smth_factor):
    """Return name for the array in output of the tpi function"""

    add = f"_SMTHFACT{smth_factor:.3g}" if smth_factor else ""
    return f"TPI_{scale}M{add}"


def circular_kernel(size):
    """Generate a circular kernel.

    Parameters
    ----------
    size : int
        Size of the circular kernel (its diameter). For size < 5, the kernel is
        a square instead of a circle.

    Returns
    -------
    2-D array with kernel values
    """

    middle = int(size / 2)
    if size < 5:
        kernel = np.ones((size, size), dtype=np.float32)
    else:
        xx, yy = np.mgrid[:size, :size]
        circle = (xx - middle) ** 2 + (yy - middle) ** 2
        kernel = np.asarray(circle <= (middle**2), dtype=np.float32)

    return kernel


def compute_std(dem_ds, scales, smth_factors=None, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'std' function to launch computations for all scales and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the TPI.
        Corresponds to the diameter of the kernel used to compute it.
    smth_factors (optional) : scalar or None or list with a combination of both.
        Fraction(s) of the scale(s) at which the DEM is smoothed first (with a
        gaussian filter). If None (default), no prior smoothing is performed.
        If a scalar, the same fraction is used to determine the smoothing scale
        of all specified scales. If a list, must match the length of arg 'scales'.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    std, circular_kernel
    """

    hlp.check_dem(dem_ds)
    logger.info(f"***Starting STD computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(smth_factors, "__iter__"):
        smth_factors = [smth_factors] * len(scales)

    scales_pxl, _ = hlp.scale_to_pixel(scales, dem_ds)
    sigmas = hlp.get_sigmas(smth_factors, scales_pxl)
    dem_val = hlp.get_da(dem_ds).values
    units = "m"

    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(
            f"Computing scale {scales[idx]} meters with smoothing factor"
            f" {smth_factors[idx]} ..."
        )
        name = _std_name(scales[idx], smth_factors[idx])
        array = std(dem_val, scale_pxl, sigma=sigmas[idx])
        array[ind_nans] = np.nan
        hlp.to_netcdf(array, dem_ds, name, crop, outdir, units)
        del array


@hlp.timer
def std(dem, size, sigma=None):
    """Compute the standard deviation over a digital elevation model within
    a rolling window.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the diameter (i.e. scale)
        in pixels at which the std is computed.
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    array with local standard deviation values

    See also
    --------
    scipy.signal.convolve, scipy.ndimage.gaussian_filter
    """
    kernel = circular_kernel(size)
    kernel_sum = np.sum(kernel)
    if sigma:
        dem = ndimage.gaussian_filter(dem, sigma)

    squared_dem = dem.astype("int32") ** 2
    sum_dem = signal.convolve(dem, kernel, mode="same")
    sum_squared_dem = signal.convolve(squared_dem, kernel, mode="same")

    variance = (sum_squared_dem - sum_dem**2 / kernel_sum) / (kernel_sum - 1)
    variance = np.clip(variance, 0, None)  # avoid small negative values

    return np.sqrt(variance)


def _std_name(scale, smth_factor):
    """Return name for the array in output of the tpi function"""

    add = f"_SMTHFACT{smth_factor:.3g}" if smth_factor else ""
    return f"STD_{scale}M{add}"


def compute_valley_ridge(
    dem_ds,
    scales,
    mode,
    flat_list=[0, 0.15, 0.3],
    smth_factors=None,
    ind_nans=[],
    crop=None,
    outdir=".",
):
    """Wrapper to 'valley_ridge' function to launch computations for all scales
    and save outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters over which we want to compute the valley or the ridge
        index. Corresponds to the size of the squared kernel used to compute it.
    mode : {valley, ridge}
        Whether to compute the valley or ridge index.
    flat_list (optional) : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.
        Default is [0, 0.15, 0.3].
    smth_factors (optional) : scalar or None or list with a combination of both.
        Fraction(s) of the scale(s) at which the DEM is smoothed first (with a
        gaussian filter). If None (default), no prior smoothing is performed.
        If a scalar, the same fraction is used to determine the smoothing scale
        of all specified scales. If a list, must match the length of arg 'scales'.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    valley_ridge, _valley_kernels, _valley_ridge_names
    """

    hlp.check_dem(dem_ds)
    logger.info(f"***Starting {mode} index computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(smth_factors, "__iter__"):
        smth_factors = [smth_factors] * len(scales)

    scales_pxl, _ = hlp.scale_to_pixel(scales, dem_ds)
    sigmas = hlp.get_sigmas(smth_factors, scales_pxl)
    dem_val = hlp.get_da(dem_ds).values
    units = "1"

    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(
            f"Computing scale {scales[idx]} meters with smoothing factor"
            f" {smth_factors[idx]} ..."
        )
        names = _valley_ridge_names(scales[idx], mode, smth_factors[idx])
        arrays = valley_ridge(dem_val, scale_pxl, mode, flat_list, sigmas[idx])

        for array, name in zip(arrays, names):
            array[ind_nans] = np.nan
            hlp.to_netcdf(array, dem_ds, name, crop, outdir, units)


@hlp.timer
def valley_ridge(dem, size, mode, flat_list=[0, 0.15, 0.3], sigma=None):
    """Compute the valley or ridge index over a digital elevation model.
    The valley/ridge index highlights valley or ridges  at various scales.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the width (i.e. scale)
        in pixels of the valleys we are trying to highlight.
    mode : {valley, ridge}
        Whether to compute the valley or ridge index.
    flat_list (optional) : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.
        Default is [0, 0.15, 0.3].
    sigma (optional) : scalar
        If provided, the DEM is first smoothed with a gaussian filter of standard
        deviation sigma (in pixel size).

    Returns
    -------
    list of two arrays :
        First element is the norm and second the direction of the valley or ridge
        index. The direction in degrees is defined from 0° to 179°, increasing
        clockwise. W-E oriented valleys have a direction close to 0° or 180°,
        while S-N oriented valleys have a direction close to 90°.

    See also
    --------
    scipy.signal.convolve, scipy.ndimage.gaussian_filter
    """

    if mode not in ("valley", "ridge"):
        raise ValueError(f"Unknown mode {mode!r}")

    if sigma:
        dem = ndimage.gaussian_filter(dem, sigma)

    dem = (dem - dem.mean()) / dem.std()
    n_y, n_x = dem.shape
    dem = np.broadcast_to(dem, (len(flat_list), n_y, n_x))
    angles = np.arange(0, 180, dtype=np.float32)
    index_norm = np.zeros((n_y, n_x), dtype=np.float32) - np.inf
    index_dir = np.empty((n_y, n_x), dtype=np.float32)

    if mode == "ridge":
        kernels = _ridge_kernels(size, flat_list)
    else:
        kernels = _valley_kernels(size, flat_list)

    for angle in angles:  # 0° = E-W valleys, 90° = S-N valleys
        kernels_rot = _rotate_kernels(kernels, angle)
        conv = signal.convolve(dem, kernels_rot, mode="same")
        conv = np.max(conv, axis=0)
        bool_greater = conv > index_norm
        index_norm[bool_greater] = conv[bool_greater]
        index_dir[bool_greater] = angle
        del bool_greater
        if angle % 20 == 0:
            logger.info(f"angle {int(angle)}/180 finished")

    index_norm = np.ndarray.clip(index_norm, min=0)
    return [index_norm, index_dir]


def _valley_ridge_names(scale, mode, smth_factor):
    """Return names for the arrays in output of the valley_ridge function"""

    add = f"_SMTHFACT{smth_factor:.3g}" if smth_factor else ""
    name_norm = f"{mode}_NORM_{scale}M{add}"
    name_dir = f"{mode}_DIR_{scale}M{add}"

    return [name_norm, name_dir]


def _valley_kernels(size, flat_list):
    """Generate normalized V-shape and U-shape kernels to compute valley index.

    Parameters
    ----------
    size : int
        Size of the kernel.
    flat_list : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.

    Returns
    -------
    3-D array with 2-D kernels for each specified flat fraction.
    """

    middle = int(np.floor(size / 2))
    kernel_tmp = np.broadcast_to(np.arange(0, middle + 1), (size, middle + 1)).T
    kernel_tmp = np.concatenate(
        (np.flip(kernel_tmp[1:, :], axis=0), kernel_tmp), axis=0
    )
    kernel_tmp = np.asarray(kernel_tmp, dtype=np.float32)
    kernels = np.broadcast_to(kernel_tmp, (len(flat_list), size, size)).copy()

    for ind, flat in enumerate(flat_list):
        halfwidth = int(np.floor(np.floor(size * flat / 2) + 0.5))
        kernels[ind, middle - halfwidth : middle + halfwidth + 1, :] = kernels[
            ind, middle - halfwidth, 0
        ]
        kernels = (kernels - np.mean(kernels, axis=(1, 2), keepdims=True)) / np.std(
            kernels, axis=(1, 2), keepdims=True
        )

    return kernels


def _ridge_kernels(size, flat_list):
    """Generate normalized flipped V-shape and U-shape kernels to compute ridge index.

    Parameters
    ----------
    size : int
        Size of the kernel.
    flat_list : list of floats in [0,1[
        Fractions of flat along the center line of the V-shape kernels. A certain
        amount of flat is use to approximate the shape of glacial valleys.

    Returns
    -------
    3-D array with 2-D kernels for each specified flat fraction.
    """

    return _valley_kernels(size, flat_list) * -1


def _rotate_kernels(kernel, angle):
    """Rotate a 3-D kernel in the plane given by the last two axes"""

    kernels_rot = ndimage.rotate(
        kernel, angle, axes=(1, 2), reshape=True, order=2, mode="constant", cval=-9999
    )
    kernels_rot = ma.masked_array(kernels_rot, mask=kernels_rot == -9999)
    kernels_rot = (
        kernels_rot - np.mean(kernels_rot, axis=(1, 2), keepdims=True)
    ) / np.std(kernels_rot, axis=(1, 2), keepdims=True)
    return ma.MaskedArray.filled(kernels_rot, 0).astype(np.float32)


def compute_gradient(dem_ds, scales, sig_ratios=1, ind_nans=[], crop=None, outdir="."):
    """Wrapper to 'gradient' function to launch computations for all scales
    and save outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the the valley or ridge
        index. Corresponds to the size of the squared kernel used to compute it.
    sig_ratios (optional) : scalar or list of scalars.
        Ratios w.r.t scales to define the smoothing scale in the perpendicular
        direction of the directional derivatives. If a list, must match the length
        of arg 'scales'. Default is 1 (i.e. same smoothing on both directions for
        all scales)
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions.
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    gradient, sobel, _gradient_names
    """

    hlp.check_dem(dem_ds)
    logger.info(f"***Starting gradients computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]
    if not hasattr(sig_ratios, "__iter__"):
        sig_ratios = [sig_ratios] * len(scales)

    scales_pxl, res_meters = hlp.scale_to_pixel(scales, dem_ds)
    sigmas = scales_pxl / CFG.scale_std
    dem_val = hlp.get_da(dem_ds).values
    all_units = ["1", "1", "degree", "degree"]

    for idx, sigma in enumerate(sigmas):
        logger.info(
            f"Computing scale {scales[idx]} meters with sigma ratio "
            f"{sig_ratios[idx]} ..."
        )
        names = _gradient_names(scales[idx], sig_ratios[idx])
        arrays = gradient(
            dem_val,
            sigma,
            res_meters,
            sig_ratio=sig_ratios[idx],
        )

        for array, name, units in zip(arrays, names, all_units):
            array[ind_nans] = np.nan
            hlp.to_netcdf(array, dem_ds, name, crop, outdir, units)

        del arrays


@hlp.timer
def gradient(dem, sigma, res_meters, sig_ratio=1):
    """Compute directional derivatives, slope and aspect over a digital elevation
    model.

    Parameters
    ----------
    dem : array representing the DEM.
    sigma : scalar
        Standard deviation for the gaussian filters. This is set at 1/4 of the
        scales to which topo descriptors are computed (i.e. scale = 4*sigma)
    res_meters : dict with two 1-D or 2-D arrays
        Resolution in meters of each DEM grid points in the x and y directions.
        This is the second element returned by the function hlp.scale_to_pixel.
    sig_ratio (optional) : scalar
        Ratio w.r.t sigma to define the standard deviation of the gaussian in
        the direction perpendicular to the derivative. Default is 1 (i.e. same
        smoothing on both directions).

    Returns
    -------
    list of four arrays :
        First element is the W-E derivative, second the S-N derivative, third
        the slope (i.e. magnitude of the gradient) and fourth the aspect (i.e.
        direction of the gradient).

    See also
    --------
    scipy.ndimage.gaussian_filter, np.gradient, topo_helpers.scale_to_pixel
    """

    if sigma <= 1:
        dx, dy = sobel(dem)  # for lowest scale, use sobel filter instead
    elif sig_ratio == 1:  # faster solution when sig_ratio is 1
        dy, dx = np.gradient(ndimage.gaussian_filter(dem, sigma))
    else:
        sigma_perp = sigma * sig_ratio
        dx = np.gradient(ndimage.gaussian_filter(dem, (sigma_perp, sigma)), axis=1)
        dy = np.gradient(ndimage.gaussian_filter(dem, (sigma, sigma_perp)), axis=0)

    _normalize_dxy(dx, dy, res_meters)

    slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)  # in degrees
    aspect = (
        180 + np.degrees(np.arctan2(dx, dy))
    ) % 360  # north faces = 0°, east faces = 90°

    return [dx, dy, slope, aspect]


def _gradient_names(scale, sig_ratio):
    """Return names for the arrays in output of the gradient function"""

    name_dx = f"WE_DERIVATIVE_{scale}M_SIGRATIO{sig_ratio:.3g}"
    name_dy = f"SN_DERIVATIVE_{scale}M_SIGRATIO{sig_ratio:.3g}"
    name_slope = f"SLOPE_{scale}M_SIGRATIO{sig_ratio:.3g}"
    name_aspect = f"ASPECT_{scale}M_SIGRATIO{sig_ratio:.3g}"

    return [name_dx, name_dy, name_slope, name_aspect]


def sobel(dem):
    """Compute directional derivatives, based on the sobel filter. The sobel is
    a combination of a derivative and a smoothing filter. It is defined over a
    3x3 kernel.

    Parameters
    ----------
    dem : array representing the DEM.

    Returns
    -------
    dx : 2-D array
        Derivative in the x direction (i.e. along second axis)
    dy : 2-D array
        Derivative in the y direction (i.e. along first axis)

    See also
    --------
    ndimage.convolve
    """

    sobel_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

    sobel_kernel /= np.sum(np.abs(sobel_kernel))
    dx = ndimage.convolve(dem, sobel_kernel)
    dy = ndimage.convolve(dem, sobel_kernel.T)

    return dx, dy


def _normalize_dxy(dx, dy, res_meters):
    """Normalize directional derivatives, based on (projected) grid resolution.
    This converts from units of "m / pixel" to "1". Normalization occurs 'in place'.

    Parameters
    ----------
    dx : 2-D array
        Derivative in the x direction (i.e. along second axis)
    dy : 2-D array
        Derivative in the y direction (i.e. along first axis)
    res_meters : dict with two 1-D or 2-D arrays
        Resolution in meters of each DEM grid points in the x and y directions.
        This is the second element returned by the function hlp.scale_to_pixel.

    See also
    --------
    topo_helpers.scale_to_pixel
    """

    y_res = res_meters["y"]
    if len(y_res.shape) == 1:
        y_res = y_res[:, np.newaxis]

    dx /= res_meters["x"]
    dy /= y_res


def compute_sx(
    dem_ds,
    azimuth,
    radius,
    height=10.0,
    azimuth_arc=10.0,
    azimuth_steps=15,
    radius_min=0.0,
    crop=None,
    outdir=".",
):
    """Wrapper to 'Sx' function to launch computations and save
    outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    azimuth : scalar
        Azimuth angle in degrees for imaginary lines.
    radius : scalar
        Maximum distance in meters for the imaginary lines.
    azimuth_arc (optional): scalar
        Angle of the circular sector centered around 'azimuth'.
    azimuth_steps (optional):
        Number of lines traced to find pixels within the circular sector.
        A higher number leads to more precise but longer computations.
    radius_min (optional): scalar
        Minimum value of radius below which pixels are excluded from imaginary lines.
    height (optional): scalar
        Parameter that accounts for instrument heights and
        reduce the impact of small proximal terrain perturbations.
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.

    See also
    --------
    sx, _sx_distance, _sx_source_idx_delta, _sx_bresenhamlines, _sx_rolling
    """
    hlp.check_dem(dem_ds)
    logger.info(
        f"***Starting Sx computation for azimuth {azimuth} meters and radius {radius}***"
    )

    array = sx(
        dem_ds,
        azimuth,
        radius,
        height=height,
        azimuth_arc=azimuth_arc,
        azimuth_steps=azimuth_steps,
        radius_min=radius_min,
    )

    units = "degree"
    name = _sx_name(radius, azimuth)
    hlp.to_netcdf(array, dem_ds, name, crop, outdir, units)


@hlp.timer
def sx(
    dem_ds,
    azimuth,
    radius,
    height=10.0,
    azimuth_arc=10.0,
    azimuth_steps=15,
    radius_min=0.0,
):
    """Compute the Sx over a digital elevation model.

    The Sx represents the maximum slope among all imaginary lines connecting a
    given pixel with all the ones lying in a specific direction and within a
    specified distance (Winstral et al., 2017). The Sx is a proven wind-specific
    terrain parameterization, as it is able to differentiate the slopes based
    on given wind directions and identify sheltered and exposed locations with
    respect to the incoming wind.
    Note that this routine computes one azimuth at a time. It is accelerated with
    Numba's Just In Time compilation, but is still computationally expensive.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    azimuth : scalar
        Azimuth angle in degrees for imaginary lines.
    radius : scalar
        Maximum distance in meters for the imaginary lines.
    azimuth_arc (optional): scalar
        Angle of the circular sector centered around 'azimuth'.
        Set to zero in order to draw a single line.
    azimuth_steps (optional): scalar integer
        Number of lines traced to find pixels within the circular sector.
        A higher number leads to more precise but longer computations.
        Defaults to 1 when 'azimuth_arc' is 0.
    radius_min (optional): scalar
        Minimum value of radius below which pixels are excluded from imaginary lines.
    height (optional): scalar
        Parameter that accounts for instrument heights and
        reduce the impact of small proximal terrain perturbations.

    Returns
    -------
    array with Sx values for one azimuth

    See also
    --------
    _sx_distance, _sx_source_idx_delta, _sx_bresenhamlines, _sx_rolling
    """

    if not isinstance(dem_ds, xr.Dataset):
        raise TypeError("Argument 'dem_ds' must be a xr.Dataset.")

    if azimuth_arc == 0:
        azimuth_steps = 1

    # define all azimuths
    azimuths = np.linspace(
        azimuth - azimuth_arc / 2, azimuth + azimuth_arc / 2, azimuth_steps
    )

    # grid resolutions
    _, res_meters = hlp.scale_to_pixel(radius, dem_ds)
    dx = res_meters["x"].mean()
    dy = res_meters["y"].mean()

    # horizontal distance in meters from center in a window of size 2*radius
    window_distance = _sx_distance(radius, dx, dy)

    # exclude pixels closer than radius_min
    window_distance[window_distance < radius_min] = np.nan

    # indices of pixels that lie at distance radius in direction azimuth
    window_center = np.floor(np.array(window_distance.shape) / 2)
    source_delta = _sx_source_idx_delta(azimuths, radius, dx, dy)
    source = (window_center + source_delta).astype(int)

    # indices of all pixels between source pixels and target (center)
    lines_indices = _sx_bresenhamlines(source, window_center)

    # compute Sx
    sx = _sx_rolling(hlp.get_da(dem_ds).values, window_distance, lines_indices, height)

    return sx


def _sx_distance(radius, dx, dy):
    """Compute distance from center in meters in a window of size 'radius'."""

    dx_abs = np.abs(dx)
    dy_abs = np.abs(dy)
    radius_pxl = max(radius / dy_abs, radius / dx_abs)

    # initialize window
    window = 2 * radius_pxl + 1  # must be odd
    center = np.floor(window / 2)
    x = np.arange(window)
    y = np.arange(window)
    x, y = np.meshgrid(x, y)

    # calculate distances from center for all points in the window
    distances = np.sqrt((((y - center) * dy) ** 2) + ((x - center) * dx) ** 2)

    return distances


def _sx_source_idx_delta(azimuths, radius, dx, dy):
    """Compute indices of pixels that lie at a distance 'radius' from
    the target, in the direction of 'azimuths'.
    """

    azimuths_rad = np.deg2rad(azimuths)
    delta_y_idx = np.rint(radius / dy * np.cos(azimuths_rad))
    delta_x_idx = np.rint(radius / dx * np.sin(azimuths_rad))

    delta = np.column_stack([delta_y_idx, delta_x_idx])

    return delta.astype(np.int64)


def _sx_bresenhamlines(start, end):
    """Compute indices of all pixels that lie between two sets of pixels."""

    max_iter = np.max(np.max(np.abs(end - start), axis=1))
    npts, dim = start.shape

    slope = end - start
    scale = np.max(np.abs(slope), axis=1).reshape(-1, 1)
    zeroslope = (scale == 0).all(1)
    scale[zeroslope] = np.ones(1)
    normalizedslope = np.array(slope, dtype=np.double) / scale
    normalizedslope[zeroslope] = np.zeros(slope[0].shape)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    blines = start[:, np.newaxis, :] + normalizedslope[:, np.newaxis, :] * stepmat

    # Approximate to nearest int
    blines = np.array(np.rint(blines), dtype=start.dtype)

    # Stop lines before center
    bsum = np.abs(blines - end).sum(axis=2)
    mask = np.diff(bsum, prepend=bsum[:, 0:1]) <= 0
    blines = blines[mask].reshape(-1, start.shape[-1])
    mask = np.all(blines == end, axis=1)
    blines = blines[~mask]

    return blines


@njit(parallel=True)
def _sx_rolling(dem, distance, blines, height):
    """Compute Sx values for the array with a loop over all elements."""

    window = int(distance.shape[0] / 2)
    ny, nx = dem.shape
    distance = np.array(
        [distance[j, i] for j, i in list(zip(blines[:, 0], blines[:, 1]))]
    )
    blines_centered = blines - window

    sx = np.zeros_like(dem)
    for j in prange(window, ny - window):
        for i in prange(window, nx - window):
            j_blines = j + blines_centered[:, 0]
            i_blines = i + blines_centered[:, 1]
            dem_blines = np.array([dem[j, i] for j, i in list(zip(j_blines, i_blines))])

            # compute tangent z / distance between P0 and all points
            z = dem_blines - (dem[j, i] + height)
            elev_angle = np.rad2deg(np.arctan(z / distance))

            # find the maximum angle in the cone
            sx[j, i] = np.nanmax(elev_angle)

    return sx


def _sx_name(radius, azimuth):
    """Return name for the array in output of the Sx function"""

    add = f"_RADIUS{int(radius)}_AZIMUTH{int(azimuth)}"
    return f"SX{add}"


def compute_foehn(
    dem_ds,
    scale=140e3,
    sigma_rims=10e3,
    slope_critical=0.7,
    slope_spread=0.1,
    thresh_slope=0.05,
    thresh_curv=0.00025,
    ind_nans=[],
    crop=None,
    outdir="."
):
    """Wrapper to 'foehn' function to launch computations
    and save outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scale : scalar
        Scale in meters at which the DEM is smoothed to make the main alpine
        ridge visible. The standard deviation of the Gaussian kernel is
        scale / CFG.scale_std. Default is 140 km.
    sigma_rims : scalar
        Standard deviation in meters of the Gaussian kernel used to smooth
        the slope and curvature rims. Default is 10 km.
    slope_critical : scalar
        Critical slope (in degrees) for the logistic transformation of the
        slope. Default is 0.7.
    slope_spread : scalar
        Scale parameter (in degrees) of the logistic transformation of the
        slope. Default is 0.1.
    thresh_slope : scalar
        Threshold on the logistic-transformed slope (in [0, 1]) for detecting
        the slope rim. Default is 0.05.
    thresh_curv : scalar
        Threshold on the curvature (gradient magnitude of the
        logistic-transformed slope) for detecting curvature rims.
        Default is 0.00025.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    foehn, _create_patches, _extract_patches
    """
    hlp.check_dem(dem_ds)
    logger.info(f"***Starting computation for Foehn descriptors***")

    scale_pxl, res_meters = hlp.meters_to_pixels(scale, dem_ds, rounding="none")
    sigma = float(scale_pxl) / CFG.scale_std
    sigma_rims_pxl, _ = hlp.meters_to_pixels(sigma_rims, dem_ds, rounding="none")
    sigma_rims_pxl = float(sigma_rims_pxl)
    dem_val = hlp.get_da(dem_ds).values

    arrays = foehn(
        dem_val,
        res_meters,
        sigma,
        sigma_rims_pxl,
        slope_critical,
        slope_spread,
        thresh_slope,
        thresh_curv,
    )

    names = _foehn_names()
    for name, array in zip(names, arrays):
        array[ind_nans] = np.nan
        # descriptors are normalized, hence dimensionless
        hlp.to_netcdf(array, dem_ds, name, crop, outdir)
        del array


def foehn(
    dem,
    res_meters,
    sigma,
    sigma_rims,
    slope_critical=0.7,
    slope_spread=0.1,
    thresh_slope=0.05,
    thresh_curv=0.00025,
):
    """Compute South Foehn descriptors over a digital elevation model.
    The Foehn descriptors capture large- and small-scale variations in
    the terrain that influence Foehn wind dynamics, helping to identify
    regions that are more or less susceptible to South Foehn events
    based on elevation, slope, and curvature patterns.

    Parameters
    ----------
    dem : array representing the DEM.
    res_meters : dict with two 1-D or 2-D arrays
        Resolution in meters of each DEM grid point in the x and y directions.
        This is the second element returned by the function
        hlp.meters_to_pixels.
    sigma : scalar
        Standard deviation of the Gaussian filter used to smooth the DEM
        (in pixel size).
    sigma_rims : scalar
        Standard deviation of the Gaussian filter used to smooth the slope and
        curvature rims (in pixel size).
    slope_critical : scalar
        Critical slope (in degrees) for the logistic transformation of the
        slope. Default is 0.7.
    slope_spread : scalar
        Scale parameter (in degrees) controlling the spread of the logistic
        transformation of the slope. Default is 0.1.
    thresh_slope : scalar
        Threshold on the logistic-transformed slope (in [0, 1]) for detecting
        the slope rim. Default is 0.05.
    thresh_curv : scalar
        Threshold on the curvature (gradient magnitude of the
        logistic-transformed slope) for detecting curvature rims.
        Default is 0.00025.

    Returns
    -------
    list of five arrays :
        - envelope: Northern alpine slope where South Foehn occurs, weighted
          by the negative DEM.
        - cos1 harmonic: represents large-scale elevation variations.
        - cos2 harmonic: represents finer-scale terrain undulations.
        - sin1 harmonic: represents large-scale transitions rather than
          absolute height.
        - sin2 harmonic: represents finer-scale phase shifts in elevation.
        Each array is divided by the mean of its non-zero absolute values,
        hence the outputs are dimensionless.

    See also
    --------
    scipy.ndimage.gaussian_filter, scipy.stats.logistic.cdf, _extract_patches,
    _create_patches, _get_ridge_divide

    Notes
    -----
    The steps are as follows:

    1. The DEM is smoothed so that the main alpine ridge becomes visible.
    2. The slope of the smooth DEM is used to derive the Northern alpine slope.
    3. The Northern alpine slope region is broadened by including the
       adjacent band of high curvature to create the envelope.
    4. The smooth DEM is rescaled to [pi/2, 5*pi/2] and input into
       trigonometric functions to produce a Fourier decomposition of the
       smooth DEM. By linearly combining Fourier components, advances
       and retractions of Foehn into the Swiss plateau can be modelled.
    5. The Fourier components are multiplied by the Northern alpine slope mask
       (envelope) to confine them to regions where South Foehn occurs.
    6. The descriptors are multiplied by the negative high-res DEM
       (max(DEM) - DEM) to de-emphasize mountaintops and emphasize valley
       bottoms where Foehn temperature anomalies are strongest.

    The patches are selected by their label from _create_patches, which
    numbers them in row-major order. The DEM is therefore assumed to have
    north at the first row (decreasing y) and to cover the Alpine domain,
    so that the first patch is the Northern alpine slope.

    References
    ----------
    This function implements the methodology developed by Louis Frey as
    part of his PhD thesis at ETHZ. Original implementation in R by
    Louis Frey (url_to_public_R_code.com).

    If you found this function useful for your own research, consider
    citing:

    .. code-block:: bibtex

        @article{lastname_papertitle_year,
          author = {First Name Last Name and Co-author's First Name Last Name},
          title = {Paper Title},
          journal = {Journal Name},
          volume = {Volume Number},
          number = {Issue Number},
          pages = {Page Range (e.g., 123--456)},
          year = {2025},
          doi = {DOI if available},
          url = {URL to the paper, if applicable}
        }
    """
    dem_smooth = ndimage.gaussian_filter(dem, sigma, radius=int(3*sigma))

    # Slope rim
    slope = gradient(dem_smooth, 1, res_meters)[2]
    slope = stats.logistic.cdf(slope, loc=slope_critical, scale=slope_spread)

    # N/S Alps divide: the flat ridge area (patch 2 of the low-slope patches)
    slope_divide = _get_ridge_divide(slope, thresh_slope, patch_idx=2)

    slope_thresh = np.where(slope >= thresh_slope, slope, np.nan)
    slope_thresh[slope_divide] = np.nan

    # Slope rim: first patch, i.e. the northern slope
    slope_rim = _extract_patches(slope_thresh, patch_idxs=1, val=slope)
    slope_rim = ndimage.gaussian_filter(
        slope_rim, sigma_rims, radius=int(3*sigma_rims)
    )

    # Curvature rims
    curvature = gradient(slope, 1, res_meters)[2]
    curvature[:2, :] = 0.  # remove edge artefacts on the northern border
    curvature_thresh = np.where(curvature >= thresh_curv, curvature, np.nan)
    curvature_thresh[slope_divide] = np.nan

    # Curvature rim: first patch, i.e. the northern slope
    curv_rim = _extract_patches(curvature_thresh, patch_idxs=1, val=curvature)
    curv_rim = ndimage.gaussian_filter(
        curv_rim, sigma_rims, radius=int(3*sigma_rims)
    )

    # Envelope: slope rim weighted 3 times more than the curvature rim
    envelope = (
        3 * slope_rim / np.nanmax(slope_rim) + curv_rim / np.nanmax(curv_rim)
    )
    envelope /= np.nanmax(envelope)

    # Map elevation to phase angles: min_elev -> pi/2, max_elev -> 5*pi/2
    elevation_range = dem_smooth.max() - dem_smooth.min()
    elevation_phase_angles = (
        (dem_smooth - dem_smooth.min()) / elevation_range
    ) * (2 * np.pi) + (np.pi / 2)

    # Harmonics
    foehn_fields = [
        1,
        np.cos(elevation_phase_angles),
        np.cos(2 * elevation_phase_angles),
        np.sin(elevation_phase_angles),
        np.sin(2 * elevation_phase_angles),
    ]

    # Final fields: harmonic * envelope * terrain_neg, normalized by the mean
    # of the non-zero absolute values
    terrain_neg = np.nanmax(dem) - dem
    for i, field in enumerate(foehn_fields):
        field = field * envelope * terrain_neg

        mean_abs = np.nanmean(np.abs(field[field != 0]))
        foehn_fields[i] = field / mean_abs

    return foehn_fields


def _create_patches(dem, neighbors: int = 8, fillna_value: float = 0):
    """Identify and label contiguous patches in a 2D array.

    Parameters
    ----------
    dem : array-like
        Array to be labeled. Any non-zero values are counted as features
        and zero values are considered the background.
    neighbors : int
        Connectivity for determining adjacent elements. Must be either 4 or 8.
        4 - considers only orthogonal neighbors (up, down, left, right).
        8 - includes diagonal neighbors as well.
        Default is 8.
    fillna_value : scalar
        Value to replace NaNs with in preprocessing. Default is 0 (background).

    Returns
    -------
    int32 array with a label for each patch, numbered from 1 in row-major
    order. The background is 0.

    See also
    --------
    scipy.ndimage.label
    """
    if neighbors == 4:
        struct = [[0, 1, 0],
                  [1, 1, 1],
                  [0, 1, 0]]
    elif neighbors == 8:
        struct = [[1, 1, 1],
                  [1, 1, 1],
                  [1, 1, 1]]
    else:
        raise ValueError("`neighbors` must be either 4 or 8")

    values = np.nan_to_num(dem, nan=fillna_value)
    labeled_array, _ = ndimage.label(values, structure=struct) # type: ignore

    return labeled_array


def _extract_patches(dem_thresh, patch_idxs, val=1.):
    """Extract specific patches from a DEM based on thresholded data.

    Parameters
    ----------
    dem_thresh : array-like
        A binary or thresholded version of the DEM where non-zero values
        indicate areas of interest. NaNs are considered background.
    patch_idxs : int or array-like
        The patch indices to extract from the labeled array. These
        correspond to the unique int identifiers assigned by
        _create_patches().
    val : array-like or float
        Value used to fill the selected patches. Either an array with the
        same shape as dem_thresh or a float. Default is 1.

    Returns
    -------
    np.ndarray or tuple of np.ndarray where the selected patch is filled with
    val and all other elements are 0.
        If patch_idxs is an int or contains a single index, returns an array.
        If patch_idxs contains multiple indices, returns a tuple of arrays.

    See also
    --------
    _create_patches
    """
    if not hasattr(patch_idxs, "__iter__"):
        patch_idxs = [patch_idxs]

    patches = _create_patches(dem_thresh)

    dem_patches = [np.where(patches == idx, val, 0) for idx in patch_idxs]

    return tuple(dem_patches) if len(patch_idxs) != 1 else dem_patches[0]


def _get_ridge_divide(dem, thresh, patch_idx):
    """Compute a thin line along a mountain ridge that separates its two sides
    (e.g. the N/S Alps divide).

    The flat area of the ridge is selected as the patch of values below
    thresh, reduced to its skeleton (medial axis) and extrapolated with a
    polynomial fit to the closest domain edges.

    Parameters
    ----------
    dem : array-like
        Field in which the ridge is flat, e.g. the logistic-transformed slope.
    thresh : float
        Values below thresh are considered part of the flat ridge area.
    patch_idx : int
        Index of the patch, as labeled by _create_patches, that corresponds
        to the ridge area.

    Returns
    -------
    Boolean array, True along the ridge divide (dilated by 3 pixels).

    See also
    --------
    _extract_patches, skimage.morphology.medial_axis,
    numpy.polynomial.polynomial.Polynomial.fit, scipy.ndimage.binary_dilation
    """

    height, width = dem.shape

    dem_bin = dem < thresh
    ridge_patch = _extract_patches(dem_bin, patch_idxs=patch_idx)
    # smooth the patch outline to avoid spurious branches in the skeleton
    ridge_patch = ndimage.gaussian_filter(ridge_patch, 9) > 0.2

    skeleton = medial_axis(ridge_patch)
    # TODO: np.where returns the pixels in row-major order, not ordered along
    # the skeleton, so the first/last points are not necessarily its endpoints
    y_idx, x_idx = np.where(skeleton)

    def extrapolate_to_closest_edge(x_idx, y_idx, width, height, before, n_fit=20, deg=2, offset=5):
        edge_idx = 0 if before else -1
        x_edge, y_edge = x_idx[edge_idx], y_idx[edge_idx]
        dist = min(
            abs(x_edge + 1 - width), x_edge, abs(y_edge + 1 - height), y_edge
        )
        if dist == 0:
            return [], []

        if before:
            idxs = slice(0, n_fit)
            u_extrap = np.linspace(-(dist + offset), -1, 2*dist)
        else:
            idxs = slice(-n_fit, None)
            u_extrap = np.linspace(len(x_idx), len(x_idx) + (dist + offset), 2*dist)

        u_fit = np.arange(n_fit)
        px = np.polynomial.polynomial.Polynomial.fit(u_fit, x_idx[idxs], deg=deg)
        py = np.polynomial.polynomial.Polynomial.fit(u_fit, y_idx[idxs], deg=deg)

        return px(u_extrap), py(u_extrap)

    x_before, y_before = extrapolate_to_closest_edge(x_idx, y_idx, width, height, before=True)
    x_after, y_after = extrapolate_to_closest_edge(x_idx, y_idx, width, height, before=False)

    # Combine extrapolated + original
    x_full = np.round(np.concatenate((x_before, x_idx, x_after))).astype(int)
    y_full = np.round(np.concatenate((y_before, y_idx, y_after))).astype(int)
    valid_mask = (
        (x_full >= 0) & (x_full < width) & (y_full >= 0) & (y_full < height)
    )
    x_smooth_idx = x_full[valid_mask]
    y_smooth_idx = y_full[valid_mask]

    ridge_divide = np.zeros_like(dem, dtype=bool)
    ridge_divide[y_smooth_idx, x_smooth_idx] = True
    ridge_divide = ndimage.binary_dilation(ridge_divide, iterations=3)

    return ridge_divide


def _foehn_names():
    """Return names for the arrays in output of the foehn function"""
    name_envelope = "FOEHN_ENVELOPE"
    name_cos1 = "FOEHN_COS1"
    name_cos2 = "FOEHN_COS2"
    name_sin1 = "FOEHN_SIN1"
    name_sin2 = "FOEHN_SIN2"

    return [name_envelope, name_cos1, name_cos2, name_sin1, name_sin2]


def compute_valley_effect(
    dem_ds,
    scales=59e3,
    valley_ref_elevation=1300,
    elevation_spread=170,
    depth_emphasis=1.75,
    ind_nans=[],
    crop=None,
    outdir="."
):
    """Wrapper to 'valley_effect' function to launch computations
    and save outputs as netCDF files.

    Parameters
    ----------
    dem_ds : xarray Dataset representing the DEM and its grid coordinates.
    scales : scalar or list of scalars
        Scale(s) in meters on which we want to compute the valley effect.
        Corresponds to the diameter of the kernel used to compute it.
    valley_ref_elevation : scalar
        Reference elevation for the logistic transformation (in meters).
        Default is 1300.
    elevation_spread : scalar
        Scale parameter controlling the spread of the logistic transformation
        (in meters). Default is 170.
    depth_emphasis : scalar
        Exponent applied to the elevation difference to emphasize deeper
        valleys. Default is 1.75.
    ind_nans (optional) : tuple of two 1D arrays
        Contains the (row, column) indices of the NaNs in the original DEM to be
        reassigned after computations. NaNs in the original DEM should be
        interpolated prior computations as they propagate in convolutions with
        the fast Fourier transform method (scipy.signal.convolve).
    crop (optional) : dict
        If specified the outputs are cropped to the given extend. Keys should be
        the coordinates labels of dem_ds and values should be slices of [min,max]
        extend. Default is None.
    outdir (optional) : string
        The path to the output directory. Save to working directory by default.

    See also
    --------
    valley_effect
    """
    hlp.check_dem(dem_ds)
    logger.info(f"***Starting valley effect computation for scales {scales} meters***")
    if not hasattr(scales, "__iter__"):
        scales = [scales]

    scales_pxl, _ = hlp.scale_to_pixel(scales, dem_ds)
    dem_val = hlp.get_da(dem_ds).values

    for idx, scale_pxl in enumerate(scales_pxl):
        logger.info(f"Computing scale {scales[idx]} meters ...")
        name = _valley_effect_name(scales[idx])
        array = valley_effect(
            dem_val,
            scale_pxl,
            valley_ref_elevation,
            elevation_spread,
            depth_emphasis
        )
        array[ind_nans] = np.nan
        # descriptor is normalized, hence dimensionless
        hlp.to_netcdf(array, dem_ds, name, crop, outdir)
        del array


def valley_effect(
    dem,
    size,
    valley_ref_elevation=1300,
    elevation_spread=170,
    depth_emphasis=1.75,
):
    """Compute a descriptor for the valley effect in alpine terrain.
    Identifies valleys by comparing a logistically transformed smoothed
    DEM with the original DEM, emphasizing deeper areas.

    Parameters
    ----------
    dem : array representing the DEM.
    size : int
        Size of the kernel for the convolution. Represents the diameter
        (i.e. scale) at which the DEM is smoothed (in pixel size).
    valley_ref_elevation : scalar
        Reference elevation for the logistic transformation (in meters).
        Default is 1300.
    elevation_spread : scalar
        Scale parameter controlling the spread of the logistic transformation
        (in meters). Default is 170.
    depth_emphasis : scalar
        Exponent applied to the elevation difference to emphasize deeper
        valleys. Default is 1.75.

    Returns
    -------
    array with valley effect values, divided by the mean of the non-zero
    values (dimensionless).

    See also
    --------
    scipy.signal.convolve, scipy.stats.logistic.cdf, circular_kernel

    References
    ----------
    This function implements the methodology developed by Louis Frey as
    part of his PhD thesis at ETHZ. Original implementation in R by
    Louis Frey (url_to_public_R_code.com).

    If you found this function useful for your own research, consider
    citing:

    .. code-block:: bibtex

        @article{lastname_papertitle_year,
          author = {First Name Last Name and Co-author's First Name Last Name},
          title = {Paper Title},
          journal = {Journal Name},
          volume = {Volume Number},
          number = {Issue Number},
          pages = {Page Range (e.g., 123--456)},
          year = {2025},
          doi = {DOI if available},
          url = {URL to the paper, if applicable}
        }
    """
    kernel = circular_kernel(size)
    kernel = kernel / kernel.sum()
    padded_dem = np.pad(dem, kernel.shape[0] // 2, mode="edge")
    dem_smooth = signal.convolve(padded_dem, kernel, mode="valid")

    dem_transf = stats.logistic.cdf(
        dem_smooth, loc=valley_ref_elevation, scale=elevation_spread
    )

    # Match original DEM range
    min_dem, max_dem = np.nanmin(dem), np.nanmax(dem)
    dem_transf = dem_transf * (max_dem - min_dem) + min_dem

    valley_pred = np.maximum(dem_transf - dem, 0.) ** depth_emphasis
    cond_mean = np.nanmean(valley_pred[valley_pred != 0])

    return valley_pred / cond_mean


def _valley_effect_name(scale):
    """Return name for the array in output of the valley_effect function"""
    return f"VALLEY_EFFECT_{int(scale)}M"
