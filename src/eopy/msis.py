"""Get atmosphere density from pymsis."""

from __future__ import annotations

import multiprocessing as mp
import platform
import sys

import astropy.units as u
import numpy as np
from astropy import coordinates as coords
from astropy.constants.iau2015 import R_earth
from astropy.time import Time
from numpy.typing import ArrayLike, NDArray
from pymsis import msis
from pymsis.utils import get_f107_ap
from tqdm import tqdm


def calc_element_density(
    date: str,
    lons: ArrayLike,
    lats: ArrayLike,
    alts: ArrayLike,
    f107: float,
    f107a: float,
    ap: ArrayLike,
    *,
    options: list[float] | None = None,
    version: float | str = 2.1,
    **kwargs: dict,
) -> NDArray:
    """Get MSIS profile along given locations.

    Parameters
    ----------
    date : str
        Date and time of interest, in ISO-8601 format.
    lons : array_like
        Longitudes of interest (from 0 to 360, or from -180 to 180), in unit
        degree.
    lats : array_like
        Latitudes of interest (from -90 to 90), in unit degree.
    alts : array_like
        Altitudes of interest, in unit km.
    f107 : float
        Daily F10.7 of the previous day for the given date.
    f107a : float
        F10.7 running 81-day average centered on the given date.
    ap : array_like
        | Ap for the given date, 1-6 only used if ``geomagnetic_activity=-1``.
        | [0] Daily Ap
        | [1] 3 hr ap index for current time
        | [2] 3 hr ap index for 3 hrs before current time
        | [3] 3 hr ap index for 6 hrs before current time
        | [4] 3 hr ap index for 9 hrs before current time
        | [5] Average of eight 3 hr ap indices from 12 to 33 hrs
        |     prior to current time
        | [6] Average of eight 3 hr ap indices from 36 to 57 hrs
        |     prior to current time
    options : array_like[25, float], optional
        A list of options (switches) to the model, if options is passed
        all keyword arguments specifying individual options will be ignored.
    version : float or str, optional
        MSIS version number, one of (0, 2.0, 2.1). The default is 2.1.
    **kwargs : dict
        Single options for the switches can be defined through keyword
        arguments.

    Returns
    -------
    density : ndarray
        | The atmospheric density:
        | [0] H  # density (m^-3),
        | [1] He # density (m^-3),
        | [2] N  # density (m^-3),
        | [3] O  # density (m^-3),
        | [4] Ar # density (m^-3)

    Other Parameters
    ----------------
    f107 : float
        Account for F10.7 variations
    time_independent : float
        Account for time variations
    symmetrical_annual : float
        Account for symmetrical annual variations
    symmetrical_semiannual : float
        Account for symmetrical semiannual variations
    asymmetrical_annual : float
        Account for asymmetrical annual variations
    asymmetrical_semiannual : float
        Account for asymmetrical semiannual variations
    diurnal : float
        Account for diurnal variations
    semidiurnal : float
        Account for semidiurnal variations
    geomagnetic_activity : float
        Account for geomagnetic activity
        (1 = Daily Ap mode, -1 = Storm-time Ap mode)
    all_ut_effects : float
        Account for all UT/longitudinal effects
    longitudinal : float
        Account for longitudinal effects
    mixed_ut_long : float
        Account for UT and mixed UT/longitudinal effects
    mixed_ap_ut_long : float
        Account for mixed Ap, UT, and longitudinal effects
    terdiurnal : float
        Account for terdiurnal variations

    Notes
    -----
    1. The 10.7 cm radio flux is at the Sun-Earth distance,
       not the radio flux at 1 AU.
    2. aps[1:] are only used when ``geomagnetic_activity=-1``.

    """
    if not (len(lons) == len(lats) == len(alts)):
        raise ValueError(
            f'length of lons ({len(lons)}), lats ({len(lats)}) and alts '
            f'({len(alts)}) must be equal'
        )

    if len(ap) != 7:
        raise ValueError(f'length of ap ({len(ap)}) must be 7')

    n = len(lons)
    dates = np.repeat(np.array(str(date), dtype=np.datetime64), n)
    f107s = np.full(n, float(f107))
    f107as = np.full(n, float(f107a))
    aps = np.full((n, 7), np.asarray(ap, dtype=np.float64))

    density = msis.run(
        dates=dates,
        lons=lons,
        lats=lats,
        alts=alts,
        f107s=f107s,
        f107as=f107as,
        aps=aps,
        options=options,
        version=version,
        **kwargs,
    )

    # NOTE: nan usually occurs at low altitude for low density components,
    #  we set these nans to zeros.
    density[np.isnan(density)] = 0.0

    density = np.column_stack(
        (
            # H
            density[:, 5],
            # He
            density[:, 4],
            # N2, N, and N in NO for MSIS 2.1
            density[:, 1] * 2.0 + density[:, 7] + density[:, 9],
            # O2, O, Anomalous O, and O in NO for MSIS 2.1
            density[:, 2] * 2.0
            + density[:, 3]
            + density[:, 8]
            + density[:, 9],
            # Ar
            density[:, 6],
        )
    )

    return density


def calc_column_density(
    src_radec: ArrayLike,
    loc_j2000: ArrayLike,
    utc: ArrayLike,
    step_size: float = 0.5,
    lower_alt: float = 5.0,
    upper_alt: float = 550.0,
    f: float = 1 / 298.257,
    mass: bool = False,
    profile: bool = False,
    cores: int | None = None,
    name: str | None = None,
    f107: ArrayLike | None = None,
    f107a: ArrayLike | None = None,
    ap: ArrayLike | None = None,
    options: list[float] | None = None,
    version: float | str = 2.1,
    progress: bool = True,
    **kwargs: dict,
) -> NDArray:
    """Compute atmospheric column density between `src_radec` and `loc_j2000`.

    Parameters
    ----------
    src_radec : array_like
        R. A. and Dec. of the source.
    loc_j2000 : array_like
        Array of J2000 coordinate (X, Y, Z), in unit m.
    utc : array_like
        Array of UTC in ISO-8601 format.
    step_size : float, optional
        The sampling step size in unit km. The default is 0.5.
    lower_alt : float, optional
        Below `lower_alt` km, assume the air density is high enough to absorb
        all X-ray photons. The default is 5 km.
    upper_alt : float, optional
        Above `upper_alt` km, assume the air density is too low to absorb X-ray
        photons. The default is 500 km.
    f : float, optional
        The flattening factor of Earth. The default is 1/298.257.
    mass : bool, optional
        Whether to return column density in unit of g cm^-2. False to return
        in units of atom cm^-2. The default is False.
    profile : bool, optional
        Whether to return the column profile along the path. Return total
        column density if False. The default is False.
    f107 : float
        Daily F10.7 of the previous day for the given UTC.
    f107a : float
        F10.7 running 81-day average centered on the given UTC.
    ap : array_like
        | Ap for the given UTC, 1-6 only used if ``geomagnetic_activity=-1``.
        | [0] Daily Ap
        | [1] 3 hr ap index for current time
        | [2] 3 hr ap index for 3 hrs before current time
        | [3] 3 hr ap index for 6 hrs before current time
        | [4] 3 hr ap index for 9 hrs before current time
        | [5] Average of eight 3 hr ap indices from 12 to 33 hrs
        |     prior to current time
        | [6] Average of eight 3 hr ap indices from 36 to 57 hrs
        |     prior to current time
    options : array_like[25, float], optional
        A list of options (switches) to the model, if options is passed
        all keyword arguments specifying individual options will be ignored.
    version : float or str, optional
        MSIS version number, one of (0, 2.0, 2.1). The default is 2.1.
    cores : int, optional
        Number of CPU to use. The default is ``max(1, mp.cpu_count()-1)``.
    name : str, optional
        This must be given as ``name=__name__`` when in Windows platform.
    progress : bool, optional
        Whether to show progress bar of calculation. The default is True.
    **kwargs : dict, optional
        Other parameters forwarded to :func:`element_density`.

    Returns
    -------
    density : ndarray
        The column density of H, He, N, O and Ar atoms, in shape (n_time, 5) if
        ``profile=False`` else (n_time, n_grid, 5).
    path_loc : ndarray
        Sampling location grids, returned only when ``profile=True``.

    """
    if cores is None:
        cores = max(1, mp.cpu_count() - 1)
    else:
        cores = int(cores)
    if 1 < cores < len(loc_j2000):
        if platform.system() == 'Windows' and name != '__main__':
            raise RuntimeError(
                "call the function under ``if __name__ == '__main__':`` "
                'and set ``name=__name__`` when multiprocessing'
            )
        parallel = True
    else:
        parallel = False

    if np.shape(src_radec) != (2,):
        raise ValueError('`src_radec` must be length of 2')

    utc = Time(np.atleast_1d(utc), format='isot', scale='utc')
    loc_j2000 = np.atleast_2d(loc_j2000)

    if f107 is None or f107a is None or ap is None:
        f107_, f107a_, ap_ = get_f107_ap(utc.datetime)
        if f107 is None:
            f107 = f107_
        if f107a is None:
            f107a = f107a_
        if ap is None:
            ap = ap_

    f107 = np.atleast_1d(f107)
    f107a = np.atleast_1d(f107a)
    ap = np.atleast_2d(ap)
    if not (len(f107) == len(f107a) == len(ap) == len(utc)):
        raise ValueError(
            f'length of utc ({len(utc)}), f107 ({len(f107)}), f107a '
            f'({len(f107a)}), and ap ({len(ap)}) must be equal'
        )

    step_size = step_size * 1000.0  # unit: m
    lower_alt = lower_alt * 1000.0 + R_earth.value  # unit: m
    upper_alt = upper_alt * 1000.0 + R_earth.value  # unit: m

    src_radec = np.asarray(src_radec, dtype=np.float64)
    src = coords.GCRS(
        ra=src_radec[0].repeat(utc.size) * u.deg,
        dec=src_radec[1].repeat(utc.size) * u.deg,
        obstime=utc,
    ).transform_to(
        coords.ITRS(
            obstime=utc, representation_type=coords.WGS84GeodeticRepresentation
        )
    )

    loc_j2000 = np.atleast_2d(loc_j2000)
    loc = coords.GCRS(
        x=loc_j2000[:, 0] * u.m,
        y=loc_j2000[:, 1] * u.m,
        z=loc_j2000[:, 2] * u.m,
        representation_type='cartesian',
        obstime=utc,
    ).transform_to(
        coords.ITRS(
            obstime=utc, representation_type=coords.WGS84GeodeticRepresentation
        )
    )

    src_x, src_y, src_z = src.cartesian.xyz.value
    loc_x, loc_y, loc_z = loc.cartesian.xyz.value  # unit: m

    # The following code solves the intersection of the ray with the ellipsoid
    # The ray can be described as:
    # r = loc + t * src
    # where t is the parameter of the ray
    # The earth can be described as an ellipsoid:
    # x^2 + y^2 + (z/f)^2 = R^2
    # Note that this is not the exact start and end point,
    # but a good enough approximation, see e.g. Harmon et al. (2002),
    # thus `lower_alt` and `upper_alt` is assumed to be low and high enough
    z_factor = (1.0 - f) ** (-2.0)
    a = src_x * src_x + src_y * src_y + z_factor * src_z * src_z
    b = 2.0 * (loc_x * src_x + loc_y * src_y + z_factor * loc_z * src_z)
    tmp = loc_x * loc_x + loc_y * loc_y + z_factor * loc_z * loc_z
    c_lower = tmp - lower_alt * lower_alt
    c_upper = tmp - upper_alt * upper_alt
    b2 = b * b
    a4 = 4.0 * a
    d_lower = b2 - a4 * c_lower
    d_upper = b2 - a4 * c_upper

    a2 = 2.0 * a
    neg_b = -b

    # There are two cases to ignore when calculating the density along the ray:
    # 1. ray goes through the lowest layer, which means the ray is blocked by
    #    the layer.
    # 2. ray does not intersect with the highest layer, which means the ray is
    #    not extincted by the lower layers.

    # Select the rays that do not intersect with the non-transparent layer.
    lmask = d_lower < 0
    # Or the location is above the layer and the ray is going upward,
    # i.e. the ray is going away from the earth.
    # We check the the sign of the intersection parameter is all negative to
    # determine if this is the case.
    # If the location is within the layer, the two intersection parameters
    # will be positive and negative respectively, and hence not all negative.
    mask = ~lmask
    # check the sign of the largest intersection parameter, -b + sqrt(d) / (2a)
    lmask[mask] = (neg_b[mask] + np.sqrt(d_lower[mask])) / a2[mask] <= 0.0

    # Select the rays that intersect with the transparent layer.
    umask = d_upper > 0
    # Check the intersection parameter is positive.
    # This is to exclude the case where the location is above the layer and
    # the ray is going upward, hence the intersection is behind the ray.
    mask = ~umask
    umask[mask] = (neg_b[mask] + np.sqrt(d_upper[mask])) / a2[mask] > 0.0

    # Combine the two masks
    mask = lmask & umask

    neg_b = -b[mask]
    sqrt_d = np.sqrt(d_upper[mask])
    a2 = 2.0 * a[mask]
    path_start = (neg_b - sqrt_d) / a2
    path_start[path_start < 0.0] = 0.0
    path_end = (neg_b + sqrt_d) / a2
    length = [
        np.arange(start, end + step_size, step_size)
        for start, end in zip(path_start, path_end, strict=True)
    ]

    utc_ = utc[mask]
    src_ = src[mask]
    loc_ = loc[mask]
    path = [
        coords.ITRS(
            loc_[i].cartesian + src_[i].cartesian * length[i] * u.m,
            obstime=utc_[i],
            representation_type=coords.WGS84GeodeticRepresentation,
        ).earth_location
        for i in range(np.sum(mask))
    ]

    kwargs |= {'options': options, 'version': version}
    utc_masked = utc[mask]
    f107_masked = f107[mask]
    f107a_masked = f107a[mask]
    ap_masked = ap[mask]

    if parallel:
        with mp.Pool(cores) as pool:
            results = [
                pool.apply_async(
                    func=calc_element_density,
                    args=(
                        utc_masked[i].value,
                        path[i].lon.value,
                        path[i].lat.value,
                        path[i].height.to(u.km).value,
                        f107_masked[i],
                        f107a_masked[i],
                        ap_masked[i],
                    ),
                    kwds=kwargs,
                )
                for i in range(np.sum(mask))
            ]

            results = [
                r.get()
                for r in tqdm(
                    results,
                    desc=f'Running on {cores} CPUs',
                    file=sys.stdout,
                    disable=not progress,
                )
            ]

    else:
        results = [
            calc_element_density(
                utc_masked[i].value,
                path[i].lon.value,
                path[i].lat.value,
                path[i].height.to(u.km).value,
                f107_masked[i],
                f107a_masked[i],
                ap_masked[i],
                **kwargs,
            )
            for i in tqdm(
                range(np.sum(mask)),
                desc='Running on 1 CPU',
                file=sys.stdout,
                disable=not progress,
            )
        ]

    if profile:
        # shape = (t, locs, elements)
        if len(results) > 0:
            density = _to_density_array(results)
        else:
            density = np.zeros((0, 1, 5), dtype=np.float32)

        # column profile of H, He, N, O and Ar, in shape (t, locs, 5)
        col_atoms = np.zeros((utc.size, density.shape[1], 5), dtype=np.float32)
        col_atoms[~lmask] = np.inf

        # atom/m^3 -> atom/m^2 -> atom/cm^2
        col_atoms[mask] = density * step_size * 1e-4

        path_loc = np.empty((utc.size, density.shape[1], 3), dtype=np.float32)
        path_loc[~lmask] = (
            -181.0,
            -91.0,
            (lower_alt - R_earth.value) * 1e-3,
        )
        path_loc[~umask] = [
            -181.0,
            -91.0,
            (upper_alt - R_earth.value) * 1e-3,
        ]
        if mask.any():
            path_loc[mask] = _to_loc_array(
                [
                    np.column_stack(
                        (i.lon.value, i.lat.value, i.height.value * 1e-3)
                    )
                    for i in path
                ]
            )
    else:
        # shape = (t, elements)
        if len(results) > 0:
            density = np.vstack([r.sum(0) for r in results])
        else:
            density = np.empty((0, 5), dtype=np.float32)

        # column density of H, He, N, O and Ar, in shape (t, 5)
        col_atoms = np.zeros((utc.size, 5), dtype=np.float32)
        col_atoms[~lmask] = np.inf

        # atom/m^3 -> atom/m^2 -> atom/cm^2
        col_atoms[mask] = density * step_size / 10000.0

    if mass:
        # atom/cm^2 to g/cm^2
        atom_mass = np.array([1.00794, 4.002602, 14.0067, 15.9994, 39.948])
        col_atoms *= atom_mass * 1.6605390666e-24

    if profile:
        return col_atoms, path_loc
    else:
        return col_atoms


def _to_density_array(v):
    # adapted from https://stackoverflow.com/a/38619350
    lens = np.array([len(item) for item in v])
    mask = lens[:, None, None] > np.tile(np.arange(lens.max()), (5, 1)).T
    out = np.zeros(mask.shape, dtype=np.float32)
    out[mask] = np.concatenate(v, None)
    return out


def _to_loc_array(v):
    # adapted from https://stackoverflow.com/a/38619350
    lens = np.array([len(item) for item in v])
    mask = lens[:, None, None] > np.tile(np.arange(lens.max()), (3, 1)).T
    out = np.empty(mask.shape, dtype=np.float32)
    out[..., 0] = -181.0
    out[..., 1] = -91.0
    out[..., 2] = -1.0
    out[mask] = np.concatenate(v, None)
    return out
