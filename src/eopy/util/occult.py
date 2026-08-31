from __future__ import annotations

import astropy.units as u
import numpy as np
from astropy import coordinates as coords
from astropy.constants.iau2015 import R_earth
from astropy.io import fits
from astropy.time import Time
from numpy.typing import ArrayLike, NDArray

from ..msis import calc_column_density
from ..xcom import calculate_cross_section
from .coordinate import radec_to_cart
from .misc import _ORBIT_CONFIG, get_sat_j2000, telescope_to_sat
from .time import met_to_utc

__all__ = ['calc_tangent_height', 'calc_transmis_coeff', 'get_oti']


def calc_tangent_height(
    src_radec: ArrayLike, utc: ArrayLike, file: str
) -> NDArray:
    """Calculate tangent height along the line of sight.

    Parameters
    ----------
    src_radec : array_like
        J2000 coordinate (R.A., Dec.) of source.
    utc : array_like
        UTC in ISO-8601 format.
    file : str
        Path of HXMT orbit, GECAM posatt, or Fermi poshist file.

    Returns
    -------
    height : ndarray
        Tangent height in unit of km.

    """
    f = 1 / 298.257
    z_factor = (1 - f) ** (-2)

    utc = Time(np.atleast_1d(utc), scale='utc')

    src_radec = np.array(src_radec, dtype=np.float64, order='C')
    src = coords.GCRS(
        ra=src_radec[0].repeat(utc.size) * u.deg,
        dec=src_radec[1].repeat(utc.size) * u.deg,
        obstime=utc,
    ).transform_to(
        coords.ITRS(
            obstime=utc, representation_type=coords.WGS84GeodeticRepresentation
        )
    )

    loc_j2000 = get_sat_j2000(utc.value, file)
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

    src_x, src_y, src_z = src.x.value, src.y.value, src.z.value
    loc_x, loc_y, loc_z = loc.x.value, loc.y.value, loc.z.value  # unit: m
    r = R_earth.value
    r2 = r * r
    a = src_x * src_x + src_y * src_y + z_factor * src_z * src_z
    b = 2 * (loc_x * src_x + loc_y * src_y + z_factor * loc_z * src_z)
    c = loc_x * loc_x + loc_y * loc_y + z_factor * loc_z * loc_z - r2
    hmin = np.sqrt(c + r2 - b * b / 4.0 / a) - r
    mask = a * b > 0.0
    hmin[mask] = (
        np.linalg.norm(
            [loc_x[mask], loc_y[mask], loc_z[mask] / (1 - f)], axis=0
        )
        - r
    )

    return hmin / 1000.0


def calc_transmis_coeff(
    src_radec, utc, energy, orbit_file, step_size=0.5, name=None
):
    src = np.array(src_radec, dtype=np.float64, order='C')

    energy = np.array(energy, dtype=np.float64, order='C')
    energy *= 1000.0  # keV to eV

    loc = get_sat_j2000(utc, orbit_file)

    # shape = (t, 5)
    d = calc_column_density(src, loc, utc, step_size=step_size, name=name)

    # shape = (e, 5)
    cs = np.column_stack(
        [
            calculate_cross_section(z, energy)['total'] * 1e-24  # barn to cm2
            for z in (1, 2, 7, 8, 18)  # H, He, N, O, Ar
        ]
    )

    # shape = (t, e)
    coeff = np.exp(-np.einsum('tE,eE->te', d, cs, optimize='optimal'))

    return coeff


def get_oti(obj, file, alt_range):
    """Get Occultation Time Intervals given altitude range of line of sight.

    Parameters
    ----------
    obj : str, or array_like of shape (2,)
        Celestial object name, or J2000 coordinate (R.A., Dec.).
    file : str
        Path of HXMT's orbit, GECAM's posatt or Fermi's poshist file.
    alt_range : array_like of shape (2,)
        Range of line-of-sight altitude, in unit of km.

    Returns
    -------
    oti : ndarray of shape (n, 2)
        Occultation Time Intervals.

    """
    with fits.open(file) as hdu_list:
        telescope = hdu_list['PRIMARY'].header['TELESCOP']
        sat = telescope_to_sat(telescope)
        orbit_ext, t, *_ = _ORBIT_CONFIG[sat]
        met = hdu_list[orbit_ext].data[t]

    utc = met_to_utc(met, sat)

    if type(obj) in [list, tuple] and len(obj) == 2:
        src_j2000 = radec_to_cart(obj)
    elif type(obj) == str and obj.lower() in (
        'sun',
        'moon',
        'mercury',
        'venus',
        'earth-moon-barycenter',
        'mars',
        'jupiter',
        'saturn',
        'uranus',
        'neptune',
    ):
        src = coords.get_body(obj, met_to_utc(met, sat, True))  # in GCRS frame
        src_j2000 = src.cartesian.xyz.value.T
    elif type(obj) == str:
        src = coords.SkyCoord.from_name(obj, frame='gcrs')
        src_j2000 = src.cartesian.xyz.value.T
    else:
        raise ValueError(f'wrong input {obj=}')

    h = calc_tangent_height(src_j2000, utc, file)

    hmask = (alt_range[0] <= h) & (h <= alt_range[1])

    met = met[hmask]
    h = h[hmask]

    diff = met[1:] - met[:-1]
    idx = np.flatnonzero(diff >= 2.0)
    idx = np.sort([0, *idx, *(idx + 1), len(met) - 1]).reshape(-1, 2)

    return met[idx], h[idx]
