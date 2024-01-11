import numpy as np
from astropy.io import fits

from .time import utc_to_met

__all__ = ['get_sat_j2000', 'telescope_to_sat']


_ORBIT_CONFIG = {
    'HXMT': ('Orbit',
             'TIME',
             'X', 'Y', 'Z',
             'VX', 'VY', 'VZ'),
    'GECAM': ('Orbit_Attitude',
              'TIME',
              'X_J2000', 'Y_J2000', 'Z_J2000',
              'VX_J2000', 'VY_J2000', 'VZ_J2000'),
    'Fermi': ('GLAST POS HIST',
              'SCLK_UTC',
              'POS_X', 'POS_Y', 'POS_Z',
              'VEL_X', 'VEL_Y', 'VEL_Z'),
}
_ORBIT_CONFIG['GECAM-A'] = _ORBIT_CONFIG['GECAM']
_ORBIT_CONFIG['GECAM-B'] = _ORBIT_CONFIG['GECAM']
_ORBIT_CONFIG['GECAM-C'] = _ORBIT_CONFIG['GECAM']


def get_sat_j2000(time, file):
    """
    Compute J2000 coordinates of satellite given the time and orbit file.

    Parameters
    ----------
    time : array_like, shape is () or (t,), and dtype is str, int, or float
        Time info, either UTC in ISO-8601 format, or MET time.
    file : str
        Path of HXMT's orbit, GECAM's posatt or Fermi's poshist file.

    Returns
    -------
    pos : array of shape (3,) or (t, 3)
        J2000 coordinates of satellite at given the time point.

    """
    with fits.open(file) as hdu_list:
        telescope = hdu_list['PRIMARY'].header['TELESCOP']
        sat = telescope_to_sat(telescope)
        orbit_ext, t, x, y, z, vx, vy, vz = _ORBIT_CONFIG[sat]

        orbit = hdu_list[orbit_ext].data

    time = np.array(time, order='C')
    if time.dtype == float or time.dtype == int:
        met = np.array(time, dtype=np.float64, order='C')
    else:
        met = utc_to_met(time, sat)

    # filter out some orbit info will not be used
    time_mask = (met.min() - 1 <= orbit[t]) & (orbit[t] <= met.max() + 1)
    orbit = orbit[time_mask]

    # get corresponding index of MET in orbit info
    indices = np.searchsorted(np.sort(orbit[t]), met) - 1

    # get time difference
    delta_t = met - orbit[t][indices]

    # use velocities to interpolate coordinate
    sat_x = orbit[x][indices] + delta_t * orbit[vx][indices]
    sat_y = orbit[y][indices] + delta_t * orbit[vy][indices]
    sat_z = orbit[z][indices] + delta_t * orbit[vz][indices]

    if met.shape == ():
        pos = np.array([sat_x, sat_y, sat_z])
    else:
        pos = np.column_stack((sat_x, sat_y, sat_z))

    return pos


def telescope_to_sat(telescope: str) -> str:
    if telescope == 'HXMT':
        return 'HXMT'
    elif telescope in ['GECAM-A', 'GECAM-B', 'GECAM-C']:
        return telescope
    elif telescope == 'HEBS':
        return 'GECAM-C'
    elif telescope == 'GLAST':
        return 'Fermi'
    elif telescope in ['XRT', 'BAT']:
        return 'Swift'
    else:
        raise ValueError(f'satellite name for {telescope=} is not defined')
