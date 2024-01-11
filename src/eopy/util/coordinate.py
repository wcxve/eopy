import numpy as np
from astropy.coordinates import get_body, SkyCoord
from astropy.io import fits
from scipy.spatial.transform import Rotation as R

from .time import met_to_utc

__all__ = ['object_angle']


def sph_to_cart(theta_phi, deg=True):
    if deg:
        theta, phi = np.radians(np.transpose(theta_phi))
    else:
        theta, phi = np.transpose(theta_phi)

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    return np.transpose([sin_theta*cos_phi, sin_theta*sin_phi, cos_theta])


def radec_to_cart(ra_dec, deg=True):
    if deg:
        ra, dec = np.radians(np.transpose(ra_dec))
    else:
        ra, dec = np.transpose(ra_dec)

    sin_dec = np.sin(dec)
    cos_dec = np.cos(dec)
    sin_ra = np.sin(ra)
    cos_ra = np.cos(ra)

    return np.transpose([cos_dec*cos_ra, cos_dec*sin_ra, sin_dec])


def cart_to_sph(xyz, deg=True):
    x, y, z = np.transpose(xyz)
    norm_xy = np.linalg.norm((x, y), axis=0)

    theta = np.arctan2(norm_xy, z)
    phi = np.arctan2(y, x)
    if phi.shape == ():
        phi += 0.0 if phi > 0.0 else 2.0*np.pi
    else:
        neg = phi < 0.0
        phi[neg] = phi[neg] + 2.0*np.pi

    if deg:
        theta = np.degrees(theta)
        phi = np.degrees(phi)

    return np.transpose([theta, phi])


def cart_to_radec(xyz, deg=True):
    x, y, z = np.transpose(xyz)
    norm_xy = np.linalg.norm((x, y), axis=0)

    ra = np.arctan2(y, x)
    if ra.shape == ():
        ra += 0.0 if ra > 0.0 else 2.0*np.pi
    else:
        neg = ra < 0.0
        ra[neg] = ra[neg] + 2.0*np.pi
    dec = np.arctan2(z, norm_xy)

    if deg:
        ra = np.degrees(ra)
        dec = np.degrees(dec)

    return np.transpose([ra, dec])


def get_payload_to_sat_quat(sat):
    SAT = sat.upper()
    if SAT == 'FERMI':
        return (0.0, 0.0, 0.0, 1.0)
    elif SAT == 'HXMT':
        return (0.0, 0.0, 0.0, 1.0)
    elif SAT in ['GECAM-A', 'GECAM-B']:
        return (1.0, 0.0, 0.0, 0.0)
    elif SAT == 'GECAM-C':
        return (0.5, 0.5, 0.5, 0.5)
    else:
        raise ValueError('`sat` must be "Fermi", "HXMT", or "GECAM-A/B/C"')


def get_sat_to_payload_quat(sat):
    SAT = sat.upper()
    if SAT == 'FERMI':
        return (0.0, 0.0, 0.0, 1.0)
    elif SAT == 'HXMT':
        return (0.0, 0.0, 0.0, 1.0)
    elif SAT in ['GECAM-A', 'GECAM-B']:
        return (1.0, 0.0, 0.0, 0.0)
    elif SAT == 'GECAM-C':
        return (-0.5, -0.5, -0.5, 0.5)
    else:
        raise ValueError('`sat` must be "Fermi", "HXMT", or "GECAM-A/B/C"')


def payload_to_sat(coord, sat):
    quat = get_payload_to_sat_quat(sat)
    return R.from_quat(quat).apply(coord)


def sat_to_j2000(quat, coord):
    # matrix shape = (ntimes, 3, 3)
    # coord shape = (ndetectors, 3)
    # return (ntimes, ndetectors, 3)
    matrix = R.from_quat(quat).as_matrix()
    return np.einsum('TCc,Dc->TDC', matrix, coord)


def j2000_to_sat(quat, coord, is_inv=False):
    # quat shape = (ntimes, 4)
    # coord shape = (ntimes, 3)
    # is_inv: if quat is used to transform from j2000 to sat
    # return (ntimes, 3)
    if not is_inv:
        r = R.from_quat(quat).inv()
    else:
        r = R.from_quat(quat)
    return r.apply(coord)


def sat_to_payload(coord, sat):
    quat = get_sat_to_payload_quat(sat)
    return R.from_quat(quat).apply(coord)


def get_det_payload(sat, det=None):
    SAT = sat.upper()
    if SAT == 'FERMI':
        DET = 'GBM' if det is None else 'LAT'
        if DET == 'GBM':
            det_theta = np.array([
                20.58, 45.31, 90.21, 45.24,
                90.27, 89.79, 20.43, 46.18,
                89.97, 45.55, 90.42, 90.32,
                90.00, 90.00
            ])
            det_phi = np.array([
                 45.89,  45.11,  58.44, 314.87,
                303.15,   3.35, 224.93, 224.62,
                236.61, 135.19, 123.73, 183.74,
                  0.00, 180.00
            ])
        elif DET == 'LAT':
            det_theta = np.array([0.0])
            det_phi = np.array([0.0])
        else:
            ValueError(f'Fermi has no {DET}!')
    elif SAT in ['GECAM-A', 'GECAM-B']:
        det_theta = np.array([
             0.0, 40.0, 40.0, 40.0, 40.0,
            40.0, 40.0, 73.5, 73.5, 73.5,
            73.5, 73.5, 73.5, 73.5, 73.5,
            73.5, 73.5, 90.0, 90.0, 90.0,
            90.0, 90.0, 90.0, 90.0, 90.0
        ])
        det_phi = 90.0 + np.array([
              0.00, 242.17, 188.67, 135.17,  62.17,
              8.67, 315.17, 260.50, 224.50, 188.50,
            152.50, 116.50,  80.50,  44.50,   8.50,
            332.50, 296.50, 270.00, 215.00, 180.00,
            125.00,  90.00,  35.00,   0.00, 305.00
        ])
    elif SAT == 'GECAM-C':
        det_theta = np.array([
              0.0,  60.0,  60.0,  60.0,  60.0,  60.0,
            180.0, 120.0, 120.0, 120.0, 120.0, 120.0
        ])
        det_phi = np.array([
            0.0, 210.0, 150.0,  90.0,  30.0, 330.0,
            0.0, 150.0, 210.0, 270.0, 330.0,  30.0
        ])
    else:
        raise ValueError('`sat` must be "Fermi", "HXMT", or "GECAM-A/B/C"')

    det_theta_phi = np.column_stack((det_theta, det_phi))

    return sph_to_cart(det_theta_phi)


def haversine(lon1, lat1, lon2, lat2, deg=True):
    """Calculates the angular separation between two points using the
    haversine equation. If degrees are passed, degrees are returned. else
    the input/output is assumed to be radians.
    lon -> azimuth
    lat -> zenith

    Args:
        lon1 (float): lon/az of first point
        lat1 (float): lat/zen of first point
        lon2 (float): lon/az of second point
        lat2 (float): lat/zen of second point
        deg (bool, optional): True if input/output in degrees.

    Returns:
        float: Angular separation between points
    """
    if deg:
        lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    d_lat = 0.5 * (lat2 - lat1)
    d_lon = 0.5 * (lon2 - lon1)

    a = np.sin(d_lat) ** 2 + (np.sin(d_lon) ** 2 * np.cos(lat1) * np.cos(lat2))
    alpha = 2. * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    if deg:
        alpha = np.rad2deg(alpha)

    return alpha


def _angle_between1(src_in_payload, det_in_payload, deg=True):
    u = src_in_payload / np.linalg.norm(src_in_payload, axis=-1)[:, None]
    v = det_in_payload / np.linalg.norm(det_in_payload, axis=-1)[:, None]

    angle = np.arccos(np.einsum('tc,dc->td', u, v))

    if deg:
        angle = np.degrees(angle)

    return angle


def _angle_between2(src_j2000, det_j2000, deg=True):
    u = src_j2000 / np.linalg.norm(src_j2000)
    v = det_j2000 / np.linalg.norm(det_j2000, axis=-1)[:, :, None]

    angle = np.arccos(np.einsum('c,TDc->TD', u, v))

    if deg:
        angle = np.degrees(angle)

    return angle


def object_angle(obj, t0, tstart, tstop, posatt_file, det=None):
    with fits.open(posatt_file) as hdul:
        sat = hdul['PRIMARY'].header['TELESCOP']
        if sat == 'GLAST':
            sat = 'Fermi'
            ext = 'GLAST POS HIST'
            t = 'SCLK_UTC'
            Q = ['QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4']
        elif sat in ['GECAM-A', 'GECAM-B']:
            ext = 'Orbit_Attitude'
            t = 'TIME'
            Q = ['Q1', 'Q2', 'Q3', 'Q4']
        elif sat == 'GECAM-C':
            ext = 'Orbit_Attitude'
            t = 'TIME'
            Q_pre = ['Q2', 'Q3', 'Q4', 'Q1']  # until 59537900
            Q = ['Q1', 'Q2', 'Q3', 'Q4']  # from 59538144
        else:
            raise ValueError('POSATT not supported!')
        posatt = hdul[ext].data

    mask = (t0 + tstart <= posatt[t]) & (posatt[t] <= t0 + tstop)
    posatt = posatt[mask]
    met = posatt[t]
    quat = np.column_stack([posatt[q] for q in Q])
    # NOTE: This is a workaround for reading GECAM-C quat, which is in
    # NOTE: wrong order before MET=59537900. This should not be applied to
    # NOTE: the correct posatt file of GECAM-C.
    if sat == 'GECAM-C':
        mask = (met <= 59537900)
        if any(mask):
            quat[mask] = np.column_stack([posatt[mask][q] for q in Q_pre])
            print('WARNING: applied workaround for reading GECAM-C quats!')

    if type(obj) in [list, tuple, np.ndarray] and len(obj) == 2:
        src_j2000 = radec_to_cart(obj)
    elif type(obj) == str and obj.lower() == 'earth':
        src_j2000 = np.column_stack(
            (-posatt['X_J2000'], -posatt['Y_J2000'], -posatt['Z_J2000'])
        )
    elif type(obj) == str and obj.lower() in (
        'sun', 'moon', 'mercury', 'venus', 'earth-moon-barycenter', 'mars',
        'jupiter', 'saturn', 'uranus', 'neptune'
    ):
        # in GCRS frame
        src = get_body(obj, met_to_utc(met, sat, True))
        src_j2000 = src.cartesian.xyz.value.T
    elif type(obj) == str:
        src = SkyCoord.from_name(obj, frame='gcrs')
        src_j2000 = src.cartesian.xyz.value.T
    else:
        raise ValueError(f'wrong input {obj=}')

    src_sat = j2000_to_sat(quat, src_j2000)
    src_payload = sat_to_payload(src_sat, sat)
    det_payload = get_det_payload(sat, det)
    angle = _angle_between1(src_payload, det_payload)

    # src_j2000 = radec_to_cart((ra, dec))
    # det_payload = get_det_payload(sat, det)
    # det_sat = payload_to_sat(det_payload, sat)
    # det_j2000 = sat_to_j2000(quat, det_sat)
    # angle = _angle_between2(src_j2000, det_j2000)

    return np.column_stack((met, angle))
