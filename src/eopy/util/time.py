from astropy.time import Time
from astropy.units import s

__all__ = ['met_to_utc', 'utc_to_met']


def get_utc0(sat):
    SAT = sat.upper()
    if SAT == 'SWIFT':
        utc0 = '2001-01-01T00:00:00'
    elif SAT == 'FERMI':
        utc0 = '2001-01-01T00:00:00'
    elif SAT == 'HXMT':
        utc0 = '2012-01-01T00:00:00'
    elif SAT in ['GECAM-A', 'GECAM-B']:
        utc0 = '2019-01-01T00:00:00'
    elif SAT == 'GECAM-C':
        utc0 = '2021-01-01T00:00:00'
    else:
        raise ValueError(
            "`sat` must be one of 'Swift', 'Fermi', 'HXMT', and 'GECAM-A/B/C'"
        )
    return utc0


def met_to_utc(met, sat, return_astropy_time=False, UTCFINIT=None):
    SAT = sat.upper()

    # Guide to Times in Swift FITS Files:
    # https://swift.gsfc.nasa.gov/analysis/suppl_uguide/time_guide.html
    if SAT == 'SWIFT' and UTCFINIT is None:
        raise ValueError('`UTCFINIT` must be provided for Swift!')
    else:
        UTCFINIT = 0.0

    utc0 = get_utc0(sat)
    utc = Time(utc0, format='isot', scale='utc') + (met + UTCFINIT) * s
    if return_astropy_time:
        return utc
    else:
        return utc.isot


def utc_to_met(utc, sat, UTCFINIT=None):
    SAT = sat.upper()

    # Guide to Times in Swift FITS Files:
    # https://swift.gsfc.nasa.gov/analysis/suppl_uguide/time_guide.html
    if SAT == 'SWIFT' and UTCFINIT is None:
        raise ValueError('`UTCFINIT` must be provided for Swift!')
    else:
        UTCFINIT = 0.0

    utc0 = get_utc0(sat)

    return (Time(utc, scale='utc') - Time(utc0, scale='utc')).sec - UTCFINIT
