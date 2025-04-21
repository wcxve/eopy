import numpy as np

from ._data_converter import NameProcess
from .interpolators import Interpolators, Material, MaterialFactory

_AVOGADRO = 0.60221367  # * 1e+24 mole^-1

ENERGY_GRID_DEFAULT = np.array(
    [
        1.0e03,
        1.5e03,
        2.0e03,
        3.0e03,
        4.0e03,
        5.0e03,
        6.0e03,
        8.0e03,
        1.0e04,
        1.5e04,
        2.0e04,
        3.0e04,
        4.0e04,
        5.0e04,
        6.0e04,
        8.0e04,
        1.0e05,
        1.5e05,
        2.0e05,
        3.0e05,
        4.0e05,
        5.0e05,
        6.0e05,
        8.0e05,
        1.0e06,
        1.022e06,
        1.25e06,
        1.5e06,
        2.0e06,
        2.044e06,
        3.0e06,
        4.0e06,
        5.0e06,
        6.0e06,
        7.0e06,
        8.0e06,
        9.0e06,
        1.0e07,
        1.1e07,
        1.2e07,
        1.3e07,
        1.4e07,
        1.5e07,
        1.6e07,
        1.8e07,
        2.0e07,
        2.2e07,
        2.4e07,
        2.6e07,
        2.8e07,
        3.0e07,
        4.0e07,
        5.0e07,
        6.0e07,
        8.0e07,
        1.0e08,
        1.5e08,
        2.0e08,
        3.0e08,
        4.0e08,
        5.0e08,
        6.0e08,
        8.0e08,
        1.0e09,
        1.5e09,
        2.0e09,
        3.0e09,
        4.0e09,
        5.0e09,
        6.0e09,
        8.0e09,
        1.0e10,
        1.5e10,
        2.0e10,
        3.0e10,
        4.0e10,
        5.0e10,
        6.0e10,
        8.0e10,
        1.0e11,
    ],
    dtype='d',
)


def calculate_attenuation(material: Material, energy: np.ndarray = None):
    """
    Calculate attenuation (cm2/gramm) for gamma-ray (at energies between 1 keV and 100 GeV) for next process:

        * Coherent scattering
        * Incoherent (Compton) scattering
        * Photoelectric absorption
        * Pair production in the field of the atomic nucleus and in the field of the atomic electrons

    Based on NIST XCOM data: https://www.nist.gov/pml/xcom-photon-cross-sections-database

    Parameters
    ----------
    material
            special class description simple material or compound

    energy
            energies of gamma-quanta in eV, used `ENERGY_GRID_DEFAULT` by default

    Returns
    -------
    data : ndarray with attenuation in cm2/gramm
    """
    if not isinstance(material, Material):
        raise Exception('Except material')

    if len(material) == 1:
        element = material.elements_by_Z[0]
        data = calculate_cross_section(element, energy)
        # Attenutaion coefficient = macro_cross_secction/denisty = \
        # = micro_cross_section/atom_weight[gr]
        # atom_weight[gr] = atom_weight[amu] / AVOGADRO
        atom_weigth = MaterialFactory.get_element_mass(element)
        for name in data.dtype.names:
            if name != 'energy':
                data[name] *= _AVOGADRO / atom_weigth
        return data
    elif len(material) > 1:
        atom_weights_amu = MaterialFactory.get_elements_mass_list(
            material.elements_by_Z
        )
        data = calculate_cross_section(material.elements_by_Z[0], energy)
        for name in data.dtype.names:
            data[name] *= material.weights[0] * _AVOGADRO / atom_weights_amu[0]

        for atom_weight_amu, element, weight in zip(
            atom_weights_amu[1:],
            material.elements_by_Z[1:],
            material.weights,
            strict=False,
        ):
            temp = calculate_cross_section(element, energy)
            for name in data.dtype.names:
                data[name] += temp[name] * weight * _AVOGADRO / atom_weight_amu
        return data
    else:
        raise Exception('Empty material')


def calculate_cross_section(
    element: int | str, energy: np.ndarray = None
) -> np.ndarray:
    """
    Calculate cross-section (barn/atom) for gamma-ray (at energies between 1 keV and 100 GeV) for next process:

        * Coherent scattering
        * Incoherent (Compton) scattering
        * Photoelectric absorption
        * Pair production in the field of the atomic nucleus and in the field of the atomic electrons

    Based on NIST XCOM data: https://www.nist.gov/pml/xcom-photon-cross-sections-database

    Parameters
    ----------
    element
            atomic number or symbol of element

    energy
            energies of gamma-quanta in eV, used `ENERGY_GRID_DEFAULT` by default

    Returns
    -------
    data : ndarray with cross-section in barn/atom
    """
    if energy is None:
        energy = ENERGY_GRID_DEFAULT

    if not isinstance(element, int):
        element = MaterialFactory.get_element_from_symbol(element)

    n = len(energy)
    dtype = np.dtype(
        [
            ('energy', 'd'),
            (NameProcess.COHERENT, 'd'),
            (NameProcess.INCOHERENT, 'd'),
            (NameProcess.PHOTOELECTRIC, 'd'),
            (NameProcess.PAIR_ATOM, 'd'),
            (NameProcess.PAIR_ELECTRON, 'd'),
            ('total_without_coherent', 'd'),
            ('total', 'd'),
        ]
    )

    data = np.zeros(n, dtype=dtype)
    data['energy'] = np.asarray(energy)

    _INTERPOLATOS = Interpolators()

    for k, v in _INTERPOLATOS.get_interpolators(element).items():
        data[k] = v(data['energy'])
        data['total'] += data[k]
    data['total_without_coherent'] -= data[NameProcess.COHERENT]
    return data
