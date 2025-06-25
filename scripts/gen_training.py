import ase
from janus_core.calculations.phonons import Phonons
import matplotlib.pyplot as plt
import numpy as np
import euphonic
from euphonic.cli.utils import _bands_from_force_constants, _get_debye_waller
from euphonic.powder import sample_sphere_structure_factor
from euphonic import ureg
import euphonic.plot as eplt
from pymatgen.ext.matproj import MPRester
import pymatgen.io.ase

# Insert your Materials Project API Key here
# https://next-gen.materialsproject.org/dashboard
MATPROJ_APIKEY = ''

def gen_single(mp_id, do_plot=False):

    with MPRester(MATPROJ_APIKEY) as mp:
        mpdat = mp.get_structure_by_material_id(mp_id)
        cc = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(mpdat)
    print(cc)
    matname = cc.get_chemical_formula()
    print(matname)
    
    # Runs calculations with ML-IAP using Janus-core and MACE
    phonons_mace = Phonons(
        struct=cc.copy(),
        arch="mace_mp",
        device="cpu",
        model="small",
        calc_kwargs={"default_dtype": "float64"},
        supercell=[2, 2, 2],
        displacement=0.01,
        temp_step=2.0,
        temp_min=0.0,
        temp_max=2000.0,
        minimize=True,
        minimize_kwargs={"filter_kwargs": {"hydrostatic_strain": False}, "fmax":0.1, 'optimizer':'MDMin'},
        force_consts_to_hdf5=True,
        plot_to_file=True,
        symmetrize=False,
        write_full=True,
        write_results=True,
    )
    phonons_mace.calc_force_constants()
    print(phonons_mace.struct.cell.cellpar())
    phonons_mace.calc_bands(write_bands=True)
    
    # Imports the calculated force constants into Euphonic (requires temporary files)
    fc = euphonic.ForceConstants.from_phonopy(summary_name=f'janus_results/{matname}-phonopy.yml', fc_name=f'janus_results/{matname}-force_constants.hdf5')
    bands, x_tick_labels, split_args = _bands_from_force_constants(fc, q_distance=0.01*ureg('1 / angstrom'), frequencies_only=True)
    bands.frequencies_unit = 'meV' # 'THz'
    if do_plot:
        fig = eplt.plot_1d(bands.get_dispersion())
    
    # Step to build an input vector.
    #input_vector = ...
    
    # The spec2d object here will be the output
    tt = 5*ureg('K')
    dw = _get_debye_waller(tt, fc) # use default grid spacing of 0.1 1/A
    qbins = np.linspace(0, 6, 101) * ureg('1 / angstrom')
    qbin_cens = (qbins[:-1] + qbins[1:]) / 2
    ebins = np.linspace(0, 60, 201) * ureg('meV')
    z_data = np.empty((len(qbin_cens), len(ebins) - 1))
    for iq, q in enumerate(qbin_cens):
        spec1d = sample_sphere_structure_factor(fc, q, dw=dw, temperature=tt, sampling='golden', jitter=True, energy_bins=ebins / 1.2)
        z_data[iq, :] = spec1d.y_data.magnitude
    spec2d = euphonic.Spectrum2D(qbins, ebins, z_data*spec1d.y_data.units)
    spec2d.broaden(x_width=0.2*ureg('1 / angstrom'), y_width=2*ureg('meV'))
    fig = eplt.plot_2d(spec2d, vmin=0, vmax=2)
    plt.savefig(f'janus_results/{matname}-powder.png')
    if do_plot:
        plt.show()

if __name__ == '__main__':
    gen_single('mp-18494', doplot=True) # Nd2MoO6
