import sys 
import numpy as np
from tools.solvers.kolmogorov import BaseFlow, NumericSolver, TimeStepper3D
from tools.misc_tools import get_ic_file, log_all_params, on_local_device
import logging

logger = logging.getLogger(__name__)

material_params = {'W': 20,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3}

system_params = {'ndim': 3,
                 'Lx': 3 * np.pi,
                 'Lz': 4 * np.pi,
                 'n': 1}

solver_params = {'Nx': 128,
                 'Ny': 64,
                 'Nz': 32,
                 'dt': 2e-3,
                 'c': 0}

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
elif len(sys.argv) == 4:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
    input_val = float(sys.argv[3])
else:
    raise Exception('Need more inputs!')    

symmetry_mode = False
kwargs = {}
if setting_mode == 0:
    # Localised AH as W decreases
    Lz = 8 * np.pi

    solver_params['Nz'] = int(16 * Lz  / np.pi)
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = input_val

    ic_dict_if_reinit = {'W': 18}
    suffix_end = 'localised'

    translate = False
    
elif setting_mode == 1:
    # Get Periodic AH from 2D AH. m=1 mode branch
    Lz = 2 * np.pi

    solver_params['Nz'] = 64
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'Lz':3/2*np.pi, 'Nz': 32 }
    suffix_end = 'periodic'
    translate = False

elif setting_mode == 2:
        # Get Periodic AH from 2D AH. m=1 mode branch
    Lz = 3/2 * np.pi

    solver_params['Nz'] = 32
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'Lz':np.pi, 'Nz': 32, 'suffix': 'recent-periodic' }
    suffix_end = 'periodic-yz'
    symmetry_mode = 'yz'
    translate = False

elif setting_mode == 3:
        # Get Periodic AH from 2D AH. PRODUCTION METHOD OF OBTAINING PERIODIC AH FROM 2D AH
    Lz = np.pi

    solver_params['Nz'] = 32
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 4e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'ndim': 2, 'noise_coeff':1e-3, 'subdir':'arrowhead_2D', 'suffix': 'recent-', 'Nx': 128, 'Ny':256 }
    suffix_end = 'periodic'
    symmetry_mode = False

elif setting_mode == 4:
        # Get Periodic AH from 2D AH. PRODUCTION METHOD OF OBTAINING PERIODIC AH FROM 2D AH
    Lz = np.pi / 2

    solver_params['Nz'] = 16
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 5e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'ndim': 2, 'noise_coeff':1e-3, 'subdir':'arrowhead_2D', 'suffix': 'recent-', 'Nx': 128, 'Ny':256 }
    suffix_end = 'periodic'
    symmetry_mode = False

elif setting_mode == 5:

    Lz = 1/4 * np.pi

    solver_params['Nz'] = 16
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 4e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'ndim': 2, 'noise_coeff':1e-3, 'subdir':'arrowhead_2D', 'suffix': 'recent-', 'Nx': 128, 'Ny':256 }
    suffix_end = 'periodic'
    symmetry_mode = False


elif setting_mode == 6:

    Lz = 1/8 * np.pi

    solver_params['Nz'] = 16
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 4e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'ndim': 2, 'noise_coeff':1e-3, 'subdir':'arrowhead_2D', 'suffix': 'recent-', 'Nx': 128, 'Ny':256 }
    suffix_end = 'periodic'
    symmetry_mode = False
    
elif setting_mode == 7:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 7.5*np.pi

    solver_params['Nz'] = 120
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 8*np.pi

    ic_dict_if_reinit = {'Nz': 128, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 8:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 7*np.pi

    solver_params['Nz'] = 112
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 8*np.pi

    ic_dict_if_reinit = {'Nz': 128, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 9:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 6.5*np.pi

    solver_params['Nz'] = 104
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 8*np.pi

    ic_dict_if_reinit = {'Nz': 128, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 10:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 6*np.pi

    solver_params['Nz'] = 96
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 8*np.pi

    ic_dict_if_reinit = {'Nz': 128, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 11:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 5.5*np.pi

    solver_params['Nz'] = 88
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 8*np.pi

    ic_dict_if_reinit = {'Nz': 128, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 12:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 5*np.pi

    solver_params['Nz'] = 80
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 8*np.pi

    ic_dict_if_reinit = {'Nz': 128, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 13:
    # Attempt to localise in x direction via stretching

    material_params['W'] = 20
    solver_params['dt'] = 2e-3

    system_params['Lz'] = 4*np.pi
    system_params['Lx'] = 10*np.pi

    solver_params['Nz'] = 64
    solver_params['Ny'] = 64
    solver_params['Nx'] = 256

    ic_dict_if_reinit = {'Nx': 64, 'Lx': 3*np.pi, 'subdir':'arrowhead_3D', 'suffix': 'recent-localised'}
    suffix_end = f'localised-xy'
    translate = False


elif setting_mode == 14:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 4.5*np.pi

    solver_params['Nz'] = 72
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 5*np.pi

    ic_dict_if_reinit = {'Nz': 80, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 15:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 4*np.pi

    solver_params['Nz'] = 64
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 5*np.pi
    ic_dict_if_reinit = {'Nz': 80, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 16:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 3.5*np.pi

    solver_params['Nz'] = 56
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 5*np.pi

    ic_dict_if_reinit = {'Nz': 80, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

elif setting_mode == 17:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz
    kwargs = {'truncate': True}
    Lz = 3*np.pi

    solver_params['Nz'] = 48
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    Lz_ic = 5*np.pi

    ic_dict_if_reinit = {'Nz': 80, 'Lz': Lz_ic }
    suffix_end = 'localised'
    symmetry_mode = False
    translate = True

log_all_params(material_params, system_params, solver_params)

ic_file, noise_coeff, _ = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='arrowhead_3D', 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff, **kwargs)

if translate:
    timestepper.translate_AH_to_centre(mode='z')

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=symmetry_mode,
                     save_over_long=True, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir=f"arrowhead_{system_params['ndim']}D", suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=f"arrowhead_{system_params['ndim']}D")