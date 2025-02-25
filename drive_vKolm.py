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
                 'Lz': 8 * np.pi,
                 'n': 1}

solver_params = {'Nx': 64,
                 'Ny': 64,
                 'Nz': 128,
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
translate = False

if setting_mode == 0:
    # Localised AH as W decreases

    material_params['W'] = input_val

    ic_dict_if_reinit = {'W': 18}
    suffix_end = 'localised'

elif setting_mode == 1:
    # Localised AH in 4pi at saddle v1

    material_params['W'] = 16

    system_params['Lz'] = 4*np.pi
    solver_params['Nz'] = 64

    ic_dict_if_reinit = {'W': 20}
    suffix_end = 'localised'
elif setting_mode == 2:
    # Localised AH in 4pi at saddle v2

    material_params['W'] = 17

    system_params['Lz'] = 4*np.pi
    solver_params['Nz'] = 64

    ic_dict_if_reinit = {'W': 20}
    suffix_end = 'localised'
elif setting_mode == 3:
    # Periodic AH in 4pi at saddle

    material_params['W'] = 17

    system_params['Lz'] = np.pi
    solver_params['Nz'] = 32

    ic_dict_if_reinit = {'W': 20}
    suffix_end = 'periodic'
elif setting_mode == 4:
    # Periodic AH in 4pi at saddle

    material_params['W'] = 17

    system_params['Lz'] = 1.5*np.pi
    solver_params['Nz'] = 48

    ic_dict_if_reinit = {'W': 20, 'Lz':np.pi, 'Nz':32}
    suffix_end = 'periodic'

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
                     plot=True, plot_dev=True, plot_subdirectory=f"arrowhead_3D_W")