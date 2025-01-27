import sys 
import numpy as np
from tools.solvers.kolmogorov import BaseFlow, NumericSolver, TimeStepper3D
from tools.misc_tools import get_ic_file, log_all_params, on_local_device

material_params = {'W': 30,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3}

system_params = {'ndim': 3,
                 'Lx': 2 * np.pi,
                 'Lz': 4 * np.pi,
                 'n': 1}

solver_params = {'Nx': 64,
                 'Ny': 64,
                 'Nz': 64,
                 'dt': 2e-3,
                 'c': 0}

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
else:
    raise Exception('Need more inputs!')

if setting_mode == 0:
    a, b = 0.5,0.5
if setting_mode == 0:
    a, b = 0.5,0.5

ic_dict_if_reinit = {'suffix': 'recent-', 'subdir':'arrowhead_3D'}
symmetry_mode = 'yz'
suffix_end = f'symm-yz-a-{a}-b-{b}'

log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff, reinit = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='windows', 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff)

if reinit:
    timestepper.window(a, b)

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=symmetry_mode,
                     save_over_long=True, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir=f"windows", suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=f"windowing_{system_params['ndim']}D")