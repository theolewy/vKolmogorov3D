import sys 
import numpy as np
from tools.solvers.kolmogorov import BaseFlow, NumericSolver, TimeStepper3D
from tools.misc_tools import get_ic_file, log_all_params, on_local_device

material_params = {'W': 20,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3}

system_params = {'ndim': 2,
                 'Lx': 3 * np.pi,
                 'n': 1}

solver_params = {'Nx': 128,
                 'Ny': 256,
                 'dt': 5e-3,
                 'c': 0}

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
else:
    raise Exception('Need more inputs!')

if setting_mode == 0:
    # Get 2D bifurcation curve
    W_list = [30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6]
    W = W_list[job_idx]

    material_params['W'] = W

    ic_dict_if_reinit = {'W': 16}
    suffix_end = ''

log_all_params(material_params, system_params, solver_params)

ic_file, noise_coeff, _ = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='arrowhead_2D', 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff)

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=False,
                     save_over_long=True, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir=f"arrowhead_{system_params['ndim']}D", suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=f"arrowhead_{system_params['ndim']}D")