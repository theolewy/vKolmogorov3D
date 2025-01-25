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
                 'Lx': 4 * np.pi,
                 'Lz': np.pi,
                 'n': 1}

solver_params = {'Nx': 128,
                 'Ny': 64,
                 'Nz': 32,
                 'dt': 2e-3,
                 'c': 0}

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    multiplier = float(sys.argv[2])
    solver_params['Nz'] = int(multiplier * 32)
    system_params['Lz'] = multiplier * np.pi
else:
    raise Exception('Need more inputs!')

log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-symmetry-xy', subdir='arrowhead_3D', 
                                   ic_dict_if_reinit={'ndim': 3, 'Nx': 128, 'Ny': 64, 
                                                      'suffix': 'recent-', 'subdir':'arrowhead_3D'})
timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff)

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=True,
                     save_over_long=True, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir='arrowhead_3D', suffix_end='symmetry-xy', 
                     plot=True, plot_dev=True, plot_subdirectory='arrowhead_3D')