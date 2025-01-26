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
    setting_mode = int(sys.argv[2])
else:
    raise Exception('Need more inputs!')

if setting_mode == 0:
    solver_params['Nz'] = 16
    system_params['Lz'] = 0.5 * np.pi
elif setting_mode == 1:
    solver_params['Nz'] = 32
    system_params['Lz'] = np.pi
elif setting_mode == 2:
    solver_params['Nz'] = 48
    system_params['Lz'] = 1.5 * np.pi
elif setting_mode == 3:
    solver_params['Nz'] = 64
    system_params['Lz'] = 2 * np.pi
elif setting_mode == 4:
    system_params['eps'] = 2e-4
    solver_params['Nx'] = 256
    solver_params['Ny'] = 128
    solver_params['Nz'] = 32
    solver_params['dt'] = 1e-3
    system_params['Lz'] = 0.5 * np.pi


log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-symmetry-xy', subdir='arrowhead_3D', 
                                   ic_dict_if_reinit={'ndim': 3, 'Nx': 128, 'Ny': 64})
timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff)

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=True,
                     save_over_long=True, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir='arrowhead_3D', suffix_end='symmetry-yz', 
                     plot=True, plot_dev=True, plot_subdirectory='arrowhead_3D')