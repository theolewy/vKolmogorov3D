import sys 
import numpy as np
from tools.solvers.kolmogorov import BaseFlow, NumericSolver, TimeStepper3D
from tools.misc_tools import get_ic_file, log_all_params, on_local_device

material_params = {'W': 20,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3}

system_params = {'ndim': 3,
                 'Lx': 3 * np.pi,
                 'Lz': 4* np.pi,
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
    # Localised AH
    Lz = 8 * np.pi

    solver_params['Nz'] = int(16 * Lz  / np.pi)
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 18

    ic_dict_if_reinit = {'W': 20}
    suffix_end = 'localised'
elif setting_mode == 1:
    # Localised AH
    Lz = 8 * np.pi

    solver_params['Nz'] = int(16 * Lz  / np.pi)
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 16

    ic_dict_if_reinit = {'W': 20}
    suffix_end = 'localised'
elif setting_mode == 2:
    # Localised AH
    Lz = 8 * np.pi

    solver_params['Nz'] = int(16 * Lz  / np.pi)
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 14

    ic_dict_if_reinit = {'W': 20}
    suffix_end = 'localised'
elif setting_mode == 3:
    # Get Periodic AH from 2D AH. PRODUCTION METHOD OF OBTAINING PERIODIC AH FROM 2D AH
    Lz = 2 * np.pi

    solver_params['Nz'] = int(16 * Lz  / np.pi)
    solver_params['Ny'] = 64
    solver_params['Nx'] = 64

    solver_params['dt'] = 2e-3

    system_params['Lz'] = Lz

    material_params['W'] = 20

    ic_dict_if_reinit = {'ndim': 2, 'noise_coeff':1e-3, 'suffix': 'recent-', 'Nx': 128, 'Ny':256 }
    suffix_end = 'periodic'

elif setting_mode == 5:
    # Get periodic arrowheads, ready for localisation (even about low pressure region)
    material_params['W'] = 20 
    solver_params['Nz'] = 64
    system_params['Lz'] = 4 * np.pi
    system_params['Lx'] = 3 * np.pi
    solver_params['Nx'] = 64
    solver_params['dt'] = 2e-3
    ic_dict_if_reinit = {'W': 30}
    symmetry_mode = 'yz'
    suffix_end = f'symmetry-yz'
elif setting_mode == 6:
    # Play with localised AH
    material_params['W'] = 20
    solver_params['Nz'] = 128
    system_params['Lz'] = 8 * np.pi
    system_params['Lx'] = 3 * np.pi
    solver_params['Nx'] = 64
    solver_params['dt'] = 2e-3
    symmetry_mode = False

    ic_dict_if_reinit = {'Lz': 6 * np.pi}
    suffix_end = 'localised'
elif setting_mode == 7:
    # Play with localised AH
    material_params['W'] = 20
    solver_params['Nz'] = 64
    system_params['Lz'] = 4 * np.pi
    system_params['Lx'] = 3 * np.pi
    solver_params['Nx'] = 64
    solver_params['dt'] = 2e-3
    symmetry_mode = False

    ic_dict_if_reinit = None
    suffix_end = 'localised'
elif setting_mode == 8:
    # Play with localised AH
    material_params['W'] = 20
    solver_params['Nz'] = 128
    solver_params['Ny'] = 128
    solver_params['Nx'] = 128

    system_params['Lz'] = 4 * np.pi
    system_params['Lx'] = 3 * np.pi
    material_params['eps'] = 2e-4
    solver_params['dt'] = 1e-3
    symmetry_mode = False

    ic_dict_if_reinit = {'suffix':'recent-localised', 'Nx':64, 'Ny':64, 'Nz':64, 'eps': 1e-3}
    suffix_end = 'localised'
elif setting_mode == 9:
    # Play with localised AH
    material_params['W'] = 20
    solver_params['Nz'] = 128
    system_params['Lz'] = 6 * np.pi
    system_params['Lx'] = 3 * np.pi
    solver_params['Nx'] = 64
    solver_params['dt'] = 2e-3
    symmetry_mode = False

    ic_dict_if_reinit = {'Nz': 64, 'Lz': 4 * np.pi}
    suffix_end = 'localised'


log_all_params(material_params, system_params, solver_params)

ic_file, noise_coeff, _ = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='arrowhead_3D', 
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