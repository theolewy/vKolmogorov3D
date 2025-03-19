import sys 
import numpy as np
from tools.kolmogorov import BaseFlow, NumericSolver, TimeStepper3D
from tools.misc_tools import get_ic_file, log_all_params, on_local_device

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
else:
    raise Exception('Need more inputs!')

# a between pi/4 and pi: 2pi/8, 3pi/8, pi/2, 5pi/8, 6pi/8
# b between pi/4 and pi: pi/8, pi/4, pi/2, 3pi/4

"""
Parameters tried:
a = pi/2 with b=pi/8, pi/4, pi/2, 3pi/4
a = 3pi/8 with b=pi/8, pi/4, pi/2
"""
symmetry_mode = 'yz'
window_x = False
tile = True
if setting_mode == 0:
    a, b = np.pi/2, np.pi/4

    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': np.pi, 'Nz': 32}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14'

elif setting_mode == 1:
    a, b = np.pi, np.pi/2
    
    material_params['W'] = 10

    solver_params['Nx'] = 64

    system_params['Lz'] = 8*np.pi
    solver_params['Nz'] = 128

    ic_dict_if_reinit = {'W': 10, 'ndim': 2, 'Lx': 3*np.pi, 'Nx': 128, 'Ny': 256, 'subdir': 'arrowhead_2D', 'suffix': 'recent-'}
    save_subdir = 'windows'
    plot_subdirectory = 'spanwise_localisation'
    symmetry_mode = 'yz'
    suffix_end = f'window-2D-AH'
    window_mode = 'z'

elif setting_mode == 3:
    a, b = np.pi/2, np.pi/4

    material_params['W'] = 17

    ic_dict_if_reinit = {'suffix': f'recent-a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14-twice'}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14-yz-twice'
    symmetry_mode = 'yz'

elif setting_mode == 4:
    a, b = np.pi/3, np.pi/6
    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': 1/2*np.pi, 'Nz': 16}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-1,57'
elif setting_mode == 5:
    a, b = np.pi/6, np.pi/12
    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': 1/4*np.pi, 'Nz': 16}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-0,785'
elif setting_mode == 6:
    a, b = np.pi/2, np.pi/4
    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': np.pi, 'Nz': 32}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14-phase'

elif setting_mode == 10:
    # STREAMWISE LOCALISATION

    a, b = np.pi / 2, np.pi / 4

    solver_params['Nx'] = 480
    solver_params['Ny'] = 64
    solver_params['Nz'] = 96

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = 5*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Nx': 480, 'Ny':64, 'Nz': 32, 'Lz': np.pi, 'suffix': f'recent-', 'subdir': 'localisation'}
    suffix_end = f'windowed'
    window_mode = 'z'
    tile = False

elif setting_mode == 11:
    # STREAMWISE LOCALISATION

    a, b = np.pi / 2, np.pi / 4

    solver_params['Nx'] = 480
    solver_params['Ny'] = 64
    solver_params['Nz'] = 48

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = 5*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Nx': 480, 'Ny':64, 'Nz': 96}
    suffix_end = f'windowed'
    window_mode = False
    tile = False

log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff, reinit = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='localisation', 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=0, tile=tile)

if reinit:
    if window_mode == 'x':
        timestepper.translate_AH_to_centre(mode='x')    # move so arrowhead is in the middle of the domain
        timestepper.window(a, b, mode='x')
    elif window_mode == 'z':
        # timestepper.translate_AH_to_centre(mode='z')    # move so arrowhead is in the middle of the domain
        timestepper.window(a, b, mode='z')

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry='yz',
                     save_over_long=False, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir=f"localisation", suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=f"streamwise_localisation")