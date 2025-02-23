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
window_x = False
if setting_mode == 0:
    a, b = np.pi/2, np.pi/4

    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': np.pi, 'Nz': 32}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14'
elif setting_mode == 1:
    a, b = np.pi, np.pi/2
    
    system_params['Lz'] = 4*np.pi
    solver_params['Nz'] = 64

    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': np.pi, 'Nz': 32}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14'

# elif setting_mode == 3:
#     a, b = 3/2 * np.pi, np.pi/2

#     ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': np.pi, 'Nz': 32}
#     suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-3,14'

elif setting_mode == 4:
    a, b = np.pi/3, np.pi/6
    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': 1/2*np.pi, 'Nz': 16}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-1,57'
elif setting_mode == 5:
    a, b = np.pi/6, np.pi/12
    ic_dict_if_reinit = {'suffix': 'recent-periodic', 'subdir':'arrowhead_3D', 'Lz': 1/4*np.pi, 'Nz': 16}
    suffix_end = f'a-{a:.4g}-b-{b:.4g}-Lz-orig-0,785'
elif setting_mode == 10:
    # Localising in a spanwise localised soln in x direction...
    a, b = np.pi, np.pi

    material_params['W'] = 20
    solver_params['dt'] = 2e-3

    system_params['Lz'] = 4*np.pi
    system_params['Lx'] = 12*np.pi

    solver_params['Nz'] = 64
    solver_params['Ny'] = 64
    solver_params['Nx'] = 256

    ic_dict_if_reinit = {'Nx': 64, 'Lx': 3*np.pi, 'subdir':'arrowhead_3D', 'suffix': 'recent-localised'}
    suffix_end = f'localised-xy-a-{a:.4g}-b-{b:.4g}'
    window_x = True
elif setting_mode == 11:
    # Localising in a spanwise localised soln in x direction...
    a, b = 5*np.pi, np.pi

    material_params['W'] = 20
    solver_params['dt'] = 2e-3

    system_params['Lz'] = 4*np.pi
    system_params['Lx'] = 30*np.pi

    solver_params['Nz'] = 64
    solver_params['Ny'] = 64
    solver_params['Nx'] = 640

    ic_dict_if_reinit = {'Nx': 256, 'Lx': 10*np.pi}
    suffix_end = f'localised-xy-a-{a:.4g}-b-{b:.4g}'
    window_x = True

log_all_params(material_params, system_params, solver_params)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

ic_file, noise_coeff, reinit = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='windows', 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=0, tile=True)

if reinit:
    if window_x:
        timestepper.translate_AH_to_centre(mode='x')    # move so arrowhead is in the middle of the domain
        timestepper.window(a, b, mode='x')
    else:
        timestepper.translate_AH_to_centre(mode='z')    # move so arrowhead is in the middle of the domain
        timestepper.window(a, b)

timestepper.simulate(T=4000, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=False,
                     save_over_long=False, 
                     save_full_data=False, full_save_freq=5,
                     save_subdir=f"windows", suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=f"windowing_{system_params['ndim']}D")