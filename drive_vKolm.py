import sys 
import numpy as np
from tools.kolmogorov import BaseFlow, NumericSolver, TimeStepper3D
from tools.misc_tools import get_ic_file, log_all_params, on_local_device
import logging

logger = logging.getLogger(__name__)

material_params = {'W': 20,
                   'beta': 0.9,
                   'Re': 0.5,
                   'L': np.infty,
                   'eps': 1e-3,
                   'a': 1}

system_params = {'ndim': 3,
                 'Lx': 3 * np.pi,
                 'Lz': 4 * np.pi,
                 'n': 1}

solver_params = {'Nx': 64,
                 'Ny': 64,
                 'Nz': 32,
                 'dt': 5e-3,
                 'c': 0}

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
elif len(sys.argv) == 4:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
    input_val = float(eval(sys.argv[3]))
else:
    raise Exception('Need more inputs!')    

track_TW = False
symmetry_mode = "yz"
kwargs = {}
translate = False
plot_subdirectory = "arrowhead_3D_Lz"
save_subdir = f"arrowhead_3D"
save_full_data = False
T=40000
change_coords = False
if setting_mode == 0:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 32
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 0.9*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 1:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 2.3*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 2:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 64
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3*np.pi, 'Nz': 48}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 3:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 2.6*np.pi, 'Nz': 48}
    suffix_end = 'after-bif'
    plot_subdirectory = 'arrowhead_3D_W'
elif setting_mode == 4:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 15

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 2.4*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 5:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 32
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 0.16*np.pi}
    suffix_end = 'm=1'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 6:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 15

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 64
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3.8*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 7:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nz'] = 32
    system_params['Lz'] = input_val*np.pi

    ic_dict_if_reinit = {'Lz': np.pi}
    suffix_end = 'periodic-yz'

elif setting_mode == 8:

    solver_params['Nz'] = 16
    system_params['Lz'] = input_val*np.pi

    ic_dict_if_reinit = {'Lz': np.pi}
    suffix_end = 'periodic-yz'

elif setting_mode == 9:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nx'] = 64
    solver_params['Ny'] = 64
    solver_params['Nz'] = 96
    system_params['Lz'] =  4.4*np.pi

    ic_dict_if_reinit = {'Nx': 32, 'Ny': 32, 'Nz': 48, 'suffix': 'recent-periodic-yz'}
    suffix_end = 'periodic-yz-from-low-res'
elif setting_mode == 10:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 20

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 64
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3.7*np.pi}
    suffix_end = 'localised-yz'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 11:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 18

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 5*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 12:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 18

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 64
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 4*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 13:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 18

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 80
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 5*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'
elif setting_mode == 14:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 16

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 2.9*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'
elif setting_mode == 15:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 16

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 80
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 4.8*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'
elif setting_mode == 16:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 96
    system_params['Lz'] =  input_val*np.pi

    ic_dict_if_reinit = {'Lz': 4.35*np.pi, 'Nz': 96}
    suffix_end = 'periodic-yz'
    solver_params['dt'] = 5e-3

elif setting_mode == 17:
    # Get Periodic AH from 2D AH. m=1 mode branch

    material_params['W'] = 20

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = int(16*input_val)
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 16*np.pi, 'Nz': 256}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 18:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 13

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 32
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'W':14}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W=13'


elif setting_mode == 19:
    material_params['W'] = input_val * 6 + 14

    system_params['Lz'] = input_val * 1.2 * np.pi + 2.8*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'W': 14, 'Lz': 3*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_connect'

elif setting_mode == 20:
    # Get Periodic AH from 2D AH. m=1 mode branch

    material_params['W'] = input_val * 6 + 14

    system_params['Lz'] = input_val * 1.2 * np.pi + 2.8*np.pi
    solver_params['Nz'] = 64
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'W': 20, 'Lz': 4*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_connect'

elif setting_mode == 21:
    # Get Periodic AH from 2D AH. m=1 mode branch

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = int(16*input_val)
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 8*np.pi, 'Nz': 128}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'
    kwargs = {'extend': True}

elif setting_mode == 22:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] = input_val * np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 4*np.pi}
    suffix_end = 'localised-yz'

elif setting_mode == 23:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 32
    system_params['Lz'] = np.pi
    solver_params['dt'] = 1e-3

    # ic_dict_if_reinit = {'noise_coeff': 1e-2, 'suffix': 'recent-', 'ndim': 2, 'W': 20, 'Nx': 128, 'Ny': 256, 'subdir': 'arrowhead_2D'}
    ic_dict_if_reinit = None
    suffix_end = 'periodic-from-2D'
    plot_subdirectory = 'arrowhead_3D_Lz'
    symmetry_mode = 'yz'
elif setting_mode == 24:
    # Get Periodic AH from 2D AH. m=1 mode branch

    system_params['ndim'] = 2
    material_params['W'] = 10
    system_params['Lx'] = 16*np.pi
    solver_params['Nx'] = 256
    solver_params['dt'] = 1e-3

    ic_dict_if_reinit = {'W': 20, 'Lx': 16*np.pi, 'Nx': 256, 'subdir': 'arrowhead_2D'}
    suffix_end = ''
    save_subdir = 'localisation'
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = False

elif setting_mode == 25:
    # Get Periodic AH from 2D AH. m=1 mode branch

    material_params['W'] = 10

    system_params['Lz'] = np.pi / 2

    solver_params['Nx'] = 64
    solver_params['Nz'] = 32

    ic_dict_if_reinit = {'Lz': np.pi / 4, 'Nz': 16}
    suffix_end = ''
    symmetry_mode = 'yz'

elif setting_mode == 26:
    
    solver_params['Nz'] = 64
    solver_params['Nx'] = 128
    system_params['Lz'] = 4*np.pi
    system_params['Lx'] = 6*np.pi
    
    a = input_val

    material_params['a'] = a

    ic_dict_if_reinit = {'Lx': 3*np.pi, 'Nx': 64}
    
    suffix_end = f'JS-a={a}'
    plot_subdirectory = 'arrowhead_3D_JS'
    save_subdir = f"arrowhead_3D"

elif setting_mode == 27:
    
    solver_params['Nz'] = 64
    solver_params['Nx'] = 96
    system_params['Lz'] = 4*np.pi
    system_params['Lx'] = 4*np.pi
    
    a = input_val

    material_params['a'] = a

    ic_dict_if_reinit = {'Lx': 3*np.pi, 'Nx': 64}
    
    suffix_end = f'JS-a={a}'
    plot_subdirectory = 'arrowhead_3D_JS'
    save_subdir = f"arrowhead_3D"


elif setting_mode == 28:
    
    solver_params['Nz'] = 96
    system_params['Lz'] = 6*np.pi
    
    a = input_val

    material_params['a'] = a
    material_params['W'] = 15

    ic_dict_if_reinit = {'suffix': 'recent-localised'}
    
    suffix_end = f'JS-a={a}'
    plot_subdirectory = 'arrowhead_3D_JS'
    save_subdir = f"arrowhead_3D"

elif setting_mode == 29:
    
    solver_params['Nz'] = 64
    system_params['Lz'] = 4*np.pi
    
    a = input_val

    material_params['a'] = a

    ic_dict_if_reinit = {'suffix': 'recent-localised'}
    
    suffix_end = f'JS-a={a}'
    plot_subdirectory = 'arrowhead_3D_JS'
    save_subdir = f"arrowhead_3D"

elif setting_mode == 30:

    solver_params['Nz'] = 64
    system_params['Lz'] = 4*np.pi

    pert = input_val
    ic_dict_if_reinit = {'suffix': 'recent-localised'}
    
    suffix_end = f'test-drift-pert-{pert}'
    plot_subdirectory = 'arrowhead_3D_drift'
    symmetry_mode = False
    track_TW = True
    save_subdir = f"arrowhead_3D"
    # kwargs = {'asymmetric_perturb': pert}

elif setting_mode == 31:

    solver_params['Nz'] = 64
    system_params['Lz'] = 4*np.pi

    pert = input_val
    ic_dict_if_reinit = {'suffix': f'recent-test-drift-pert-{pert}'}
    
    suffix_end = f'test-drift-pert-{pert}-twice'
    plot_subdirectory = 'arrowhead_3D_drift'
    symmetry_mode = False
    track_TW = True
    save_subdir = f"arrowhead_3D"
    kwargs = {'asymmetric_perturb': pert}

elif setting_mode == 32:
    # Get Periodic AH from 2D AH. m=1 mode branch
    
    solver_params['Nx'] = 480
    solver_params['Ny'] = 48
    solver_params['Nz'] = 32

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = 2*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Nx': 480, 'Ny': 48, 'Nz': 16, 'Lz': np.pi}
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"
    translate = False
elif setting_mode == 33:
    # Get Periodic AH from 2D AH. m=1 mode branch
    
    solver_params['Nx'] = 480
    solver_params['Ny'] = 48
    solver_params['Nz'] = 32

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = 1.5*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Nx': 480, 'Ny': 48, 'Nz': 16, 'Lz': np.pi}
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"
    translate = False

log_all_params(material_params, system_params, solver_params)

ic_file, noise_coeff, reinit = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir=save_subdir, 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)
timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff, **kwargs)

if translate and reinit:
    timestepper.translate_AH_to_centre(mode='z', shift=24)

timestepper.simulate(T=T, ifreq=100, 
                     track_TW=track_TW, 
                     enforce_symmetry=symmetry_mode,
                     save_over_long=True, 
                     save_full_data=save_full_data, full_save_freq=2,
                     save_subdir=save_subdir, suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=plot_subdirectory)