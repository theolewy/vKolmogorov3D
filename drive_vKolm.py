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
                   'eps': 1e-3}

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

symmetry_mode = "yz"
kwargs = {}
translate = False
plot_subdirectory = "arrowhead_3D_Lz"
save_subdir = f"arrowhead_3D"
save_full_data = False
T=4000

if setting_mode == 0:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 32
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 0.4*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 1:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 2.9*np.pi}
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

    solver_params['Nz'] = 96
    system_params['Lz'] = 4.4*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 4*np.pi, 'Nz': 64}
    suffix_end = 'periodic-yz'


elif setting_mode == 5:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 14

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 32
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 1.48*np.pi}
    suffix_end = 'm=1'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 6:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 32
    system_params['Lz'] = input_val*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 0.75*np.pi, 'Nz': 32, 'noise_coeff': 1e-3}
    suffix_end = 'periodic-2-yz'
    plot_subdirectory = "arrowhead_3D_2_wavelengths"

elif setting_mode == 7:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nx'] = 32
    solver_params['Ny'] = 32
    solver_params['Nz'] = 48
    system_params['Lz'] =  4.5*np.pi
    solver_params['dt'] = 1e-2

    ic_dict_if_reinit = {'Nx': 64, 'Ny': 64, 'Nz': 96}
    suffix_end = 'periodic-yz'

elif setting_mode == 8:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nx'] = 64
    solver_params['Ny'] = 64
    solver_params['Nz'] = 96
    system_params['Lz'] =  4.5*np.pi

    ic_dict_if_reinit = {'Nx': 32, 'Ny': 32, 'Nz': 48, 'suffix': 'recent-periodic-yz'}
    suffix_end = 'periodic-yz-from-low-res'

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

    kwargs = {'truncate': True}
    material_params['W'] = 20

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = int(16*input_val)
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 16*np.pi, 'Nz': 256}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 18:
    # Get Periodic AH from 2D AH. m=1 mode branch
    material_params['W'] = 16

    system_params['Lz'] = input_val*np.pi
    solver_params['Nz'] = 64
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3.8*np.pi}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 19:
    # Get Periodic AH from 2D AH. m=1 mode branch

    material_params['W'] = input_val

    system_params['Lz'] = 3*np.pi
    solver_params['Nz'] = 48
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'W': 15}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

elif setting_mode == 20:
    # Get Periodic AH from 2D AH. m=1 mode branch

    material_params['W'] = input_val

    system_params['Lz'] = 4*np.pi
    solver_params['Nz'] = 64
    
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'W': 15.2}
    suffix_end = 'localised'
    plot_subdirectory = 'arrowhead_3D_W'

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

    system_params['ndim'] = 2
    material_params['W'] = 10
    system_params['Lx'] = 32*np.pi
    solver_params['Nx'] = 480
    solver_params['dt'] = 1e-3

    ic_dict_if_reinit = {'W': 20, 'Lx': 32*np.pi, 'Nx': 450, 'subdir': 'arrowhead_2D'}
    suffix_end = ''
    save_subdir = 'localisation'
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = False
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
    # Get Periodic AH from 2D AH. m=1 mode branch

    material_params['W'] = 10

    system_params['Lz'] = np.pi 

    solver_params['Nx'] = 64
    solver_params['Nz'] = 32

    ic_dict_if_reinit = {'Lz': np.pi / 4, 'Nz': 16}
    suffix_end = ''
    symmetry_mode = 'yz'


elif setting_mode == 27:
    # Get Periodic AH from 2D AH. m=1 mode branch
    
    solver_params['Nx'] = 640
    solver_params['Ny'] = 64
    solver_params['Nz'] = 32

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = np.pi
    solver_params['dt'] = 2e-3

    ic_dict_if_reinit = {'Lx': 32*np.pi,'Nx': 480, 'Ny': 64, 'subdir': 'localisation'}
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"

elif setting_mode == 28:
    solver_params['Nz'] = 96
    system_params['Lz'] = 8*np.pi
    solver_params['Nx'] = 64
    system_params['Lx'] = 4*np.pi

    ic_dict_if_reinit = {'Lz': 6*np.pi}
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"

elif setting_mode == 29:
    # Get Periodic AH from 2D AH. m=1 mode branch
    
    solver_params['Nx'] = 480
    solver_params['Ny'] = 64
    solver_params['Nz'] = 16

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = np.pi/2
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lx': 32*np.pi,'Nx': 450, 'Ny': 64, 'ndim':2, 'noise_coeff': 1e-3, 'subdir': 'arrowhead_2D', 'suffix': 'recent-'}
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"

elif setting_mode == 30:
    # Get Periodic AH from 2D AH. m=1 mode branch
    
    solver_params['Nx'] = 480
    solver_params['Ny'] = 64
    solver_params['Nz'] = 32

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz':np.pi/2, 'Nz':16 }
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"
    translate = True
elif setting_mode == 31:
    # Get Periodic AH from 2D AH. m=1 mode branch
    
    solver_params['Nx'] = 480
    solver_params['Ny'] = 48
    solver_params['Nz'] = 16

    system_params['Lx'] = 32*np.pi
    system_params['Lz'] = np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Nx': 480, 'Ny': 64, 'Nz': 32}
    suffix_end = ''
    plot_subdirectory = 'streamwise_localisation'
    symmetry_mode = 'yz'
    save_subdir = f"localisation"
    translate = False
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
                     track_TW=False, 
                     enforce_symmetry=symmetry_mode,
                     save_over_long=True, 
                     save_full_data=save_full_data, full_save_freq=2,
                     save_subdir=save_subdir, suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=plot_subdirectory)