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
                 'dt': 2e-3,
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

symmetry_mode = 'yz'
kwargs = {}
translate = False

if setting_mode == 0:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] =  3.5*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3*np.pi, 'Nz': 64}
    suffix_end = 'periodic-yz'

elif setting_mode == 1:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] = 4*np.pi

    ic_dict_if_reinit = {'Lz': 3*np.pi, 'Nz': 64}
    suffix_end = 'periodic-yz'

elif setting_mode == 2:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 96
    system_params['Lz'] = 4.7*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 4.5*np.pi, 'suffix': 'recent-periodic-dt2-yz'}
    suffix_end = 'periodic-yz'

elif setting_mode == 3:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 96
    system_params['Lz'] = 4.6*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 4.5*np.pi, 'suffix': 'recent-periodic-dt2-yz'}
    suffix_end = 'periodic-yz'
elif setting_mode == 4:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 96
    system_params['Lz'] = 4.5*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'suffix': 'recent-periodic-yz'}
    suffix_end = 'periodic-dt2-yz'


elif setting_mode == 5:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] = 4*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'suffix': 'recent-periodic-yz'}
    suffix_end = 'periodic-dt2-yz'


elif setting_mode == 6:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 32
    system_params['Lz'] = 1.5 * np.pi
    solver_params['dt'] = 3e-2

    ic_dict_if_reinit = {'suffix': 'recent-periodic'}
    suffix_end = 'periodic-yz'

elif setting_mode == 7:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nx'] = 32
    solver_params['Ny'] = 32
    solver_params['Nz'] = 32
    solver_params['dt'] = 1e-2
    system_params['Lz'] =  4.5*np.pi

    ic_dict_if_reinit = {'Nx': 64, 'Ny': 64, 'Nz': 96}
    suffix_end = 'periodic-yz'

elif setting_mode == 8:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nx'] = 32
    solver_params['Ny'] = 32
    solver_params['Nz'] = 32
    solver_params['dt'] = 1e-2
    system_params['Lz'] =  4.5*np.pi

    ic_dict_if_reinit = {'Nx': 64, 'Ny': 64, 'Nz': 72, 'suffix': 'recent-localised'}
    suffix_end = 'localised-yz'
elif setting_mode == 9:
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nz'] = 64
    system_params['Lz'] = 2.5*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = None
    suffix_end = 'periodic-yz'
elif setting_mode == 10:
    # Stretch AH

    system_params['Lz'] = 12*np.pi
    solver_params['Nz'] = 192

    ic_dict_if_reinit = {'Lz':8*np.pi, 'Nz':128, 'suffix': 'recent-localised'}
    suffix_end = 'localised-stretch'

elif setting_mode == 11:
    # Truncate to Saddle
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nz'] = 64
    system_params['Lz'] =  3.75*np.pi

    ic_dict_if_reinit = {'Lz': 3.7*np.pi}
    suffix_end = 'localised-yz'
elif setting_mode == 12:
    # Truncate to Saddle
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nz'] = 64
    system_params['Lz'] =  3.7*np.pi

    ic_dict_if_reinit = {'Lz': 3.7*np.pi}
    suffix_end = 'localised-yz'

elif setting_mode == 13:
    # Truncate to Saddle
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nz'] = 64
    system_params['Lz'] =  3.68*np.pi
    solver_params['dt'] = 5e-3

    kwargs = {'truncate': True}
    ic_dict_if_reinit = {'Lz': 3.7*np.pi}
    suffix_end = 'localised-yz'
elif setting_mode == 14:
    # Truncate to Saddle
    # Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

    solver_params['Nz'] = 64
    system_params['Lz'] =  3.69*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3.68*np.pi}
    suffix_end = 'localised-yz'
elif setting_mode == 15:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] =  3*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3*np.pi, 'Nz': 64}
    suffix_end = 'periodic-yz'
elif setting_mode == 16:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 96
    system_params['Lz'] =  input_val*np.pi

    ic_dict_if_reinit = {'Lz': 5*np.pi, 'Nz': 96}
    suffix_end = 'periodic-yz'
    solver_params['dt'] = 5e-3

elif setting_mode == 17:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 16
    system_params['Lz'] =  np.pi/5

    ic_dict_if_reinit = {'Lz': np.pi/4, 'Nz': 16}
    suffix_end = 'periodic-yz'

elif setting_mode == 20:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] =  4*np.pi

    solver_params['dt'] = 4e-3

    ic_dict_if_reinit = {'Lz': 3.5*np.pi, 'Nz': 56, 'suffix': 'recent-localised-yz'}
    suffix_end = 'jockey-yz'

elif setting_mode == 23:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 32
    system_params['Lz'] =  input_val*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 1.5*np.pi, 'Nz': 32}
    suffix_end = 'periodic-yz'
elif setting_mode == 24:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 64
    system_params['Lz'] =  2*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 3*np.pi, 'Nz': 64}
    suffix_end = 'periodic-yz'

elif setting_mode == 25:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 16
    system_params['Lz'] =  input_val*np.pi

    ic_dict_if_reinit = {'Lz': np.pi/2, 'Nz': 16}
    suffix_end = 'periodic-yz'

elif setting_mode == 26:
    # Get Periodic AH from 2D AH. m=1 mode branch

    solver_params['Nz'] = 32
    system_params['Lz'] =  input_val*np.pi
    solver_params['dt'] = 5e-3

    ic_dict_if_reinit = {'Lz': 0.9*np.pi, 'Nz': 32}
    suffix_end = 'below-periodic-yz'


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
                     plot=True, plot_dev=True, plot_subdirectory=f"arrowhead_3D_Lz")