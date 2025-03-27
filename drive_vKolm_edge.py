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
elif len(sys.argv) == 5:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
    input_val = float(eval(sys.argv[3]))
    input_val2 = float(eval(sys.argv[4]))
else:
    raise Exception('Need more inputs!')    

symmetry_mode = "yz"
kwargs = {}
plot_subdirectory = "arrowhead_3D_localised_edge"
save_subdir = f"arrowhead_3D_localised_edge"
save_full_data = False
T=4000

if setting_mode == 0:

    solver_params['Nz'] = input_val2
    system_params['Lz'] =  input_val*np.pi

    ic_dict_if_reinit = {'subdir': 'arrowhead_3D', 'suffix': 'recent-localised'}
    suffix_end = ''


log_all_params(material_params, system_params, solver_params)

ic_file, noise_coeff, reinit = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir=save_subdir, 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=noise_coeff, **kwargs)

if reinit:
    timestepper._convert_to_edge_guess()

timestepper.simulate(T=T, ifreq=100, 
                     track_TW=False, 
                     enforce_symmetry=symmetry_mode,
                     save_over_long=True, 
                     save_full_data=save_full_data, full_save_freq=2,
                     save_subdir=save_subdir, suffix_end=suffix_end, 
                     plot=True, plot_dev=True, plot_subdirectory=plot_subdirectory)