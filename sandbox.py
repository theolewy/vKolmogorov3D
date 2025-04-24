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
                 'Nz': 64,
                 'dt': 5e-3}

symmetry_mode = False
kwargs = {}


# Reduce Lz from 8pi down. Nz MUST be over 16 per pi in Lz

ic_dict_if_reinit = {'Nx':16, 'Ny': 16, 'Nz':16}
suffix_end = 'localised'
symmetry_mode = 'yz'

log_all_params(material_params, system_params, solver_params)

ic_file, noise_coeff, _ = get_ic_file(material_params, system_params, solver_params, suffix=f'recent-{suffix_end}', subdir='arrowhead_3D', 
                                   ic_dict_if_reinit=ic_dict_if_reinit)

timestepper = TimeStepper3D(material_params=material_params, system_params=system_params, solver_params=solver_params)

timestepper.numeric_solver.base_solver.plot_base_state(fname=f'a={material_params["a"]}')
# timestepper.ic(ic_file=ic_file, flow=None, noise_coeff=0e-1, **kwargs)

# timestepper.simulate(T=4000, ifreq=10, 
#                      track_TW=False, 
#                      enforce_symmetry=symmetry_mode,
#                      save_over_long=True, 
#                      OVERRIDE_LOCAL_SAVE=False,
#                      save_full_data=False, full_save_freq=5,
#                      save_subdir=f"arrowhead_{system_params['ndim']}D", suffix_end=suffix_end, 
#                      plot=True, plot_dev=True, plot_subdirectory=f"arrowhead_{system_params['ndim']}D")