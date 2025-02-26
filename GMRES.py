import numpy as np
import pickle
import sys
import os
from dedalus import public as de
from dedalus.extras import flow_tools
from mpi4py import MPI
import logging

logger = logging.getLogger(__name__)

from tools.misc_tools import get_roots, on_local_device, log_all_params
from tools.misc_tools import get_fpath_sim, get_fpath_s_file

from tools.newton_tools import *

material_params = {'W': 18,
                'beta': 0.9,
                'Re': 0.5,
                'L': np.infty,
                'eps': 1e-3}

system_params = {'ndim': 3,
                'Lx': 3*np.pi,
                'Lz': 8*np.pi,
                'n': 1}

solver_params = {'Nx': 64,
                'Ny': 64,
                'Nz': 128,
                'dt': 2e-3}

gmres_params = {
    'dns_nproc': 16,         # parallelise over this many cores
    'Delta_start': 128000., # hookstep parameter  
    'eps_gm': 2e-3,     # controls GMRES residual threshold
    'eps_newt': 1e-6,   # controls size of newton step  
    'TW_flag': True,    # seach for a travelling wave
    'sim_time_TW': 3,   # number of steps to take in each simulation of a travelling wave    
                }

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
elif len(sys.argv) == 2 and on_local_device():
    setting_mode = int(sys.argv[1])
else:
    raise Exception("Needs 1 inputs")

if setting_mode == 0:
    label = 'W=17'
    material_params['W'] = 17
elif setting_mode == 1:
    label = 'W=18'
    material_params['W'] = 18

ic_file_in = get_fpath_s_file(material_params, system_params, solver_params, suffix='recent-localised', subdir='arrowhead_3D')

fpath_out = get_fpath_sim(material_params, system_params, solver_params, suffix='recent-localised', subdir='arrowhead_3D', dir='newtonWrapper')

if on_local_device(): gmres_params['dns_nproc'] = 1

log_all_params(material_params=material_params, solver_params=solver_params, system_params=system_params)

converge_TW(material_params, system_params, solver_params, ic_file_in, fpath_out, gmres_params, label=label)