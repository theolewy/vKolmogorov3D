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

material_params = {'W': 20,
                'beta': 0.9,
                'Re': 0.5,
                'L': np.infty,
                'eps': 1e-3}

system_params = {'ndim': 3,
                'Lx': 3*np.pi,
                'Lz': 8*np.pi,
                'n': 1}

solver_params = {'Nx': 32,
                'Ny': 32,
                'Nz': 32,
                'dt': 1e-2}

gmres_params = {
    'dns_nproc': 16,         # parallelise over this many cores
    'Delta_start': 128000., # hookstep parameter  
    'eps_gm': 1e-3,     # controls GMRES residual threshold
    'eps_newt': 1e-6,   # controls size of newton step  
    'TW_flag': False,    # seach for a travelling wave
    'sim_time_TW': 5,   # number of steps to take in each simulation of a travelling wave    
                }

if len(sys.argv) == 3:
    job_idx = int(sys.argv[1])
    setting_mode = int(sys.argv[2])
elif len(sys.argv) == 2 and on_local_device():
    setting_mode = int(sys.argv[1])
else:
    raise Exception("Needs 1 inputs")

if setting_mode == 0:
    label = 'Lz=4pi'
    system_params['Lz'] = 4*np.pi

ic_file_in = get_fpath_s_file(material_params, system_params, solver_params, suffix='recent-localised-yz', subdir='arrowhead_3D')
# ic_file_in = "/Users/theolewy/Documents/projects/vKolmogorov3D/storage/simulations/arrowhead_3D/sim_W_20_Re_0,5_beta_0,9_eps_0,001_L_inf_Lx_9,4248_Lz_12,566_ndim_3_N_64-64-64_recent-localised/sim_W_20_Re_0,5_beta_0,9_eps_0,001_L_inf_Lx_9,4248_Lz_12,566_ndim_3_N_64-64-64_recent-localised_s1.h5"
fpath_out = get_fpath_sim(material_params, system_params, solver_params, suffix='recent-localised-yz', subdir='arrowhead_3D', dir='newtonWrapper')

if on_local_device(): gmres_params['dns_nproc'] = 1

log_all_params(material_params=material_params, solver_params=solver_params, system_params=system_params)

converge_TW(material_params, system_params, solver_params, ic_file_in, fpath_out, gmres_params, T_guess=None, label=label)