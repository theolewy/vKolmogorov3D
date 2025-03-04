# a series of functions that interact with the DNS code
import copy
import os
import signal
import shlex, subprocess
import shutil
import sys

import numpy as np
import mpi4py as MPI


import h5py
import time

import logging

from dedalus.tools import post

from cfd_tools.newtonWrapper.interact_dedalus import *
import cfd_tools.newtonWrapper.dedalus_io as DIO
import cfd_tools.newtonWrapper.equilibria as EQ

from tools.kolmogorov import TimeStepper3D as TimeStepper
from tools.misc_tools import get_roots

from tools.misc_tools import get_fpath_sim


logger = logging.getLogger(__name__)

def write_driveFile(material_params, system_params, solver_params, a1, a2, Tmin, lam='', data_root=''):

    Re, W, beta, L, eps, = material_params['Re'], material_params['W'], material_params['beta'], \
                           material_params['L'], material_params['eps']

    Lx, Lz = system_params['Lx'], system_params['Lz']

    Nx, Ny, Nz, dt = solver_params['Nx'], solver_params['Ny'], solver_params['Nz'], solver_params['dt']

    core_root, _ = get_roots()

    if (L == np.inf): L = 'np.inf'

    f = open(f'{data_root}/drive_flow.py', 'w')

    f.write('import numpy as np\n')
    f.write("import sys\n")
    f.write("import os\n")
    f.write(f"sys.path.append('{core_root}')\n")
    f.write('from mpi4py import MPI\n')
    f.write('import time\n')
    f.write('from dedalus import public as de\n')
    f.write('from dedalus.extras import flow_tools\n')
    f.write('import logging\n')
    f.write('logger = logging.getLogger(__name__)\n')
    f.write('from tools.kolmogorov import TimeStepper3D as TimeStepper\n')

    f.write('comm = MPI.COMM_WORLD\n')
    f.write('rank = comm.Get_rank()\n')

    f.write(f"material_params = {{'W': {W},\n")
    f.write(f"'beta': {beta},\n")
    f.write(f"'Re': {Re},\n")
    f.write(f"'L': {L},\n")
    f.write(f"'eps': {eps} }}\n")

    f.write(f"system_params = {{")
    f.write(f"'ndim': {3},\n")
    f.write(f"'n': {1},\n")
    f.write(f"'Lz': {Lz},\n")
    f.write(f"'Lx': {Lx} }}\n")

    f.write(f"solver_params = {{'Nx': {Nx},\n")
    f.write(f"'Ny': {Ny},\n")
    f.write(f"'Nz': {Nz},\n")
    f.write(f"'dt': {dt} }}\n")

    f.write("timestepper = TimeStepper(material_params, system_params, solver_params)\n")
    f.write(f"timestepper.ic(ic_file='{data_root}/ic_half.h5')\n")
    f.write(f"timestepper.update_dy()\n")   # required as ic_file only contains u,v,p etc, not uy, vy...
    f.write('timestepper.solver.sim_time = 0.0\n')
    f.write('timestepper.solver.iteration = 0\n')
    f.write('timestepper.solver.start_time = timestepper.solver.sim_time\n')

    f.write('timestepper.solver.stop_sim_time  = np.inf\n')
    f.write('timestepper.solver.stop_wall_time = np.inf\n')
    f.write('timestepper.solver.stop_iteration = np.inf\n')

    f.write('timestepper.trace_metric_list = [] \n')
    f.write('timestepper.KE_metric_list = [] \n')
    f.write('timestepper.u_metric_list = [] \n')
    f.write('timestepper.time_list = [] \n')

    f.write('ifreq = 100\n')

    f.write(f"snapshots = timestepper.solver.evaluator.add_file_handler('{data_root}/snapshots',\n") 
    f.write("           sim_dt = 1.0, max_writes=50, mode='overwrite')\n")
    f.write('snapshots.add_system(timestepper.solver.state)\n')
    f.write(f"scalars = timestepper.solver.evaluator.add_file_handler('{data_root}/scalars',\n") 
    f.write("           sim_dt = 0.05, max_writes=np.inf, mode='overwrite')\n")
    f.write("scalars.add_task('integ(c11+c22+c33)/area',name='vol_tr')\n")
    f.write("logger.info('Starting loop')\n")
    f.write('while timestepper.solver.ok:\n')
    f.write("    timestepper.solver.step(solver_params['dt'])\n")

    f.write("    vol_tr = timestepper.flow.volume_average('trace')\n")
    f.write("    vol_Tr = timestepper.flow.volume_average('trace_base')\n")
    f.write("    trace_metric = (vol_tr - vol_Tr) / vol_Tr \n")
    f.write("    vol_KE = timestepper.flow.volume_average('KE')\n")
    f.write("    vol_KE_l = timestepper.flow.volume_average('KE_base')\n")
    f.write("    KE_metric = (vol_KE - vol_KE_l) / vol_KE_l \n")
    f.write("    vol_u = timestepper.flow.volume_average('|u|')\n")
    f.write("    vol_U = timestepper.flow.volume_average('|U|')\n")
    f.write("    u_metric = (vol_u - vol_U) / vol_U \n")

    f.write('    if (trace_metric > %e and timestepper.solver.sim_time > %e):\n'%(a2,Tmin))
    f.write('       if rank == 0:\n')
    f.write(f"           file = open('{data_root}/is2','w')\n")
    f.write("           file.write('%e, True'%(timestepper.solver.sim_time))\n")
    f.write("       logger.info('Stop trajectory: goes to field 2')\n")
    f.write('       break\n')
    f.write('    if (trace_metric < %e and timestepper.solver.sim_time > %e):\n'%(a1,Tmin))
    f.write('       if rank == 0:\n')
    f.write(f"           file = open('{data_root}/is2','w')\n")
    f.write("           file.write('%e, False'%(timestepper.solver.sim_time))\n")
    f.write("       logger.info('Stop trajectory: goes to field 1')\n")
    f.write('       break\n')

    f.write('    if timestepper.solver.iteration % ifreq == 0:\n')
    f.write("        KE = timestepper.flow.volume_average('KE')\n")
    f.write("        trace = timestepper.flow.volume_average('trace')\n")
    f.write("        logger.info('It: %i, t: %.3e, dt: %.2e trace: %.4e, min(det(C)): %.4e, KE: %.4e'\n")
    f.write("                 % (timestepper.solver.iteration, timestepper.solver.sim_time, timestepper.dt,\n")
    f.write("                 trace, timestepper.flow.min('det(C)'),\n")
    f.write("                 KE))\n")

    f.write("        timestepper.trace_metric_list.append(trace_metric)   \n")
    f.write("        timestepper.KE_metric_list.append(KE_metric)   \n")
    f.write("        timestepper.u_metric_list.append(u_metric)   \n")
    f.write("        timestepper.time_list.append(timestepper.solver.sim_time)   \n")
    f.write(f"        timestepper._plot_arrays_and_metrics(subdirectory='edge_track', suffix_end='lambda={lam:.10g}', plot_dev=True)   \n")

    f.write('end_time = time.time()\n')
    f.close()
