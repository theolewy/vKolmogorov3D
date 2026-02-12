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

def write_driveFile(dt, Nstep, material_params, system_params, solver_params, label):

    Re, W, beta, L, eps, = material_params['Re'], material_params['W'], material_params['beta'], \
                           material_params['L'], material_params['eps']

    Lx, Lz = system_params['Lx'], system_params['Lz']

    Nx, Ny, Nz = solver_params['Nx'], solver_params['Ny'], solver_params['Nz']

    core_root, data_root = get_roots()

    if (L == np.inf): L = 'np.inf'

    f = open(f'{data_root}/newtonWrapper/drive_flow_{label}.py', 'w')

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

    f.write('from tools.plotter import plot_from_array\n')

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

    f.write("timestepper = TimeStepper(material_params, system_params, solver_params, no_base=True)\n")
    f.write(f"timestepper.ic(ic_file='{data_root}newtonWrapper/ic_{label}.h5')\n")
    f.write(f"timestepper.update_dy()\n")   # required as ic_file only contains u,v,p etc, not uy, vy...
    f.write('timestepper.solver.sim_time = 0.0\n')
    f.write('timestepper.solver.iteration = 0\n')
    f.write('timestepper.solver.start_time = timestepper.solver.sim_time\n')

    f.write('timestepper.solver.stop_sim_time  = np.inf\n')
    f.write('timestepper.solver.stop_wall_time = np.inf\n')
    f.write('timestepper.solver.stop_iteration = %i\n'%(int(Nstep)+1))
    f.write('ifreq = 100\n')
    f.write(f"snapshots = timestepper.solver.evaluator.add_file_handler('{data_root}newtonWrapper/snapshots_{label}',\n")
    f.write("           iter = %i, max_writes=50, mode='overwrite')\n"%(int(Nstep)))
    f.write('snapshots.add_system(timestepper.solver.state)\n')
        
    f.write("logger.info('Starting loop')\n")
    f.write('while timestepper.solver.ok:\n')
    f.write("    timestepper.solver.step(solver_params['dt'])\n")
    f.write('    if timestepper.solver.iteration % ifreq == 0:\n')
    f.write("        KE = timestepper.flow.volume_average('KE')\n")
    f.write("        trace = timestepper.flow.volume_average('trace')\n")
    f.write("        logger.info('It: %i, t: %.3e, dt: %.2e trace: %.4e, min(det(C)): %.4e, KE: %.4e'\n")
    f.write("                 % (timestepper.solver.iteration, timestepper.solver.sim_time, timestepper.dt,\n")
    f.write("                 trace, timestepper.flow.min('det(C)'),\n")
    f.write("                 KE))\n")
    f.write('end_time = time.time()\n')
    f.close()

def converge_TW(material_params, system_params, solver_params, ic_file, output_fpath, gmres_params, T_guess=None, label=''):

    logger.info('Setting up configuration')
    fc_io = TimeStepper(material_params, system_params, solver_params)

    """
    The IO setup is used to keep the structure
    of a Dedalus file. It also loads the parameters to the 
    """

    # the variables which GMRES is performed on (NB we can reconstruct uy etc from these)
    data_vars = ['p', 'u', 'v', 'w', 'c11', 'c22', 'c33', 'c12', 'c23', 'c13']

    # how is each variable scaled in the algorithm - NB c11 is normally orders of magnitude larger than p,u,v

    processed_file = process_ic(material_params, system_params, solver_params, ic_file_in=ic_file, label=label)

    if T_guess is None:
        T_guess = predict_period(material_params, system_params, solver_params, label=label)

    reader = DIO.dnsReader(fc_io, data_vars)
    reader.read_field('u_r0', file_name=processed_file)
    u0 = reader.return_field_array('u_r0')

    _, data_root = get_roots()
    fname = os.path.join(output_fpath, output_fpath.split('/')[-2]+'_s1.h5')

    eq_solve = EQ.ecsSearch(u0, fc_io, T_init=T_guess, output_name=fname, data_vars=data_vars,
                            data_root=data_root, label=label,
                            write_driveFile=write_driveFile,
                            process_tasks_func=process_fields_to_tasks,
                            dns_dt_sim=solver_params['dt'],
                            **gmres_params)

    eq_solve.iterate()

    # if success:
        # turn an .h5 file that has correct ['p', 'u', 'v', 'c11', 'c22', 'c33', 'c12'] into one with all tasks correct
    #     process_fields_to_tasks(material_params, system_params, solver_params, fname, label)
    # else:
    #     # save laminar state as have passed the saddle node
    #     save_laminar_state(material_params, system_params, solver_params, fname)


def converge_TW_branch(material_params, system_params, solver_params, continuation_param, ic_file_in0, ic_file_in_1, save_folder, gmres_params, param_scaling_func=lambda x: x, inv_param_scaling_func=lambda x: x, label=''):
    
    """
    The IO setup is used to keep the structure
    of a Dedalus file.
    """

    logger.info('Setting up configuration')
    fc_io = TimeStepper(material_params, system_params, solver_params)

    # ensure we have a start_processed file...
    process_ic(material_params, system_params, solver_params, ic_file_in=ic_file_in0, label=label)

    # the variables which GMRES is performed on (NB we can reconstruct uy etc from these)
    data_vars = ['p', 'u', 'v', 'w', 'c11', 'c22', 'c33', 'c12', 'c23', 'c13']

    # obtain 2 intial states on branch, and concatenate with period and the continuation parameter
    reader = DIO.dnsReader(fc_io, data_vars)
    reader.read_field(['u_r0', 'u_r_1'], file_name=[ic_file_in0, ic_file_in_1])

    X_r0 = reader.return_field_array('u_r0', include_T=True, include_continuation_param=continuation_param, param_scaling_func=param_scaling_func)
    X_r_1 = reader.return_field_array('u_r_1', include_T=True, include_continuation_param=continuation_param, param_scaling_func=param_scaling_func)

    _, data_root = get_roots()

    eq_solve = EQ.ecsArclength(fc_io, X_r0, X_r_1, 
                            continuation_param, 
                            T_init=X_r0[-2], dns_dt_sim=solver_params['dt'],
                            param_scaling_func=param_scaling_func, inv_param_scaling_func=inv_param_scaling_func, 
                            write_driveFile=write_driveFile, process_tasks_func=process_fields_to_tasks,
                            data_vars=data_vars, data_root=data_root, 
                            **gmres_params)


    os.makedirs(save_folder, exist_ok=True)
    eq_solve.iterate_branch(40, output_dir=save_folder)

def _get_arrow_junction(channel):

    channel.set_scale(1)
    c11 = channel.c11['g']
    c11_mean = np.mean(c11, axis=(1,2))
    rough_x = channel.x[np.argmax(c11_mean),0,0] % channel.Lx
    interp_grid = np.linspace(max(rough_x - 0.5, 0), min(rough_x + 0.5, channel.Lx), 500)   
    interp_c11 = np.interp(interp_grid, channel.x[:,0,0], c11_mean)
    x = interp_grid[np.argmax(interp_c11)]
    return x

####################################################################################################################################
# EVERYTHING UNDER HERE DOESN'T NEED TO CHANGE WHEN A NEW SYSTEM IS MADE
####################################################################################################################################


def process_ic(material_params, system_params, solver_params, ic_file_in='newtonWrapper/start.h5',
               label=''):
    """
    converts ic of generic Nx, Ny to the correct Nx, Ny for the simulation
    """

    ic_file_out=f'start_processed_{label}'

    _, data_root = get_roots()
    save_folder = os.path.join(data_root, 'newtonWrapper', ic_file_out)

    os.makedirs(os.path.join(data_root, 'newtonWrapper'), exist_ok=True)
    
    channel = TimeStepper(material_params, system_params, solver_params, logger_on=True, no_base=True)
    channel.ic(ic_file_in)
    channel.update_dy()

    snapshots = channel.solver.evaluator.add_file_handler(save_folder, \
                                                          iter=1, max_writes=100, mode='overwrite')
    snapshots.add_system(channel.solver.state, layout='g')

    channel.solver.step(dt=1e-6)

    post.merge_process_files(save_folder, cleanup=True)

    made_file = os.path.join(save_folder, save_folder.split('/')[-1] + '_s1.h5')
    destination = save_folder + '.h5'
    shutil.move(made_file, destination)
    os.rmdir(save_folder)

    return destination

def predict_period(material_params, system_params, solver_params, label=''):
    """
    Predicted the period T of a travelling wave solutions
    """

    _, data_root = get_roots()
    fpath = os.path.join(data_root, 'newtonWrapper', f'start_processed_{label}.h5')

    channel = TimeStepper(material_params, system_params, solver_params, logger_on=True, no_base=True)
    channel.ic(fpath)
    channel.update_dy()

    start_x_track = _get_arrow_junction(channel)
    start_sim_time = channel.solver.sim_time

    def track_TW(x_track_old):
        x_track_old_mod = x_track_old % channel.Lx
        x_track_old_num = x_track_old // channel.Lx

        x_track_new_mod = _get_arrow_junction(channel)

        if (x_track_new_mod - x_track_old_mod) > channel.Lx - 0.1:  # going leftwards
            x_track_new_num = x_track_old_num - 1
            logger.info('Looped round leftward!')
        elif (x_track_old_mod - x_track_new_mod) > channel.Lx - 0.1:  # going rightwards
            x_track_new_num = x_track_old_num + 1
            logger.info('Looped round rightward!')
        else:
            x_track_new_num = x_track_old_num

        x_track_new = x_track_new_num * channel.Lx + x_track_new_mod

        return x_track_new

    def estimate_T(x_track):

        c = (x_track - start_x_track) / (channel.solver.sim_time - start_sim_time)
        T = channel.Lx / np.abs(c)

        logger.info(
            f'Current estimate of T is {T}, distance travelled is {x_track - start_x_track} and time is {channel.solver.sim_time - start_sim_time}')

        return T

    T_pred_list = []

    x_track_new = start_x_track

    while True:

        x_track_old = x_track_new

        channel.solver.step(dt=solver_params['dt'])
        x_track_new = track_TW(x_track_old)
        if channel.solver.iteration % 100 == 0:
            T = estimate_T(x_track_new)
            T_pred_list.append(T)
            if len(T_pred_list) > 10 and np.std(T_pred_list[-6:]) / np.mean(T_pred_list[-6:]) < 0.002:
                return T
            elif len(T_pred_list) > 200:
                return np.mean(T_pred_list[100:])
           
def process_fields_to_tasks(material_params, system_params, solver_params, h5_file, label=''):
    channel = TimeStepper(material_params, system_params, solver_params, logger_on=True)
    channel.ic(h5_file)
    channel.update_dy()
    
    _, data_root = get_roots()
    save_folder = os.path.join(data_root, 'newtonWrapper')
    os.makedirs(save_folder, exist_ok=True)
    save_folder = os.path.join(save_folder, f'temp_{label}')
    
    snapshots = channel.solver.evaluator.add_file_handler(save_folder, \
                                                          iter=1, max_writes=100, mode='overwrite')

    channel.add_tasks_to_handler(snapshots, save_all_fields=True)
    
    channel.solver.step(dt=1e-8)

    post.merge_process_files(save_folder, cleanup=True)

    made_file = os.path.join(save_folder, save_folder.split('/')[-1] + '_s1.h5')

    # get period T
    file1 = h5py.File(h5_file, mode='r+')
    T = np.array(file1['tasks']['T'])
    file1.close()

    # save period and params, needed for continuations
    file = h5py.File(made_file, mode='r+')
    file['tasks']['T'] = T
    for material_param_name, material_param in material_params.items():
        file['tasks'][material_param_name] = material_param
    for solver_param_name, solver_param in solver_params.items():
        file['tasks'][solver_param_name] = solver_param
    for system_param_name, system_param in system_params.items():
        file['tasks'][system_param_name] = system_param
    file.close()

    shutil.move(made_file, h5_file)

    logger.info(f"Processed and saved to {h5_file}")

def save_laminar_state(material_params, system_params, solver_params, h5_file_output):
    
    channel = TimeStepper(material_params, system_params, solver_params, logger_on=True)
    channel.ic(ic_file=None)

    _, data_root = get_roots()
    save_folder = os.path.join(data_root, 'newtonWrapper')
    os.makedirs(save_folder, exist_ok=True)
    save_folder = os.path.join(save_folder, 'temp')

    snapshots = channel.solver.evaluator.add_file_handler(save_folder, \
                                                          iter=1, max_writes=100, mode='overwrite')

    snapshots.add_system(channel.solver.state, layout='g')
    snapshots.add_task("integ( U ** 2 + V ** 2, 'x', 'y') / Lx / Ly / 2", layout='g', name='<KE_lam>')
    snapshots.add_task("integ(C11 + C22 + C33, 'y', 'x')/(Lx*Ly)", layout='g', name='<trace_lam>')
    snapshots.add_task("integ(c11 + c22 + c33, 'y', 'x')/(Lx*Ly)", layout='g', name='<trace>')
    snapshots.add_task("integ( u ** 2 + v ** 2, 'x', 'y') / Lx / Ly / 2", layout='g', name='<KE>')

    channel.solver.step(dt=1e-8)

    post.merge_process_files(save_folder, cleanup=True)

    made_file = os.path.join(save_folder, save_folder.split('/')[-1] + '_s1.h5')

    directory_path = '/'.join(h5_file_output.split('/')[:-1])
    os.makedirs(directory_path, exist_ok=True)
    shutil.move(made_file, h5_file_output)
    logger.info(f"Laminar state saved to {h5_file_output}")

