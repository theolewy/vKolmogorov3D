import os
import sys 
import numpy as np
from cfd_tools.edge_track.edge_track import edgeTrack
from tools.edge_tools import write_driveFile
import logging
import csv
from tools.misc_tools import get_roots

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
                 'Nz': 64,
                 'dt': 5e-3}

a1     = 10.02   # when to declare ultimate state as field 1
a2     = 10.15   # when to declare ultimate state as field 2
lamb   = 0.5    # current lambda
lamb1  = 0.0    # lower bound
lamb2  = 1.0    # upper bound
accmin = 1e-12
# Tmin   = 5
Tmin   = 0.01

_, data_root = get_roots()
data_root = data_root + 'edge_track/'
lambda_root = data_root + '/lambda.out'

# obtain lambda from lambda file if it exists
if os.path.exists(lambda_root):
        logger.info('Loading lambda bounds from file...')
        with open(lambda_root) as file:
                csv_reader = csv.reader(file, delimiter=' ')
                for row in csv_reader:
                        lamb, lamb1, lamb2, = row[0], row[1], row[2]
        logger.info(f'lambda={lamb}, lambda1={lamb1}, lambda2={lamb2}..')

# if last run unfinished... initialise from it?

edge_tracker = edgeTrack(material_params, system_params, solver_params,
                        a1, a2, lamb, lamb1, lamb2, accmin, Tmin, 
                        variables=['u', 'v', 'w', 'p', 'c11', 'c12', 'c22', 'c33', 'c13', 'c23'],
                        write_driveFile=write_driveFile, data_root=data_root)

#------------ MAIN LOOP --------------------
logger.info('Here Field 1 is laminar, and Field 2 is the localised 3D AH')

logger.info('Starting edge tracking loop')

while edge_tracker.acc >= accmin:
        
        edge_tracker.build_ic()

        edge_tracker.run_traj(nproc=32)

        edge_tracker.up_lamb()