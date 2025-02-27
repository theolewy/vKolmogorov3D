import sys 
import numpy as np
from cfd_tools.edge_track.edge_track import edgeTrack
from tools.edge_tools import write_driveFile
import logging

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
                 'dt': 2e-3}

a1     = 0.05
a2     = 0.09
lamb   = 0.6
lamb1  = 0.0
lamb2  = 1.0
accmin = 1e-12
Tmin   = 50

_, data_root = get_roots()
data_root = data_root + 'edge_track/'
edge_tracker = edgeTrack(material_params, system_params, solver_params,
                        a1, a2, lamb, lamb1, lamb2, accmin, Tmin, 
                        variables=['u', 'v', 'w', 'p', 'c11', 'c12', 'c22', 'c33', 'c13', 'c23'],
                        write_driveFile=write_driveFile, data_root=data_root)

#------------ MAIN LOOP --------------------

logger.info('Starting edge tracking loop')

logger.info('Here Field 1 is laminar, and Field 2 is the localised 3D AH')

while edge_tracker.acc >= accmin:
        
        edge_tracker.build_ic()

        edge_tracker.run_traj(nproc=32)

        edge_tracker.up_lamb()