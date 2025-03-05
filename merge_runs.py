import copy
import logging
import os

import numpy as np
from tools.misc_tools import get_roots
from dedalus import public as de
from dedalus.tools import post

from mpi4py import MPI
import sys

logger = logging.getLogger(__name__)

_, data_root = get_roots()

simulations_path = os.path.join(data_root, 'simulations')
subdirectories = os.listdir(simulations_path)

for subdirectory in subdirectories:
    subdirectory_path = os.path.join(simulations_path, subdirectory)
    for base_dir in os.listdir(subdirectory_path):
        base_path = os.path.join(subdirectory_path, base_dir)
        post.merge_process_files(base_path, cleanup=True)
        # for split_dir in os.listdir(base_path):
            # split_path = os.path.join(base_path, split_dir)
            # post.merge_process_files(split_path, cleanup=True)

edge_path = os.path.join(data_root, 'edge_track')
subdirectories = os.listdir(edge_path)

for subdirectory in subdirectories:
    field1_path = os.path.join(simulations_path, subdirectory, 'field1')
    field2_path = os.path.join(simulations_path, subdirectory, 'field2')

    for base_dir in os.listdir(field1_path):
        base_path = os.path.join(subdirectory_path, base_dir)
        post.merge_process_files(base_path, cleanup=True)
    for base_dir in os.listdir(field2_path):
        base_path = os.path.join(subdirectory_path, base_dir)
        post.merge_process_files(base_path, cleanup=True)
        # for split_dir in os.listdir(base_path):
            # split_path = os.path.join(base_path, split_dir)
            # post.merge_process_files(split_path, cleanup=True)



# simulations_files = os.listdir(simulations_path)

# for file in simulations_files:
#     subdir_path = os.path.join(simulations_path, file)
#     if os.path.isdir(subdir_path):
#         merge_files_in_dir(subdir_path)


logger.info('Done')