import time
import random
import os
import numpy as np

import init_sdac
from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron

def main_mat_in(c, B, d, A = None, b = None, mps_fn='', results_dir='results', max_time=300, sd_method='dual_simplex', reset=False):

				
				print('Building polyhedron...')
				P = Polyhedron(B, d, A, b, c)

				print('Finding feasible solution...')
				x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

				print('\nSolving with simplex method...')
				lp_result = P.solve_lp(verbose=False, record_objs=True)
				print('\nSolution using simplex method:')
				print(lp_result)

				print('\nSolving with old steepest descent...')
				old_sd_result  = init_sdac.old_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				print('\nSolution for {} using old steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
				print(old_sd_result)

				print('\nSolving with new steepest descent...')
				new_sd_result = init_sdac.new_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				print('\nSolution for {} using new steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
				print(new_sd_result)

				print('\nSolving with trivial steepest descent...')
				trivial_sd_result = init_sdac.trivial_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				# x_1_time = init_sdac.trivial_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				print('\nSolution for {} using new steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
				print(trivial_sd_result)

				# return old_sub, old_solve, old_phase, new_sub, new_solve, new_phase
				# return x_1_time

				# #####Save results
				if results_dir:
					if not os.path.exists(results_dir): os.mkdir(results_dir)
					if mps_fn:
						prefix = os.path.basename(mps_fn).split('.')[0]
						lp_fn = os.path.join(results_dir, prefix + '_lp.p')
						old_sd_fn = os.path.join(results_dir, prefix + '_old_sd.p')
						new_sd_fn = os.path.join(results_dir, prefix + '_new_sd.p')
						trivial_sd_fn = os.path.join(results_dir, prefix + '_trivial_sd.p')
						lp_result.save(lp_fn)
						old_sd_result.save(old_sd_fn)
						new_sd_result.save(new_sd_fn)
						trivial_sd_result.save(trivial_sd_fn)