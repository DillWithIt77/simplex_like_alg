import time
import random
import os
import numpy as np

import init_sdac
from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron

def main(mps_fn='', results_dir='results', max_time=300, sd_method='dual_simplex', reset=False, **kwargs):
				mat_input = kwargs.get('mat_input', False)
				A = kwargs.get('A', [])
				B = kwargs.get('B', [])
				b = kwargs.get('b', [])
				d = kwargs.get('d', [])
				c = kwargs.get('c', [])


				if mat_input == False:
					print('Reading {}...'.format(mps_fn))
					c, B, d, A, b = read_mps_preprocess(mps_fn)
				else: 
					print('Using matrix input')

				print('Building polyhedron...')
				P = Polyhedron(B, d, A, b, c)

				print('Finding feasible solution...')
				x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

				print('\nSolving with simplex method...')
				lp_result = P.solve_lp(verbose=False, record_objs=True)
				print('\nSolution using simplex method:')
				print(lp_result)

				print('\nSolving with old steepest descent...')
				if mat_input == False:
					old_sd_result  = init_sdac.old_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				else:
					old_sd_result  = init_sdac.old_sdac(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
				print('\nSolution for {} using old steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
				print(old_sd_result)

				print('\nSolving with new steepest descent...')
				if mat_input == False:
					new_sd_result = init_sdac.new_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				else:
					new_sd_result = init_sdac.new_sdac(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
				print('\nSolution for {} using new steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
				print(new_sd_result)

				print('\nSolving with trivial steepest descent...')
				if mat_input == False:
					trivial_sd_result = init_sdac.trivial_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				else:
					trivial_sd_result = init_sdac.trivial_sdac(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
				# x_1_time = init_sdac.trivial_sdac(mps_fn,results_dir, max_time, reset,sd_method)
				print('\nSolution for {} using trivial steepest-descent augmentation:'.format(os.path.basename(mps_fn)))
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