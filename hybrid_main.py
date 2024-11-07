import time
import random
import os
import numpy as np

import hybrid_sdac as hyb_sd
from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron

def hybrid_main(mps_fn='', results_dir='results', max_time=300, sd_method='dual_simplex', reset=False):

				print('Reading {}...'.format(mps_fn))
				c, B, d, A, b = read_mps_preprocess(mps_fn)
				print('Building polyhedron...')
				P = Polyhedron(B, d, A, b, c)

				print('Finding feasible solution...')
				x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

				print('\nSolving with simplex method...')
				lp_result = P.solve_lp(verbose=False, record_objs=True)
				print('\nSolution using simplex method:')
				print(lp_result)

				print('\nSolving with original steepest descent...')
				old_sd_result = hyb_sd.old_sdac(mps_fn,results_dir, max_time, reset,sd_method)

				# print('\nSolving with alt steepest descent...')
				# alt_sd_result = hyb_sd.alt_sdac(mps_fn,results_dir, max_time, reset,sd_method)

				print('\nSolving with hybrid steepest descent with init active inds...')
				hybrid_init_act_sd_result = hyb_sd.hybrid_init_act(mps_fn,results_dir, max_time, reset,sd_method)

				print('\nSolving with hybrid steepest descent with updated active inds...')
				hybrid_up_act_sd_result = hyb_sd.hybrid_up_act(mps_fn,results_dir, max_time, reset,sd_method)


				#####Save results
				if results_dir:
					if not os.path.exists(results_dir): os.mkdir(results_dir)
					if mps_fn:
						prefix = os.path.basename(mps_fn).split('.')[0]
						lp_fn = os.path.join(results_dir, prefix + '_lp.p')
						old_sd_fn = os.path.join(results_dir, prefix + '_old_sd.p')
						# alt_sd_fn = os.path.join(results_dir, prefix + '_alt_sd.p')
						reg_hybrid_init_act_sd_fn = os.path.join(results_dir, prefix + '_reg_hybrid_up_act_sd.p')
						reg_hybrid_up_act_sd_fn = os.path.join(results_dir, prefix + '_reg_hybrid_init_act_sd.p')
						lp_result.save(lp_fn)
						old_sd_result.save(old_sd_fn)
						# alt_sd_result.save(alt_sd_fn)
						hybrid_init_act_sd_result.save(reg_hybrid_init_act_sd_fn)
						hybrid_up_act_sd_result.save(reg_hybrid_up_act_sd_fn)