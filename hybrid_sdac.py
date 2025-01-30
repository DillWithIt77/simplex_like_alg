import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import issparse
import gurobipy as gp
import math
import sympy
import contextlib
import time
import os
from utils import result, METHODS, INF, EPS
from mps_reader_preprocessor import read_mps_preprocess
from polyhedral_model import PolyhedralModel
from polyhedron import Polyhedron


def old_sdac(mps_fn, results_dir, max_time, reset, sd_method, **kwargs):
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        ### Build Initial Polyhedron from file
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else:
        print('Using matrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    # print(len(active_inds))
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    t2 = time.time()
    iter_times.append(t2 - t1)

    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    
    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
        pts_visited.append(x_current)
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        # compute steepest-descent direction
        # print(f'Number of Active Inds for Iteration {iteration}: {len(active_inds)}')
        pm.set_active_inds(active_inds)
        descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(
                                                                                                verbose=verbose)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited)

def alt_sdac(mps_fn, results_dir, max_time, reset, sd_method, **kwargs):
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        ### Build Initial Polyhedron from file
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else:
        print('Using matrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    # print(len(active_inds))
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method, alt = True)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    t2 = time.time()
    iter_times.append(t2 - t1)

    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    
    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
        pts_visited.append(x_current) 
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        pm.set_active_inds(active_inds)
        descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(
                                                                                                verbose=verbose)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited)

def hybrid_up_act(mps_fn, results_dir, max_time, reset, sd_method, **kwargs):
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        ### Build Initial Polyhedron from file
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else:
        print('Using matrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    if P.m_A != 0:
        if issparse(A):
            u, s, vt = svds(A, k=min(A.shape)-1)
            rank_A = np.sum(s > tol)
        else:
            rank_A = np.linalg.matrix_rank(A)
    else: rank_A = 0

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}
    active_inds_it = []
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    active_inds_it.append(active_inds)
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    step_types = []
    step_types.append(0)
    simp_times = []
    aug_times = []
    facet_times = []
    t2 = time.time()
    iter_times.append(t2 - t1)

    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    restrict_facets = []
    restrict_facets.append(False)
    did_simp = False

    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        if did_simp == False:
            x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
            pts_visited.append(x_current)


        if len(active_inds) < (P.n - rank_A)-math.floor(P.n/10):
            restrict_facets.append(True)
        else:
            restrict_facets.append(False)
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        active_inds_it.append(active_inds)
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        # compute steepest-descent direction
        if (restrict_facets[-2] and not restrict_facets [-1] and P.bfs_soln_chck(x_current)):
            facet_end = time.time()
            facet_times.append(facet_end-facet_start)
            simp_start = time.time()
            did_simp = True
            num_dec_places = Polyhedron.num_aft_dec(obj_value)
            x_current = P.second_vert(x_current, obj_value, num_dec_places, verbose=False)
            pts_visited.append(x_current)
            active_inds = P.get_active_constraints(x_current)
            alpha = 0.0
            simp_end = time.time()
            simp_times.append(simp_end-simp_start)
            step_types.append(2)
        elif restrict_facets[-1]:
            facet_start = time.time()
            pm.set_active_inds(active_inds, inds = active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(1)
        else:
            if restrict_facets[-2] and not restrict_facets[-1]:
                facet_end = time.time()
                facet_times.append(facet_end-facet_start)
            pm.set_active_inds(active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(0)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited,
                  step_types = step_types, simp_times = simp_times, facet_times = facet_times)

def hybrid_init_act(mps_fn, results_dir, max_time, reset, sd_method, **kwargs):
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        ### Build Initial Polyhedron from file
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else: 
        print('Using matrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    if P.m_A != 0:
        if issparse(A):
            u, s, vt = svds(A, k=min(A.shape)-1)
            rank_A = np.sum(s > tol)
        else:
            rank_A = np.linalg.matrix_rank(A)
    else: rank_A = 0

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}
    active_inds_it = []
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    active_inds_it.append(active_inds)
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    step_types = []
    step_types.append(0)
    simp_times = []
    aug_times = []
    facet_times = []
    t2 = time.time()
    iter_times.append(t2 - t1)

    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    restrict_facets = []
    restrict_facets.append(False)
    did_simp = False

    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        if did_simp == False:
            x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
            pts_visited.append(x_current)

        if len(active_inds) < (P.n - rank_A)-math.floor(P.n/10):
            restrict_facets.append(True)
        else:
            restrict_facets.append(False)
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        active_inds_it.append(active_inds)
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        if restrict_facets[-1] and not restrict_facets[-2]:
            init_inds = active_inds
        
        # compute steepest-descent direction
        if (restrict_facets[-2] and not restrict_facets[-1] and P.bfs_soln_chck(x_current)):
            facet_end = time.time()
            facet_times.append(facet_end-facet_start)
            simp_start = time.time()
            did_simp = True
            num_dec_places = Polyhedron.num_aft_dec(obj_value)
            x_current = P.second_vert(x_current, obj_value, num_dec_places, verbose=False)
            pts_visited.append(x_current)
            active_inds = P.get_active_constraints(x_current)
            alpha = 0.0
            simp_end = time.time()
            simp_times.append(simp_end-simp_end)
            step_types.append(2)
        elif restrict_facets[-1]:
            facet_start = time.time()
            pm.set_active_inds(active_inds, inds = init_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(1)
        else:
            if restrict_facets[-2] and not restrict_facets[-1]:
                facet_end = time.time()
                facet_times.append(facet_end-facet_start)
            pm.set_active_inds(active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(0)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited,
                  step_types = step_types)

def alt_hybrid_up_act(mps_fn, results_dir, max_time, reset, sd_method, **kwargs):
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        ### Build Initial Polyhedron from file
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else:
        print('Using matrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    if P.m_A != 0:
        if issparse(A):
            u, s, vt = svds(A, k=min(A.shape)-1)
            rank_A = np.sum(s > tol)
        else:
            rank_A = np.linalg.matrix_rank(A)
    else: rank_A = 0

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method, alt = True)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}
    active_inds_it = []
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    active_inds_it.append(active_inds)
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    step_types = []
    step_types.append(0)
    t2 = time.time()
    iter_times.append(t2 - t1)


    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    restrict_facets = []
    restrict_facets.append(False)
    did_simp = False

    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        if not did_simp:
            x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
            pts_visited.append(x_current)

        if len(active_inds) < (P.n - rank_A)-math.floor(P.n/10):
            restrict_facets.append(True)
        else:
            restrict_facets.append(False)
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        active_inds_it.append(active_inds)
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        # compute steepest-descent direction
        if (restrict_facets[-1] and P.bfs_soln_chck(x_current)):
            did_simp = True
            num_dec_places = Polyhedron.num_aft_dec(obj_value)
            x_current = P.second_vert(x_current, obj_value, num_dec_places, verbose=False)
            pts_visited.append(x_current)
            active_inds = P.get_active_constraints(x_current)
            alpha = 0.0
            step_types.append(2)
        elif restrict_facets[-1]:
            pm.set_active_inds(active_inds, inds = active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(1)
        else:
            pm.set_active_inds(active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(0)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited,
                  step_types = step_types)

def alt_hybrid_init_act(mps_fn, results_dir, max_time, reset, sd_method, **kwargs):
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        ### Build Initial Polyhedron from file
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else:
        print('Using matrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    if P.m_A != 0:
        if issparse(A):
            u, s, vt = svds(A, k=min(A.shape)-1)
            rank_A = np.sum(s > tol)
        else:
            rank_A = np.linalg.matrix_rank(A)
    else: rank_A = 0

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method, alt = True)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}
    active_inds_it = []
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    active_inds_it.append(active_inds)
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    step_types = []
    step_types.append(0)
    t2 = time.time()
    iter_times.append(t2 - t1)


    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    restrict_facets = []
    restrict_facets.append(False)
    did_simp = False

    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        if not did_simp:
            x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
            pts_visited.append(x_current)

        if len(active_inds) < (P.n - rank_A)-math.floor(P.n/10):
            restrict_facets.append(True)
        else:
            restrict_facets.append(False)
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        active_inds_it.append(active_inds)
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        if restrict_facets[-1] and not restrict_facets[-2]:
            init_inds = active_inds
        
        # compute steepest-descent direction
        if (restrict_facets[-1] and P.bfs_soln_chck(x_current)):
            did_simp = True
            num_dec_places = Polyhedron.num_aft_dec(obj_value)
            x_current = P.second_vert(x_current, obj_value, num_dec_places, verbose=False)
            pts_visited.append(x_current)
            active_inds = P.get_active_constraints(x_current)
            alpha = 0.0
            step_types.append(2)
        elif restrict_facets[-1]:
            pm.set_active_inds(active_inds, inds = init_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(1)
        else:
            pm.set_active_inds(active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(0)
        
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited,
                  step_types = step_types)

def heuristic_no_t(mps_fn, results_dir, max_time, reset, sd_method,**kwargs):
    s = kwargs.get('s', 1)
    mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
        print(f'Reading {mps_fn}...')
        c, B, d, A, b = read_mps_preprocess(mps_fn)
    else:
        print('Using amtrix input')

    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)
    
    if P.m_A != 0:
        if issparse(A):
            u, s, vt = svds(A, k=min(A.shape)-1)
            rank_A = np.sum(s > tol)
        else:
            rank_A = np.linalg.matrix_rank(A)
    else: rank_A = 0

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    M = -float('inf')
    omega = float('inf')
    if save_first_steps:
        np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds, M, omega = P.get_active_constraints(x_current, alt = True)

    R = omega/((P.m_B-P.n)*M)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method, alt = True)

    if first_warm_start is not None:
        print('Using custom warm start')
        pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}
    active_inds_it = []
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    active_inds_it.append(active_inds)
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    pts_visited = []
    pts_visited.append(x_current)
    step_types = []
    step_types.append(0)
    omegas = []
    omegas.append(omega)
    Ms = []
    Ms.append(M)
    t2 = time.time()
    iter_times.append(t2 - t1)


    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    restrict_facets = False
    # restrict_facets.append(False)
    did_simp = False
    first_step = True
    G = -float('inf')
    while abs(steepness) > EPS:
        
        t3 = time.time()
        if reset:
            pm.reset()
        
        # take maximal step
        if not did_simp:
            print('\nIteration {}'.format(iteration))
            print('took maximal step')
            x_current, alpha, active_inds = P.take_maximal_step(descent_direction, y_pos, y_neg)
            pts_visited.append(x_current)

        if first_step:
            G_guess = (1/R)*P.c.dot(pts_visited[-1] - pts_visited[-2])
            G = G_guess - P.c.dot(pts_visited[-1] - pts_visited[-2])
            first_step = False
        elif P.c.dot(pts_visited[-1] - pts_visited[-2]) >= (R/s)*G:
            G_old = G
            G = G_old - P.c.dot(pts_visited[-1] - pts_visited[-2])
        else:
            restrict_facets = True

        if P.bfs_soln_chck(x_current) and restrict_facets:
            M = -float('inf')
            omega = float('inf')
            for i in range(P.m_B):
                diff = P.d[i] - P.B_x_current[i]
                if diff > M:
                    M = diff
                if diff > 0 and diff < omega:
                    omega = diff
            R = omega/((P.m_B-P.n)*M)
            first_step = True
        
        if iteration % 50 == 0 or iteration == 1:
            print('\nIteration {}'.format(iteration))
            print('Objective: {}'.format(obj_value))
            print('Steepness: {}'.format(steepness))
            print('Step length: {}'.format(alpha))
        
        t4 = time.time()
        active_inds_it.append(active_inds)
        obj_value = P.c.dot(x_current)
        obj_values.append(obj_value)
        omegas.append(omega)
        Ms.append(M)
        iter_times.append(t4 - t1)
        sub_times['step'].append(t4 - t3) 
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)     
                
        if math.isinf(alpha):
            # problem is unbounded
            result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        if restrict_facets:
            print('did restrict facets')
            pm.set_active_inds(active_inds, inds = active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(1)
        else:
            print('did not restrict facets')
            pm.set_active_inds(active_inds)
            did_simp = False
            descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
            step_types.append(0)
    
        t5 = time.time()
        sub_times['sd'].append(t5 - t4)
        sub_times['solve'].append(solve_time)
        sub_times['phase_times'].append(phase_times)
        simplex_iters.append(num_steps)
        
        iteration += 1
        current_time = t5 - t1
        if current_time > max_time:
            result(status=2)
        if iteration <= save_first_steps:
            np.save('solutions/{}_{}.npy'.format(problem_name, iteration), x_current)

    t6 = time.time()
    total_time = t6 - t1   
    print('Total time for steepest-descent scheme: {}'.format(total_time))

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes), solve_time=total_time,
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values, pts_visited = pts_visited,
                  step_types = step_types, omegas = omegas, Ms = Ms)