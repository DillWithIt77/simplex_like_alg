import pickle
import random
import numpy as np

METHODS = {'auto': -1,
		   'primal_simplex': 0,
           'dual_simplex': 1,
           'barrier': 2,
           'concurrent': 3,
           'deterministic_concurrent': 4,
           'deterministic_concurrent_simplex': 5,}

INF = 10e100
EPS = 10e-8

def avg(x):
    return float(sum(x)) / float(len(x))


class result:  
    def __init__(self, status, x=None, obj=None, n_iters=None, solve_time=None, iter_times=[], alg_type='simplex',
                 circuits=[], steps=[], simplex_iters=[], solve_times=[], sub_times=None,
                 obj_values=[], iter_counts=[], 
                 step_types = [], pts_visited = [], aug_times = [], facet_times = [], simp_times = [], omegas = [], Ms = []):
        self.status = status
        # self.switch_iter = switch_iter
        self.x = x
        self.obj = obj
        self.n_iters = n_iters
        self.iter_times = iter_times
        self.solve_time = solve_time
        self.iter_counts = iter_counts
        self.alg_type = alg_type
        
        self.circuits = circuits
        self.steps = steps
        self.simplex_iters = simplex_iters
        self.solve_times = solve_times
        
        self.sub_times = sub_times
        self.obj_values = obj_values

        self.step_types = step_types
        self.pts_visited = pts_visited
        self.aug_times = aug_times
        self.facet_times = facet_times
        self.simp_times = simp_times
        self.omegas = omegas
        self.Ms = Ms
          
    def __str__(self):
        if self.status == 1:
            return ('Problem is unbounded.'
                        + '\nSteepest descent unbounded circuit: ' + str(self.circuits[-1].T)
                    )
        elif self.status == 0:
            output = ('\nOptimal objective: {}'.format(self.obj)
                   + '\nTotal solve time: {}'.format(self.solve_time)
                   + '\nNumber of iterations: {}'.format(self.n_iters)
            )
            if self.alg_type == 'steepest-descent':
                output += ('\nFirst simplex iterations {}'.format(self.simplex_iters[0])
                       + '\nAverage num simplex iterations {}'.format(sum(self.simplex_iters)/len(self.simplex_iters))
                       + '\nTotal simplex iterations: {}'.format(sum(self.simplex_iters))
                       + '\nFirst solve time {}'.format(self.solve_times[0])
                       + '\nAverage solve time {}'.format(sum(self.solve_times)/len(self.solve_times))
                       + '\nTotal solve time: {}'.format(sum(self.solve_times))
                       )
            return output
        else:
            return 'Problem unsolved'
        
    def save(self, fn):
        results = {'obj': self.obj, 
                   'obj_values': self.obj_values,
                   'n_iters': self.n_iters, 
                   'solve_time_total': self.solve_time,
                   'iter_times': self.iter_times,
                   'iter_counts': self.iter_counts,
                   'alg_type': self.alg_type}
        if self.alg_type == 'steepest-descent':
            results['simplex_iters'] = self.simplex_iters
            results['solve_times'] = self.solve_times
            results['sub_times'] = self.sub_times

            results['step_types'] = self.step_types
            results['pts_visited'] = self.pts_visited
            results['aug_times'] = self.aug_times
            results['facet_times'] = self.facet_times
            results['simp_times'] = self.simp_times
            results['omegas'] = self.omegas
            results['Ms'] = self.Ms

            
        with open(fn, 'wb') as f:
            pickle.dump(results, f)
