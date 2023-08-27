import math
import numpy

def steepest_descent_augmentation_algorithm(P, c, x):
    """
Given a polyhedron P with feasible point x and an objective function c,
solve the linear program min{c^T x : x in P} via the steepest descent circuit augmentation scheme.
Returns result object containing optimal solution and objective function value.
    """
    
    descent_circuits = []
    step_sizes = []
    x_current = x
    descent_direction, steepness = P.get_steepest_descent_circuit(x=x_current, c=c)
    
    
    while steepness != 0:
        alpha = P.get_max_step_size(x=x_current, g=descent_direction)
        descent_circuits.append(descent_direction)
        step_sizes.append(alpha)
        
        if math.isinf(alpha):
            return result(status=1, circuits=descent_circuits, steps=step_sizes)
        
        x_current = x_current + alpha*descent_direction
        descent_direction, steepness = P.get_steepest_descent_circuit(x=x_current, c=c)
        
    return result(status=0, x=x_current, circuits=descent_circuits, steps=step_sizes, c=c)
        

class result:
    def __init__(self, status, x=None, circuits=None, steps=None, c=None):
        self.status = status
        self.x = x
        self.circuits = circuits
        self.steps = steps
        
        if self.status == 0:
            self.augmentations =  len(steps)
            self.obj = numpy.dot(c, x)
            
    def __str__(self):
        if self.status == 1:
            return ('Problem is unbounded.'
                        + '\nSteepest descent unbounded circuit: ' + str(self.circuits[-1].T)
                    )
        elif self.status == 0:
            return ('Optimal solution is x = ' + str(self.x.T) 
                        + '\nOptimal objective: ' + str(self.obj)
                        + '\nNumber of iterations: ' + str(len(self.circuits))
                    )

            