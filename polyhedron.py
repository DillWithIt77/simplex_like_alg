import numpy
import sympy
from scipy import optimize

import naive_algorithm
import polyhedral_model as pm

#constant epsilon
EPS = 10**-8

#class for representing a general polyhedron of the form:
# P = {x in R^n : Ax = b, Bx <= d}
class Polyhedron:
    
    #initiallize with matrices and vectors given by numpy arrays
    def __init__(self, B, d, A=None, b=None):
        self.B = B
        self.d = d
        self.A = A
        self.b = b
        
    
    #use naive algorithm to enumerate set of circuits C(A,B)
    def naive_circuit_enumeration(self):
        return naive_algorithm.enumerate_circuits(B_ineq=self.B, A_eq=self.A)
    
    
    #use polyhedral model to enumerate set of circuits C(A,B)
    def polyhedral_model_circuit_enumeration(self, sign=None):
        return pm.enumerate_circuits(B_ineq=self.B, A_eq=self.A, sign=sign)
    
    
    #use polyhedral model to enumerate subset of circuits that are 
    #sign-compatible with a given direction u with respect to B.
    def get_sign_compatible_circuits(self, u):
        sign = self.B.dot(u)
        return pm.enumerate_circuits(B_ineq=self.B, A_eq=self.A, sign=sign)
    
    
    #given an x in P, enumerates circuits of P which are strictly feasible at x
    def get_strictly_feasible_circuits(self, x):
        sign = self.get_feasibility_sign(x)
        return pm.enumerate_circuits(B_ineq=self.B, A_eq=self.A, sign=sign)
    
    
    #returns sign list associated with feasible directions at x in P
    def get_feasibility_sign(self, x):
        B_x = self.B.dot(x)
        m = self.B.shape[0]
        sign = []
        for i in range(m):
            if B_x[i] == self.d[i]:
                sign.append(-1)
            else:
                sign.append(None)
        return sign
    
    
    #enumerates circuits the related standard form polyhedron 
    #returns list of the actual circuits of P and list of its standard form circuits
    def get_standard_form_circuits(self):
        m_B, n = self.B.shape
        M = numpy.concatenate((self.B, numpy.eye(m_B, dtype=int)), axis=1)
        if self.A is not None:
            m_A = self.A.shape[0]
            M_A = numpy.concatenate((self.A, numpy.zeros((m_A, m_B), dtype=int)), axis=1)
            M = numpy.concatenate((M_A, M), axis=0)
        standard_circuits = naive_algorithm.enumerate_circuits( B_ineq=numpy.eye((n+m_B), dtype=int), A_eq = M)
           
        #post-processing to determine the actual ciruits of P
        circuits = []
        for y in standard_circuits:
            g = y[:n]
            B_g = self.B.dot(g)
            B_0 = numpy.zeros((1, n), dtype=int)
            for i in range(m_B):
                if B_g[i]==0:
                    B_0 = numpy.concatenate((B_0, self.B[i,:].reshape((1, n))))
            if self.A is not None:
                B_0 = numpy.concatenate((self.A, B_0))
            rank = numpy.linalg.matrix_rank(numpy.matrix(B_0))
            if rank == n - 1:
                circuits.append(g)
        
        return circuits, standard_circuits
    
    
    #given a point in P and a linear objective c, compuate a steepest descent circuit at x
    def get_steepest_descent_circuit(self, x, c):
        sign = self.get_feasibility_sign(x)
        return self.get_steepest_descent_sign_comp_circuit(sign=sign, c=c)
        
        
    #determine if a vector g is a circuit direction of P
    def is_circuit(self, g):
        m,n = self.B.shape
        B_g = self.B.dot(g)
        B_0 = numpy.zeros((1,n), dtype=int)
        for i in range(m):
            if abs(B_g[i]) <= EPS:
                B_0 = numpy.concatenate((B_0, self.B[i,:].reshape((1,n))))
        if self.A is not None:
                B_0 = numpy.concatenate((self.A, B_0))
        rank = numpy.linalg.matrix_rank(numpy.matrix(B_0))
        return (rank == n - 1)
    
    
    #return normalized circuit given a circuit direction of P
    def get_normalized_circuit(self, g):
        m,n = self.B.shape
        B_g = self.B.dot(g)
        B_0 = numpy.zeros((1,n), dtype=int)
        
        for i in range(m):
            if abs(B_g[i]) <= EPS:
                B_0 = numpy.concatenate((B_0, self.B[i,:].reshape((1,n))))
        if self.A is not None:
                B_0 = numpy.concatenate((self.A, B_0), axis=0)
                
        D = sympy.Matrix(B_0)
        ker_D = D.nullspace()
        if len(ker_D) != 1:
            raise ValueError('The direction ' + str(g.T)  +' is not a circuit of P')
        circuit = numpy.array(ker_D[0]).reshape(n)
        circuit= circuit*sympy.lcm([circuit[i].q for i in range(n) if circuit[i] != 0]) #normalize
        
        #make sure circuit has correct sign
        for i in range(n):
            if abs(g[i]) >= EPS:
                if circuit[i]*g[i] < 0:
                    circuit = -1*circuit
                break
        return circuit     
    
    
    #given a point x in P with feasible direction g, compute the maximum step size alpha
    def get_max_step_size(self, x, g):
        m,n = self.B.shape
        B_g = self.B.dot(g)     
        B_x = self.B.dot(x)
        alpha = float('inf')
        
        for i in range(m):
            if B_g[i] > 0:
                a = (self.d[i] - B_x[i])/float(B_g[i])
                if a <= alpha:
                    alpha = a 
        return alpha
            
    #given a u in ker(A) or two points in P, construct a sign-compatible sum of circuits.
    #if a linear function f(x) = c^T x is provided, returns an f-optimal sign-compatible sum
    #returns the list of circuits and corresponding positive weights lambdas for the sum
    def get_sign_compatible_sum(self, u=None, x_1=None, x_2=None, c=None):
        m,n = self.B.shape 
        if c is None:
            c = numpy.ones(n, dtype=int)
        if u is not None:
            w = u
        else:
            w = x_2 - x_1
            
        if self.A is not None:
            if any(self.A.dot(w) != 0):
                raise ValueError('The direction w or x_2 - x_1 must belong to ker(A).')
       
        circuits = []
        lambdas = []
        
        while not self.is_circuit(w):
            sign = self.B.dot(w)
            g = self.get_steepest_descent_sign_comp_circuit(sign=sign, c=c)[0]
            B_g = self.B.dot(g)
            B_w = self.B.dot(w)       
            lam = min([(B_w[i]/B_g[i]) for i in range(m) if B_w[i]*B_g[i] > 0])
            
            circuits.append(g)
            lambdas.append(lam)           
            w = w - lam*g
            
        g = self.get_normalized_circuit(w)
        circuits.append(g)
        for i in range(n):
            if g[i] > 0:
                lambdas.append(w[i]/g[i])
                break
        return circuits, lambdas
        
    
    #####Might also need to make edits here for simplex-like stuff##### 
    # things to add: r.h.s for (Bx_0)_i (maybe make serperate function to deteremine which i's are satisfied)  
     
    #returns steepest descent circuit with respect to c sign-compatible with sign    
    def get_steepest_descent_sign_comp_circuit(self, sign, c):
        m_B, n = self.B.shape
        y_vars = pm.determine_y_vars(m_B, sign=sign)
        n_y_vars = len(y_vars)
        
        #constraint matrix and r.h.s vector for linear program
        M = pm.build_augmented_matrix(self.B, y_vars, A=self.A)
        one_norm = numpy.concatenate((numpy.zeros(n, dtype=int),numpy.ones(n_y_vars, dtype=int))).reshape(1,n+n_y_vars)
        M = numpy.concatenate((M, one_norm))
        b_eq = numpy.zeros(M.shape[0], dtype=int)
        b_eq[-1]=1
        
        #upper and lower bounds for variables
        bounds = [(None, None)]*n
        for i in range(n_y_vars):
            bounds.append((0, None))
        
        #objective function
        obj = numpy.concatenate((c, numpy.zeros(n_y_vars, dtype=int)))
        
        #solve linear program
        result = optimize.linprog(c=obj, A_eq=M, b_eq=b_eq, bounds=bounds, method='simplex')
        #print(result)
        if result.status == 2:
            raise ValueError('Unable to find feasible circuit direction')
        g = result.x[:n]
        steepness = result.fun
        
        if steepness == 0:
            return numpy.zeros(n), 0
        
        if self.is_circuit(g):
            #normalize to coprime integer components
            circuit = self.get_normalized_circuit(g)       
            return circuit, steepness     
        
        #if the linear program does not return a vertex solution, optimize over optimal face until a vertex solution is found
        #there are issues with scipy's optimize.linprog simplex solver when A_eq is not full row rank,
        #so this should only be used for toy examples
        M = numpy.concatenate((M, obj.reshape((1,n + n_y_vars))))
        b_eq = numpy.append(b_eq, steepness)

        count = 1
        while not self.is_circuit(g):
            if count > 10:
                raise ValueError('Failed to find vertex solution to linear program') 
            
            rand_obj = numpy.random.randint(low=0,high=100,size=n)
            rand_obj = numpy.concatenate((rand_obj, numpy.zeros(n_y_vars,dtype=int)))
            
            result = optimize.linprog(c=rand_obj, A_eq=M, b_eq=b_eq, bounds=bounds, method='interior-point')
            #print(result)
            g = result.x[:n]
            
            if self.is_circuit(g):
                circuit = self.get_normalized_circuit(g) 
                return circuit, steepness 
            count += 1