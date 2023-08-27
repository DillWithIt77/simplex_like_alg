import numpy
import sympy
import cdd


#Given matrices A and B, computes set of circuits C(A,B) by enumerating the extreme rays
#  of the corresponding polyhedral model 
#Input: A_eq and B_ineq are (m_a x n) and (m x n) numpy arrays;
#       sign is m-dimensional vector or list which gives desired orthant of Bg if any
#Output: list of circuits in C(A,B) given by n-dimensional numpy arrays
def enumerate_circuits(B_ineq, A_eq=None, sign=None):
    A = A_eq
    B = B_ineq
    m_B,n = B.shape
    m_a = 0
    if A is not None:
        m_a = A.shape[0]
    
    y_vars = determine_y_vars(m_B, sign=sign)
    n_y_vars = len(y_vars)
    
    #####This is where I will need to edit to add in extra constraints mentioned in Steffen's Grant####
    #####figure out how this relates to the sign_comp_steep circs in polyhedron file######
    
    #build constraint matrix M for conic polyhedral model Mr >= 0,
    #where first column of M is the r.h.s. vector 0.
    M = build_augmented_matrix(B, y_vars, A=A)
    M = numpy.concatenate((M, -1*M))
    I_y = numpy.concatenate((numpy.zeros((n_y_vars, n)), numpy.eye(n_y_vars)), axis=1).astype(int)
    M = numpy.concatenate((M, I_y))
    M = numpy.concatenate((numpy.zeros((2*m_a + 2*m_B + n_y_vars, 1), dtype=int), M), axis=1)
    
    #use cdd to enumerate extreme rays of the cone <- might be able to replace this with vertex enumeration things
    mat = cdd.Matrix(M, number_type='fraction')
    mat.rep_type = cdd.RepType.INEQUALITY
    poly = cdd.Polyhedron(mat)
    #print(poly)
    rays = numpy.array(poly.get_generators())
    #print(rays)

    #obtain circuits from extreme rays
    circuits = []
    num_rays = rays.shape[0]
    for i in range(num_rays):
        g = rays[i,1:n+1]
        if not numpy.array_equal(g, numpy.zeros(n)):
            g = g*sympy.lcm([g[j].denominator for j in range(n) if g[j] != 0]) #normalize
            circuits.append(g)  
    return circuits


#returns list of y-variables for polyhedral model associated with given sign list.
#each y-variable is described by a tuple (i, 1 or -1) which indicates which row i
#of B the y-variable corresponds to and whether the variable is the positive or negative
#part of (Bx)_i
def determine_y_vars(m,sign=None):
    y_vars = []
    if sign is None:
       for i in range(m):
           y_vars.append([i, 1])
           y_vars.append([i, -1])
    else:
        for i in range(m):
            s = sign[i]
            if s is None:
                y_vars.append([i, 1])
                y_vars.append([i, -1])
            elif s > 0:
                y_vars.append([i, 1])
            elif s < 0:
                y_vars.append([i, -1])
    return y_vars


#returns augmented equality matrix for polyhedral model
def build_augmented_matrix(B, y_vars, A=None):
    m_B, n = B.shape
    n_y_vars = len(y_vars)
    
    M = numpy.concatenate((B, numpy.zeros((m_B, n_y_vars), dtype=int)), axis=1)
    for j in range(n_y_vars):
        i = y_vars[j][0]
        M[i][n+j] = -1*y_vars[j][1]
    
    if A is not None:
        m_A = A.shape[0]
        A_aug = numpy.concatenate((A, numpy.zeros((m_A, n_y_vars), dtype=int)), axis=1)
        M = numpy.concatenate((A_aug, M), axis=0)
        
    return M
    
    
    
        

