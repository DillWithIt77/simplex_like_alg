import numpy as np

def convert_to_standard_form(c, A, b, bounds):
    """
    Convert a linear programming problem to standard form.

    Parameters:
    - c: Coefficients of the objective function.
    - A: Coefficient matrix for the constraints.
    - b: Right-hand side vector for the constraints.
    - bounds: List of tuples (lower_bound, upper_bound) for each variable.

    Returns:
    - Converted coefficients, constraint matrix, and right-hand side vector.
    """
    # Initialize new lists for constraints and objective
    A_new = []
    b_new = []

    # Convert inequalities to equalities
    for i in range(len(b)):
        if np.any(A[i, :] < 0):
            # Add slack/surplus variables
            A_new.append(np.hstack([A[i, :], np.eye(1)]))
            b_new.append(b[i])
        else:
            A_new.append(np.hstack([A[i, :], np.zeros(1)]))
            b_new.append(b[i])

    # Add constraints for non-negativity of variables
    for i, (lb, ub) in enumerate(bounds):
        if lb < 0:
            A_new.append(np.hstack([np.zeros(i), np.eye(1), np.zeros(len(bounds)-i-1)]))
            b_new.append(-lb)
        if ub < np.inf:
            A_new.append(np.hstack([np.zeros(i), np.eye(1), np.zeros(len(bounds)-i-1)]))
            b_new.append(ub)

    # Update objective coefficients
    c_new = np.hstack([c, np.zeros(len(b))])

    return np.array(c_new), np.array(A_new), np.array(b_new)
def simplex(c, A, b):
    """
    Solve the linear programming problem:
        maximize c^T * x
        subject to A * x <= b
        x >= 0

    Parameters:
    - c: Coefficients of the objective function (1D array).
    - A: Coefficient matrix for the constraints (2D array).
    - b: Right-hand side vector for the constraints (1D array).

    Returns:
    - Optimal solution vector and the optimal value of the objective function.
    """

    # Convert inputs to numpy arrays
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)

    # Number of variables and constraints
    num_vars = len(c)
    num_constraints = len(b)

    # Create the initial simplex tableau
    tableau = np.hstack([A, np.eye(num_constraints), b.reshape(-1, 1)])
    tableau = np.vstack([np.hstack([-c, np.zeros(num_constraints + 1)]), tableau])

    def pivot(tableau, row, col):
        """Perform pivot operation on the tableau."""
        tableau[row, :] /= tableau[row, col]
        for r in range(tableau.shape[0]):
            if r != row:
                tableau[r, :] -= tableau[r, col] * tableau[row, :]

    while True:
        # Check if all the coefficients in the objective row are non-negative
        if np.all(tableau[0, 1:-1] >= 0):
            break

        # Find the entering variable (most negative coefficient in the objective row)
        col = np.argmin(tableau[0, 1:-1]) + 1

        # Find the leaving variable (minimum ratio test)
        ratios = tableau[1:, -1] / tableau[1:, col]
        ratios[ratios <= 0] = np.inf
        row = np.argmin(ratios) + 1

        # Perform the pivot operation
        pivot(tableau, row, col)

    # Extract the solution
    solution = np.zeros(num_vars)
    for i in range(num_constraints):
        if np.count_nonzero(tableau[i + 1, :-1]) == 1:
            col = np.where(tableau[i + 1, :-1] == 1)[0][0]
            solution[col] = tableau[i + 1, -1]

    # Optimal value
    optimal_value = -tableau[0, -1]

    return solution, optimal_value