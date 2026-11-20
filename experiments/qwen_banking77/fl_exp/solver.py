"""Small dense convex QPs for client weighting (NumPy, CPU, no QP package).

The feasible set is a bounded simplex, optionally with an extra box variable
and one halfspace. A feasible active-set method solves at most 11 variables;
diagnostics report feasibility and stationarity in a scaled objective.
"""

import numpy as np


def solve_client_qp(gram, linear=None, simplex_dim=None, lower=None, upper=None,
                    halfspace=None, tol=1e-8, max_iter=1000):
    """Minimize .5*x@gram@x+linear@x; sum(x[:simplex_dim])=1.

    Bounds are componentwise. ``halfspace`` means halfspace @ x >= 0.
    Raise on infeasibility or failure rather than returning unchecked weights.
    A 1e-12 relative diagonal stabilizer handles singular gradient Gram matrices.
    """
    matrix = np.asarray(gram, dtype=np.float64)
    dimension = matrix.shape[0]
    n = simplex_dim if simplex_dim is not None else dimension
    if matrix.shape != (dimension, dimension) or not 1 <= n <= dimension:
        raise ValueError("Invalid QP dimensions")
    vector = np.zeros(dimension) if linear is None else np.asarray(linear, dtype=np.float64)
    lo = np.zeros(dimension) if lower is None else np.asarray(lower, dtype=np.float64)
    hi = np.ones(dimension) if upper is None else np.asarray(upper, dtype=np.float64)
    if vector.shape != (dimension,) or lo.shape != (dimension,) or hi.shape != (dimension,):
        raise ValueError("Invalid QP vector/bound dimensions")
    if (not np.isfinite(matrix).all() or not np.isfinite(vector).all()
            or np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo > hi)):
        raise ValueError("Nonfinite QP or invalid bounds")
    if tol <= 0 or max_iter < 1:
        raise ValueError("QP tolerance and iteration limit must be positive")
    equality = np.zeros((1, dimension))
    equality[0, :n] = 1.0
    x = np.zeros(dimension)
    x[:n] = 1.0 / n
    x[n:] = lo[n:]
    if np.any(x < lo - tol) or np.any(x > hi + tol):
        raise ValueError("The equal-client initial point must satisfy QP bounds")
    constraints = [*np.eye(dimension), *(-np.eye(dimension))]
    limits = [*lo, *(-hi)]
    if halfspace is not None:
        row = np.asarray(halfspace, dtype=np.float64)
        if row.shape != (dimension,) or not np.isfinite(row).all():
            raise ValueError("Invalid fairness halfspace")
        norm = np.max(np.abs(row))
        if norm > 1e-14:
            row = row / norm
            if row @ x < -tol:
                # FairMOO uses the unrestricted simplex, so a vertex provides
                # a feasible start whenever a convex combination can do so.
                best = int(np.argmax(row[:n]))
                if row[best] < -tol or np.any(lo[:n] > 0) or np.any(hi[:n] < 1):
                    raise RuntimeError("FairMOO fairness-preserving simplex is infeasible")
                x[:n] = 0.0
                x[best] = 1.0
            constraints.append(row)
            limits.append(0.0)
    c = np.asarray(constraints)
    b = np.asarray(limits)
    finite = np.isfinite(b)
    c, b = c[finite], b[finite]
    matrix = (matrix + matrix.T) / 2
    scale = max(float(np.linalg.norm(matrix, ord=np.inf)), float(np.max(np.abs(vector))), 1e-30)
    p, q = matrix / scale, vector / scale
    if np.linalg.eigvalsh(p).min() < -1e-8:
        raise ValueError("QP Gram matrix is not positive semidefinite")
    stabilized = p + np.eye(dimension) * 1e-12
    active = []

    def independent(index):
        rows = np.vstack([equality, c[active]]) if active else equality
        return np.linalg.matrix_rank(np.vstack([rows, c[index]]), tol=1e-10) > len(rows)

    for index in np.flatnonzero(np.abs(c @ x - b) <= tol):
        if independent(index):
            active.append(int(index))
    residual = float("inf")
    for iteration in range(1, max_iter + 1):
        working = np.vstack([equality, c[active]]) if active else equality
        gradient = p @ x + q
        kkt = np.block([[stabilized, -working.T],
                        [working, np.zeros((len(working), len(working)))]] )
        rhs = np.r_[-gradient, np.zeros(len(working))]
        solution = np.linalg.lstsq(kkt, rhs, rcond=1e-14)[0]
        step, multipliers = solution[:dimension], solution[dimension:]
        residual = float(np.max(np.abs(gradient - working.T @ multipliers)))
        if np.linalg.norm(step, ord=np.inf) <= tol or residual <= tol:
            if active and float(np.min(multipliers[1:])) < -tol:
                active.pop(int(np.argmin(multipliers[1:])))
                continue
            if residual <= tol * 10:
                break
        movement = c @ step
        slack = c @ x - b
        candidates = [index for index in range(len(c)) if index not in active and movement[index] < -tol]
        alpha, blocking = 1.0, None
        for index in candidates:
            bound = max(0.0, slack[index]) / -movement[index]
            if bound < alpha:
                alpha, blocking = bound, index
        x += alpha * step
        if blocking is not None and independent(blocking):
            active.append(blocking)
    else:
        raise RuntimeError("Client QP failed to converge after {} iterations (residual={:.3g})".format(max_iter, residual))
    violation = max(float(np.max(b - c @ x)), abs(float((equality @ x).item()) - 1.0), 0.0)
    if violation > tol * 10 or not np.isfinite(x).all():
        raise RuntimeError("Client QP returned infeasible weights")
    return x, {"qp_iterations": iteration, "qp_kkt_residual": residual,
               "qp_primal_violation": violation, "qp_objective": float(0.5 * x @ matrix @ x + vector @ x),
               "qp_objective_scale": scale}
