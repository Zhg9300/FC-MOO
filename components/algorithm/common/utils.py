import torch
import numpy as np
import cvxopt
from cvxopt import matrix
import random
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['abstol'] = 1e-3
cvxopt.solvers.options['reltol'] = 1e-3
cvxopt.solvers.options['feastol'] = 1e-3
cvxopt.solvers.options['maxiters'] = 50


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):

    P = P.astype(np.double)
    q = q.astype(np.double)
    args = [matrix(P), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    optimal_flag = 1
    if 'optimal' not in sol['status']:
        optimal_flag = 0
    return np.array(sol['x']).reshape((P.shape[1],)), optimal_flag


def setup_qp_and_solve(vec):
    P = np.dot(vec, vec.T)
    n = P.shape[0]
    q = np.zeros(n)

    G = - np.eye(n)
    h = np.zeros(n)

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False

    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def quadprog(P, q, G, h, A, b):
    P = cvxopt.matrix(P.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(P, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])

def setup_qp_and_solve_for_mgdaplus_1(vec, epsilon, lambda0):

    P = np.dot(vec, vec.T)

    n = P.shape[0]

    q = np.array([[0] for i in range(n)])

    A = np.ones(n).T
    b = np.array([1])

    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    sol = quadprog(P, q, G, h, A, b).reshape(-1)

    return sol, 1


def setup_qp_and_solve_for_FairMOO(grads, g_fair):

    P = np.dot(grads, grads.T)
    n = P.shape[0]
    q = np.zeros(n)
    G = np.zeros((n+1, n))
    h = np.zeros(n+1)
    for i in range(n):
        G[i][i] = -1
    G[n] = -grads @ g_fair

    A = np.ones((1, n))
    b = np.ones(1)

    cvxopt.solvers.options['show_progress'] = False
    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, A, b)
    return sol, optimal_flag


def setup_qp_and_solve_for_fc_moo(task_grads, g_fair, phi, mu_cap):
    """
    QP form of deterministic FC-MOO:

        min_x 1/2 x^T P x + q^T x
        s.t.  Gx <= h
              Ax  = b

    where x = [lambda_1, ..., lambda_n, mu]^T.
    """
    task_grads = task_grads.detach().cpu().numpy()      # [n, d]
    g_fair = g_fair.detach().cpu().numpy()              # [d]

    if torch.is_tensor(phi):
        phi = float(phi.detach().cpu().item())
    else:
        phi = float(phi)

    mu_cap = float(mu_cap)

    n = task_grads.shape[0]

    # A_mat rows are [g_1; ...; g_n; g_fair]
    A_mat = np.vstack([task_grads, g_fair[None, :]])    # shape [n+1, d]

    # Objective:
    # 1/2 || A_mat^T x ||^2 - phi * mu
    # = 1/2 x^T (A_mat A_mat^T) x + q^T x
    P = A_mat @ A_mat.T                                  # shape [n+1, n+1]
    P = 0.5 * (P + P.T)                                  # numerical symmetrization

    q = np.zeros(n + 1, dtype=np.float64)
    q[-1] = -phi

    # Inequality constraints:
    # lambda_i >= 0      -> -lambda_i <= 0
    # mu >= 0            -> -mu <= 0
    # mu <= mu_cap       ->  mu <= mu_cap
    G = np.zeros((n + 2, n + 1), dtype=np.float64)
    h = np.zeros(n + 2, dtype=np.float64)

    # -lambda_i <= 0
    for i in range(n):
        G[i, i] = -1.0

    # -mu <= 0
    G[n, n] = -1.0
    h[n] = 0.0

    # mu <= mu_cap
    G[n + 1, n] = 1.0
    h[n + 1] = mu_cap

    # Equality constraint:
    # sum_i lambda_i = 1
    Aeq = np.zeros((1, n + 1), dtype=np.float64)
    Aeq[0, :n] = 1.0
    beq = np.array([1.0], dtype=np.float64)

    sol, optimal_flag = cvxopt_solve_qp(P, q, G, h, Aeq, beq)
    return sol, optimal_flag

def get_d_FairMOO(grads, device, g_fair):
    sol, optimal_flag = setup_qp_and_solve_for_FairMOO(grads.cpu().detach().numpy(), g_fair.cpu().detach().numpy())
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ grads
    return d, sol

def get_d_fc_moo(task_grads, g_fair, device, phi, mu_cap):
    sol, optimal_flag = setup_qp_and_solve_for_fc_moo(
        task_grads=task_grads,
        g_fair=g_fair,
        phi=phi,
        mu_cap=mu_cap,
    )

    if sol is None or (optimal_flag is not None and not optimal_flag):
        raise RuntimeError("QP solver failed in get_d_fc_moo.")

    sol = torch.from_numpy(sol).float().to(device)

    n = task_grads.shape[0]
    lam = sol[:n]                 # lambda
    mu = sol[n]                   # mu

    d = lam @ task_grads + mu * g_fair
    return d, sol

def get_d_mgdaplus_d(grads, device, epsilon, lambda0):

    vec = grads
    sol, optimal_flag = setup_qp_and_solve_for_mgdaplus_1(
        vec.cpu().detach().numpy(), epsilon, lambda0)

    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ grads

    descent_flag = 1
    c = -(grads @ d)
    if not torch.all(c <= 1e-6):
        descent_flag = 0

    return d, sol, descent_flag


def get_fedmdfg_d(grads, value, add_grads, alpha, fair_guidance_vec, force_active, device):
    fair_grad = None
    value_norm = torch.norm(value)
    norm_values = value / value_norm
    fair_guidance_vec /= torch.norm(fair_guidance_vec)
    cos = float(norm_values @ fair_guidance_vec)
    cos = min(1, cos)
    cos = max(-1, cos)
    bias = np.arccos(cos) / np.pi * 180
    pref_active_flag = (bias > alpha) | force_active
    norm_vec = torch.norm(grads, dim=1)
    indices = list(range(len(norm_vec)))
    grads = norm_vec[indices].reshape(-1, 1) * grads / (norm_vec + 1e-6).reshape(-1, 1)
    if not pref_active_flag:
        vec = grads
        pref_active_flag = 0
    else:
        pref_active_flag = 1
        h_vec = (fair_guidance_vec @ norm_values * norm_values - fair_guidance_vec).reshape(1, -1)
        h_vec /= torch.norm(h_vec)
        fair_grad = h_vec @ grads
        vec = torch.cat((grads, fair_grad))
    if add_grads is not None:
        norm_vec = torch.norm(add_grads, dim=1)
        indices = list(range(len(norm_vec)))
        random.shuffle(indices)
        add_grads = norm_vec[indices].reshape(-1, 1) * add_grads / (norm_vec + 1e-6).reshape(-1, 1)
        vec = torch.vstack([vec, add_grads])
    sol, _ = setup_qp_and_solve(vec.cpu().detach().numpy())
    sol = torch.from_numpy(sol).float().to(device)
    d = sol @ vec
    return d, vec, pref_active_flag, fair_grad

def Gram_Schmidt(grads, loss, pow):
    num_vectors, dim = grads.shape
    orthogonal = torch.zeros_like(grads, dtype=torch.float64)

    for i in range(num_vectors):
        vec = grads[i].double().clone()
        numerator = grads[i].double().clone()
        denominator = loss[i] ** pow
        for j in range(i):
            numerator -= torch.dot(orthogonal[j], vec) / torch.dot(orthogonal[j], orthogonal[j]) * orthogonal[j]
            denominator -= torch.dot(orthogonal[j], vec) / torch.dot(orthogonal[j], orthogonal[j])
        orthogonal[i] = numerator / denominator
    return orthogonal.to(grads.dtype)

def get_d_adafed(grads):
    total = 0
    for grad in grads:
        total += 1.0 / torch.norm(grad) ** 2
    d = torch.zeros_like(grads[0])
    for grad in grads:
        lamb = 1.0 / (torch.norm(grad) ** 2 * total)
        d += lamb * grad
    return d
