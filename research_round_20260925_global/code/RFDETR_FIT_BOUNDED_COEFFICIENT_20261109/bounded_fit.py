"""Fit one-step and iterative mask score changes under one fixed trust budget."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import brentq
from scipy.special import expit

PENALTY = .002
ITERATIONS = 160


@dataclass
class Fit:
    one_step: np.ndarray
    iterative: np.ndarray
    start_loss: float
    one_loss: float
    final_loss: float
    projected_gradient: float
    backtracks: int
    one_seconds: float
    iterative_extra_seconds: float


def project_ball(x):
    return x / max(1., float(np.linalg.norm(x)))


def quadratic_ball(hessian, gradient):
    """Solve the positive semidefinite quadratic on the Euclidean unit ball."""
    s, v = np.linalg.eigh(hessian)
    if s.min() < -1e-10:
        raise ValueError("Trust Hessian must be positive semidefinite")
    s = np.maximum(s, 1e-12)
    g = v.T @ gradient
    raw = g/s
    if np.linalg.norm(raw) <= 1.:
        return -v @ raw
    upper = float(np.linalg.norm(g))
    shift = brentq(lambda mu: np.linalg.norm(g/(s+mu))-1., 0., upper, xtol=1e-13)
    return project_ball(-v @ (g/(s+shift)))


def solve_bounded(design, metric, native, labels, weights):
    """Optimize weighted proximal BCE in a fixed ellipsoid."""
    started = perf_counter()
    lower = np.linalg.cholesky(metric)
    # If theta = L^T delta, A delta = (A L^-T) theta.
    a = solve_triangular(lower, design.T, lower=True).T.copy()

    def loss_and_grad(theta):
        change = a @ theta
        z = native + change
        loss = weights @ (np.logaddexp(0.,z)-labels*z+PENALTY/2*change**2)
        residual = weights*(expit(z)-labels+PENALTY*change)
        return float(loss), a.T @ residual

    zero = np.zeros(design.shape[1])
    start, gradient = loss_and_grad(zero)
    p = expit(native)
    hessian = a.T @ ((weights*(p*(1-p)+PENALTY))[:,None]*a)
    one = quadratic_ball(hessian,gradient)
    backtracks = 0
    one_loss, _ = loss_and_grad(one)
    while one_loss > start and backtracks < 8:
        one *= .5
        one_loss, _ = loss_and_grad(one)
        backtracks += 1
    if one_loss > start:
        one, one_loss = zero, start

    one_finished = perf_counter()
    gram = a.T @ (weights[:,None]*a)
    lipschitz = max(1e-12, (.25+PENALTY)*np.linalg.eigvalsh(gram)[-1])
    theta = one.copy()
    final, grad = loss_and_grad(theta)
    for _ in range(ITERATIONS):
        theta = project_ball(theta-grad/lipschitz)
        final, grad = loss_and_grad(theta)
    stationarity = float(np.linalg.norm(theta-project_ball(theta-grad/lipschitz))*lipschitz)
    if not np.isfinite(final) or final > one_loss+1e-9:
        raise ValueError("Bounded logistic iteration must preserve or improve one-step loss")
    back = lambda x: solve_triangular(lower.T,x,lower=False)
    return Fit(back(one),back(theta),start,one_loss,final,stationarity,backtracks,
               one_finished-started,perf_counter()-one_finished)
