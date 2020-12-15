import numpy as np
from scipy.optimize import newton as newton_from_scipy


def cubic_roots(a, b, c, d, complex_roots=False):

    n_eq = a.shape[0]
    roots = np.zeros((n_eq, 3), dtype=np.complex)

    d0 = b ** 2 - 3 * a * c
    d1 = 2 * b ** 3 - 9 * a * b * c + 27 * a ** 2 * d

    tmp = np.sqrt((d1 ** 2 - 4 * d0 ** 3).astype(np.complex))

    C3 = d1 + tmp
    # tmp can be added or subtracted, but the sum should not be zero
    I = np.abs(C3) < 1e-10
    C3[I] = d1[I] - tmp[I]

    C = (C3 / 2.0) ** (1 / 3.0)

    # find the other roots
    rot = (-1 + 1j * np.sqrt(3)) / 2
    rot = np.array([[1.0, rot, np.conj(rot)]])
    CC = C[:, None] * rot
    roots = -(b[:, None] + CC + d0[:, None] / CC) / (3 * a[:, None])

    return roots


def quartic_roots(a, b, c, d, e):

    n = a.shape[0]

    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    d = np.array(d)
    e = np.array(e)

    d0 = c ** 2 - 3 * b * d + 12 * a * e
    d1 = 2 * c ** 3 - 9 * b * c * d + 27 * b ** 2 * e + 27 * a * d ** 2 - 72 * a * c * e

    p = (8 * a * c - 3 * b ** 2) / (8 * a ** 2)
    q = (b ** 3 - 4 * a * b * c + 8 * a ** 2 * d) / (8 * a ** 3)

    r = np.sqrt((d1 ** 2 - 4 * d0 ** 3).astype(np.complex))
    u = d1 + r
    # r can be added or subtracted, but the sum should not be zero
    I = np.abs(u) < 1e-10
    u[I] = d1[I] - r[I]
    Q = (0.5 * u) ** (1.0 / 3.0)

    S = 0.5 * np.sqrt(-(2.0 / 3.0) * p + (Q + d0 / Q) / (3 * a))

    roots = np.zeros((n, 4), dtype=np.complex)
    k1 = -b / (4 * a)
    k2 = -4 * S ** 2 - 2 * p

    m1 = 0.5 * np.sqrt(k2 + q / S)
    roots[:, 0] = k1 - S + m1
    roots[:, 1] = k1 - S - m1

    m1 = 0.5 * np.sqrt(k2 - q / S)
    roots[:, 2] = k1 + S + m1
    roots[:, 3] = k1 + S - m1

    return roots


def f(lambda_, phi, v, z, df_needed=False):
    va = np.abs(v) ** 2
    f = (
        np.sum((va / phi) * (lambda_[:, None] / (lambda_[:, None] - phi)) ** 2, axis=1)
        - lambda_
        + z
    )

    if df_needed:
        df = -2 * lambda_ * np.sum(va / (lambda_[:, None] - phi) ** 3, axis=1) - 1
        return f, df
    else:
        return f


def f_prime(lambda_, phi, v, z):
    va = np.abs(v) ** 2
    return -2 * lambda_ * np.sum(va / (lambda_[:, None] - phi) ** 3, axis=1) - 1


def f_second(lambda_, phi, v, z):
    va = np.abs(v) ** 2
    return 2 * np.sum(
        va * (2 * lambda_[:, None] + phi) / (lambda_[:, None] - phi) ** 4, axis=1
    )


def g_obj(lambda_, phi, v, z):
    va = np.abs(v) ** 2
    o1 = -np.sum(va / (phi * (lambda_[:, None] - phi)), axis=1)
    o2 = -np.log(lambda_) + 1 - z / lambda_
    return o1 + o2


def show_constraint(ln, phi, v, z):

    import matplotlib.pyplot as plt

    lambdas = np.linspace(1e-5, 2 * ln, 10000,)
    f_val = f(lambdas, phi, v, z)
    obj_val = g_obj(lambdas, phi, v, z)

    plt.figure()
    plt.plot(lambdas, f_val, label="constraint $f(\lambda)$")
    plt.plot(lambdas, obj_val, "orange", label="objective $h(\lambda)$")
    plt.plot(lambdas, np.zeros_like(lambdas), "--", label="zero")

    plt.legend()
    plt.xlabel("$\lambda$")
    plt.ylabel("$f(\lambda)$")
    plt.ylim([f_val[-1], f_val[0] + 2 * (f_val[0] - f_val[-1])])
    plt.title(f"Objective and constraint, $\lambda^*={ln}$")

    print(f"phi={phi} v={v} z={z}")

    plt.show()


def init_newton(phi, v, z):

    n = phi.shape[0]

    va = np.abs(v[:, -1]) ** 2
    phi_max = phi[:, -1]

    a = -np.ones(n)
    b = va / phi_max + 2 * phi_max + z
    c = -phi_max * (phi_max + 2 * z)
    d = phi_max ** 2 * z

    roots = cubic_roots(a, b, c, d)

    I = np.abs(np.imag(roots)) < 1e-10
    candidates = np.real(roots * I)
    candidates[candidates - phi_max[:, None] <= 0] = np.inf

    candidates = np.min(candidates, axis=1)

    I_inf = np.isinf(candidates)
    candidates[I_inf] = phi_max[I_inf] + 1e-4

    return candidates


def newton_s(lambda_, phi, v, z, max_iter=100, atol=1e-7, verbose=False):
    def obj(x):
        return f(x, phi, v, z)

    def deriv(x):
        return f_prime(x, phi, v, z)

    def deriv2(x):
        return f_second(x, phi, v, z)

    root, r, converged = newton_from_scipy(
        obj,
        lambda_,
        fprime=deriv,
        # fprime2=deriv2,
        tol=atol,
        maxiter=max_iter,
        full_output=True,
    )

    return root


def newton(
    lambda_,
    phi,
    v,
    z,
    lower_bound=None,
    max_iter=100,
    atol=1e-7,
    verbose=False,
    plot=False,
    return_iter=False,
):
    """
    Parameters
    ----------
    lambda_: ndarray (n_problems,)
        The initial value
    phi: ndarray (n_problems, n_dim)
        The eigenvalues (> 0)
    v: ndarray (n_problems, n_dim)
        The constant vector
    z: ndarray (n_problems)
        The constant offset
    max_iter: int
        The maximum number of iterations
    atol: float
        The tolerance for the final function value
    verbose: bool
        Print progress of algorithm if true
    """

    f_val, df_val = f(lambda_, phi, v, z, df_needed=True)
    I = np.where(np.abs(f_val) > atol)[0]
    f_val = f_val[I]
    df_val = df_val[I]

    for epoch in range(max_iter):

        new_lambda = lambda_[I] - f_val / df_val

        if lower_bound is not None:

            violate = new_lambda <= lower_bound[I]
            new_lambda[violate] = (
                0.5 * lower_bound[I[violate]] + 0.5 * lambda_[I[violate]]
            )

        lambda_[I] = new_lambda

        f_val, df_val = f(lambda_[I], phi[I, :], v[I, :], z[I], df_needed=True)
        left = np.abs(f_val) > atol

        # if no value is larger than the tolerance, stop
        if np.sum(left) == 0:
            break

        I = I[left]
        f_val = f_val[left]
        df_val = df_val[left]

    if verbose:
        f_val = f(lambda_, phi, v, z, df_needed=False)
        root_max_val = np.max(np.abs(f_val))
        print(f"newton: epochs={epoch} tol={root_max_val}")

    if plot:
        imax = I[np.argmax(np.abs(f_val))]
        show_constraint(lambda_[imax], phi[imax, :], v[imax, :], z[imax])

    if return_iter:
        return lambda_, epoch, f_val
    else:
        return lambda_


def bisection(lo, hi, phi, v, z, max_iter=100, atol=1e-7, verbose=False, return_iter=False):
    def obj(x, I=None):
        if I is not None:
            x = x[I]
            pphi = phi[I, :]
            vv = v[I, :]
            zz = z[I]
        else:
            pphi = phi
            vv = v
            zz = z
        return f(x, pphi, vv, zz, df_needed=False)

    lo_bound = np.max(phi, axis=1)

    # lo = np.max(phi, axis=1) + 1.0
    # hi = np.max(phi, axis=1) + 10.0

    # lo = init[:, 0]
    # hi = init[:, 1]

    lo_val = obj(lo)
    hi_val = obj(hi)

    for epoch in range(max_iter):

        mid = (lo + hi) / 2.0
        mid_val = obj(mid)

        # distance from zero
        eps = np.max(np.abs(mid_val))

        if eps < atol:
            break

        # upper bound is not far enough
        I = hi_val > 0
        hi[I] *= 2.0
        hi_val[I] = obj(hi, I)

        # lower bound is too far
        I = lo_val < 0
        lo[I] = 0.5 * (lo_bound[I] + lo[I])
        lo_val[I] = obj(lo, I)

        I_others = np.logical_and(lo_val >= 0, hi_val <= 0)

        # middle point is too far
        I = np.logical_and(I_others, mid_val < 0)
        hi[I] = mid[I]
        hi_val[I] = mid_val[I]

        # middle point is not far enough
        I = np.logical_and(I_others, mid_val >= 0)
        lo[I] = mid[I]
        lo_val[I] = mid_val[I]

    if verbose:
        print(f"bisection: epochs={epoch} tol={np.max(np.abs(mid_val))}")

    if return_iter:
        return mid, epoch
    else:
        return mid
