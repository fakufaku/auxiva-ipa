# Copyright 2020 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This script generates figures 1 and 2 from the paper
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from bss.newton_root import init_newton, newton
from bss.random import crandn
from bss.utils import tensor_H


def rand_psd(*shape, dtype=np.complex):
    shape = list(shape) + [2 * shape[-1]]
    X = np.random.randn(*shape)
    return X @ tensor_H(X) / shape[-1]


def compute_quad_form(X, Y, A, b):
    X = X - b[0]
    Y = Y - b[1]
    return A[0, 0] * X ** 2 + (A[0, 1] + A[1, 0]) * X * Y + A[1, 1] * Y ** 2


def objective(X, Y, A, b, C, d, z):

    t1 = compute_quad_form(X, Y, A, b)
    t2 = compute_quad_form(X, Y, C, d) + z

    return -np.log(t2) + t1


def lqpqm_solver(A, b, C, d, z):

    G_H = np.linalg.cholesky(A)

    U = np.linalg.solve(G_H, tensor_H(np.linalg.solve(G_H, C)))
    v = np.linalg.solve(G_H, C @ (b - d))

    phi, Sigma = np.linalg.eigh(U)

    v_tilde = tensor_H(Sigma) @ v

    lambda_0 = init_newton(phi[None, :], v_tilde[None, :], np.array([z]))

    lambda_ = newton(
        lambda_0,
        phi[None, :],
        v_tilde[None, :],
        np.array([z]),
        lower_bound=phi[None, -1],
        max_iter=100,
        verbose=True,
    )
    lambda_ = lambda_[0]

    y = np.linalg.solve(lambda_ * np.eye(A.shape[0]) - U, v)
    x = np.linalg.solve(tensor_H(G_H), y) + b

    return x, lambda_, phi, v_tilde


def f_test(lambda_, phi, v, z):
    va = np.abs(v) ** 2
    x1 = -np.log(lambda_)
    x2 = np.sum(va / (phi * (lambda_[:, None] - phi)), axis=1)
    return 1 + x1 - x2 - z / lambda_


def f_obj(lambda_, phi, v, z):
    va = np.abs(v) ** 2
    f = np.sum(va / (lambda_[:, None] - phi) ** 2, axis=1)
    return f - np.log(lambda_)


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


def show_constraint(
    ln, phi, v, z, show_one_out=False, show_init=False, figsize=None, save=None
):

    n_colors = 2
    if show_one_out:
        n_colors += 1
    if show_init:
        n_colors += 1

    sns.set_palette("viridis", n_colors=n_colors)

    lambdas = np.linspace(1e-5, 1.3 * ln[0], 50000,)
    f_val = f(lambdas, phi, v, z)
    obj_val = f_obj(lambdas, phi, v, z)
    test_val = f_test(lambdas, phi, v, z)

    cm = 0.39  # 1 cm in inches
    plt.figure(figsize=(8 * cm, 6.0 * cm))

    # init
    if show_init:
        va = np.abs(v[0, -1]) ** 2
        phi_max = phi[0, -1]
        a = -1
        b = va / phi_max + 2 * phi_max + z
        c = -phi_max * (phi_max + 2 * z)
        d = phi_max ** 2 * z
        poly = a * lambdas ** 3 + b * lambdas ** 2 + c * lambdas + d
        plt.plot(lambdas, poly, label="Cubic approx. of $f(\lambda)$")

    plt.plot(lambdas, test_val, label="$g(\lambda)$ (objective)")
    plt.plot(lambdas, f_val, label="$f(\lambda)$ (constraint)")

    # Highlight the location of the solution
    ax = plt.gca()
    opt_loc = matplotlib.patches.Ellipse(
        (ln[0], 0), 1.5 * 0.4, 1.5 * 1.5 * 1.1, facecolor="None", edgecolor="k"
    )
    ax.add_artist(opt_loc)
    ax.annotate(
        "$\lambda^\star$",
        xy=(ln[0] + 0.4, 1.1),
        # xycoords="data",
        xytext=(ln[0] + 3.4, 7.1),
        # textcoords="offset points",
        arrowprops=dict(facecolor="black", edgecolor="black", arrowstyle="->"),
        horizontalalignment="right",
        verticalalignment="bottom",
    )

    if show_one_out:
        o = list(range(phi.shape[1]))
        o.remove(phi.shape[1] // 2 + 1)
        one_out_val = f_test(lambdas, phi[:, o], v[:, o], z)
        one_out_con = f(lambdas, phi[:, o], v[:, o], z)

        plt.plot(lambdas, one_out_val, "b", label="one out obj")
        plt.plot(lambdas, one_out_con, "r", label="one out con")

    ticks = np.concatenate(([0], phi[0], [ln[0]]))
    labels = (
        [0] + [f"$\\varphi_{i+1}$" for i in range(len(phi[0]))] + ["$\lambda^\star$"]
    )

    plt.xticks(ticks, labels)
    plt.yticks([0], [0])

    handles, labels = ax.get_legend_handles_labels()
    leg = plt.legend(
        handles[::-1], labels[::-1], bbox_to_anchor=(0.0, 1.1), loc=2, borderaxespad=0.0
    )
    # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
    # leg = plt.legend(loc=0)
    plt.xlabel("$\lambda$")
    plt.ylim([f_val[-1], f_val[0] + 2.0 * (f_val[0] - f_val[-1])])
    plt.tight_layout(pad=0.1)


if __name__ == "__main__":

    np.random.seed(1)

    dim = 2
    bound = 3.0
    x = np.linspace(-bound, bound, 100)
    y = np.linspace(-bound, bound, 100)

    A = rand_psd(dim)
    b = np.random.randn(dim)
    C = rand_psd(dim) * 1000
    d = np.random.randn(dim)
    z = np.abs(np.random.randn()) + 10.0

    # normalized form
    A = np.eye(2)
    b = np.zeros(2)
    C = np.array([[12.0, 0.5], [0.5, 1.0]])
    d = np.array([-0.13, 1.0])

    w, v = np.linalg.eigh(C)
    w[0] *= 3.0
    C = (v * w[None, :]) @ v.T

    d_hat = v.T @ d
    d_hat[0] *= 0.4
    d_hat[1] *= 2.0
    d = v @ d_hat

    z = 0.07

    # solve the problem
    sol, lambda_, phi, v_tilde = lqpqm_solver(A, b, C, d, z)

    X, Y = np.meshgrid(x, y)
    obj = objective(X, Y, A, b, C, d, z)

    sns.set(context="paper", style="whitegrid")
    sns.set_palette(sns.cubehelix_palette(4))

    _cm = 0.39  # 1 cm in inches
    fig = plt.figure(figsize=(8.5 * _cm, 5.0 * _cm))

    ax1 = fig.add_subplot(1, 2, 2)
    ct = ax1.contourf(x, y, obj, levels=20, cmap=cm.coolwarm)
    ax1.plot(sol[0], sol[1], "x")
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    sns.despine(left=True, bottom=True)
    # fig.colorbar(ct, ax=ax1)

    ax2 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2.view_init(50, 35 + 180)
    ax2.set_xlabel("$x_1$")
    ax2.set_ylabel("$x_2$")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.contour3D(X, Y, obj, cmap=cm.coolwarm, levels=30)

    if not os.path.exists("figures"):
        os.mkdir("figures")

    fig.tight_layout()
    fig.savefig("figures/figure1_loss_landscape.pdf")

    show_constraint(
        lambda_[None], phi[None, :], v_tilde[None, :], np.array([z]), show_init=False,
    )
    sns.despine(left=True, bottom=True)
    plt.savefig("figures/figure2_secular_eq.pdf")
