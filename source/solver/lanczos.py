# (C) Copyright 2023 Marco Bauer
# 
# This file is part of polaritonic_adcc.
# 
# polaritonic_adcc is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# polaritonic_adcc is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with polaritonic_adcc. If not, see <http://www.gnu.org/licenses/>.
#
import warnings
import numpy as np
import scipy.linalg as la

from ..functions import lincomb

from adcc.solver.common import select_eigenpairs
from .LanczosIterator import LanczosIterator
from adcc.solver.explicit_symmetrisation import IndexSymmetrisation
from adcc.solver.lanczos import LanczosState, check_convergence

"""
This function is mostly copied from adcc, because the evaluate and licomb
functions needed to be adapted. This could also be circumvented, if one could
create a libtensor array of specified shape from the python side.
"""

def compute_true_residuals(subspace, rvals, rvecs, epair_mask):
    """
    Compute the true residuals and residual norms (and not the ones estimated from
    the Lanczos subspace).
    """
    V = subspace.subspace
    AV = subspace.matrix_product

    def form_residual(rval, rvec):
        coefficients = np.hstack((rvec, -rval * rvec))
        return lincomb(coefficients, AV + V, evaluate=True)
    residuals = [form_residual(rvals[i], rvec)
                 for i, rvec in enumerate(np.transpose(rvecs))
                 if i in epair_mask]
    eigenvectors = [lincomb(rvec, V, evaluate=True)
                    for i, rvec in enumerate(np.transpose(rvecs))
                    if i in epair_mask]
    rnorms = np.array([np.sqrt(r @ r) for r in residuals])

    # Note here the actual residual norm (and not the residual norm squared)
    # is returned.
    return eigenvectors, residuals, rnorms


def amend_true_residuals(state, subspace, rvals, rvecs, epair_mask):
    """
    Compute the true residuals and residual norms (and not the ones estimated from
    the Lanczos subspace) and amend the `state` accordingly.
    """
    res = compute_true_residuals(subspace, rvals, rvecs, epair_mask)
    state.eigenvectors, state.residuals, state.residual_norms = res

    # TODO For consistency with the Davidson the residual norms are
    #      squared to give output in the same order of magnitude.
    state.residual_norms = state.residual_norms**2

    return state


def lanczos_iterations(iterator, n_ep, min_subspace, max_subspace, conv_tol=1e-9,
                       which="LA", max_iter=100, callback=None,
                       debug_checks=False, state=None):
    """Drive the Lanczos iterations

    Parameters
    ----------
    iterator : LanczosIterator
        Iterator generating the Lanczos subspace (contains matrix, guess,
        residual, Ritz pairs from restart, symmetrisation and orthogonalisation)
    n_ep : int
        Number of eigenpairs to be computed
    min_subspace : int
        Subspace size to collapse to when performing a thick restart.
    max_subspace : int
        Maximal subspace size
    conv_tol : float, optional
        Convergence tolerance on the l2 norm squared of residuals to consider
        them converged
    which : str, optional
        Which eigenvectors to converge to (e.g. LM, LA, SM, SA)
    max_iter : int, optional
        Maximal number of iterations
    callback : callable, optional
        Callback to run after each iteration
    debug_checks : bool, optional
        Enable some potentially costly debug checks
        (Loss of orthogonality etc.)
    """
    if callback is None:
        def callback(state, identifier):
            pass

    # TODO For consistency with the Davidson the conv_tol is interpreted
    #      as the residual norm *squared*. Arnoldi, however, uses the actual norm
    #      to check for convergence and so on. See also the comment in Davidson
    #      around the line computing state.residual_norms
    #
    #      See also the squaring of the residual norms below
    tol = np.sqrt(conv_tol)

    if state is None:
        state = LanczosState(iterator)
        callback(state, "start")
        state.timer.restart("iteration")
        n_applies_offset = 0
    else:
        n_applies_offset = state.n_applies

    for subspace in iterator:
        b = subspace.rayleigh_extension
        with state.timer.record("rayleigh_ritz"):
            rvals, rvecs = np.linalg.eigh(subspace.subspace_matrix)

        if debug_checks:
            eps = np.finfo(float).eps
            orthotol = max(tol / 1000, subspace.n_problem * eps)
            orth = subspace.check_orthogonality(orthotol)
            state.subspace_orthogonality = orth

        is_rval_converged, eigenpair_error = check_convergence(subspace, rvals,
                                                               rvecs, tol)

        # Update state
        state.n_iter += 1
        state.n_applies = subspace.n_applies + n_applies_offset
        state.converged = False
        state.eigenvectors = None  # Not computed in Lanczos
        state.subspace_vectors = subspace.subspace
        state.subspace_residual = subspace.residual

        epair_mask = select_eigenpairs(rvals, n_ep, which)
        state.eigenvalues = rvals[epair_mask]
        state.residual_norms = eigenpair_error[epair_mask]
        converged = np.all(is_rval_converged[epair_mask])

        # TODO For consistency with the Davidson the residual norms are squared
        #      again to give output in the same order of magnitude.
        state.residual_norms = state.residual_norms**2

        callback(state, "next_iter")
        state.timer.restart("iteration")

        if converged:
            state = amend_true_residuals(state, subspace, rvals,
                                         rvecs, epair_mask)
            state.converged = True
            callback(state, "is_converged")
            state.timer.stop("iteration")
            return state

        if state.n_iter >= max_iter:
            warnings.warn(la.LinAlgWarning(
                f"Maximum number of iterations (== {max_iter}) "
                "reached in lanczos procedure."))
            state = amend_true_residuals(state, subspace, rvals,
                                         rvecs, epair_mask)
            state.timer.stop("iteration")
            state.converged = False
            return state

        if len(rvecs) + subspace.n_block > max_subspace:
            callback(state, "restart")

            epair_mask = select_eigenpairs(rvals, min_subspace, which)
            V = subspace.subspace
            vn, betan = subspace.ortho.qr(subspace.residual)

            Y = [lincomb(rvec, V, evaluate=True)
                 for i, rvec in enumerate(np.transpose(rvecs))
                 if i in epair_mask]
            Theta = rvals[epair_mask]
            Sigma = rvecs[:, epair_mask].T @ b @ betan.T

            iterator = LanczosIterator(
                iterator.matrix, vn, ritz_vectors=Y, ritz_values=Theta,
                ritz_overlaps=Sigma,
                explicit_symmetrisation=iterator.explicit_symmetrisation
            )
            state.n_restart += 1
            return lanczos_iterations(
                iterator, n_ep, min_subspace, max_subspace, conv_tol, which,
                max_iter, callback, debug_checks, state)

    state = amend_true_residuals(state, subspace, rvals, rvecs, epair_mask)
    state.timer.stop("iteration")
    state.converged = False
    warnings.warn(la.LinAlgWarning(
        "Lanczos procedure found maximal subspace possible. Iteration cannot be "
        "continued like this and will be aborted without convergence. "
        "Try a different guess."))
    return state


def lanczos(matrix, guesses, n_ep, max_subspace=None,
            conv_tol=1e-9, which="LM", max_iter=100,
            callback=None, debug_checks=False,
            explicit_symmetrisation=IndexSymmetrisation,
            min_subspace=None):
    """Lanczos eigensolver for ADC problems

    Parameters
    ----------
    matrix
        ADC matrix instance
    guesses : list
        Guess vectors (fixes also the Lanczos block size)
    n_ep : int
        Number of eigenpairs to be computed
    max_subspace : int or NoneType, optional
        Maximal subspace size
    conv_tol : float, optional
        Convergence tolerance on the l2 norm squared of residuals to consider
        them converged
    which : str, optional
        Which eigenvectors to converge to (e.g. LM, LA, SM, SA)
    max_iter : int, optional
        Maximal number of iterations
    callback : callable, optional
        Callback to run after each iteration
    debug_checks : bool, optional
        Enable some potentially costly debug checks
        (Loss of orthogonality etc.)
    explicit_symmetrisation : optional
        Explicit symmetrisation to use after orthogonalising the
        subspace vectors. Allows to correct for loss of index or spin
        symmetries during orthogonalisation (type or instance).
    min_subspace : int or NoneType, optional
        Subspace size to collapse to when performing a thick restart.
    """
    if explicit_symmetrisation is not None and \
            isinstance(explicit_symmetrisation, type):
        explicit_symmetrisation = explicit_symmetrisation(matrix)
    iterator = LanczosIterator(matrix, guesses,
                               explicit_symmetrisation=explicit_symmetrisation)

    if not isinstance(guesses, list):
        guesses = [guesses]
    if not max_subspace:
        max_subspace = max(2 * n_ep + len(guesses), 20, 8 * len(guesses))
    if not min_subspace:
        min_subspace = n_ep + 2 * len(guesses)
    if conv_tol < matrix.shape[1] * np.finfo(float).eps:
        warnings.warn(la.LinAlgWarning(
            "Convergence tolerance (== {:5.2g}) lower than "
            "estimated maximal numerical accuracy (== {:5.2g}). "
            "Convergence might be hard to achieve."
            "".format(conv_tol, matrix.shape[1] * np.finfo(float).eps)
        ))

    return lanczos_iterations(iterator, n_ep, min_subspace, max_subspace,
                              conv_tol, which, max_iter, callback, debug_checks)
