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
from adcc.exceptions import InputError
import warnings
#from adcc.solver.lanczos import lanczos
#from adcc.solver.davidson import jacobi_davidson
import adcc.solver.lanczos as lanczos
import adcc.solver.davidson as davidson
from adcc.solver.explicit_symmetrisation import (IndexSpinSymmetrisation,
                                             IndexSymmetrisation)
import sys
from adcc.guess import (guesses_any, guesses_singlet, guesses_spin_flip,
                    guesses_triplet)

# the following things should be adapted in adcc, because then all the
# functions in this file can just be taken from adcc.

# validate_state_parameters only needs to be public
# estimate_n_guesses only needs to be public
# diagonalize_adcmatrix only needs to be public, but it would also be nice,
# if the callback function could be given as an argument
# setup_solver_printing only needs to be public, but it would be nicer
# to write a new one, where qed_adc is given as the method
# obtain_guesses_by_inspection only needs to be public


def validate_state_parameters(reference_state, n_states=None, n_singlets=None,
                              n_triplets=None, n_spin_flip=None, kind="any"):
    """
    Check the passed state parameters for consistency with itself and with
    the passed reference and normalise them. In the end return the number of
    states and the corresponding kind parameter selected.
    Internal function called from run_adc.
    """
    if sum(nst is not None for nst in [n_states, n_singlets,
                                       n_triplets, n_spin_flip]) > 1:
        raise InputError("One may only specify one out of n_states, "
                         "n_singlets, n_triplets and n_spin_flip")

    if n_singlets is not None:
        if not reference_state.restricted:
            raise InputError("The n_singlets parameter may only be employed "
                             "for restricted references")
        if kind not in ["singlet", "any"]:
            raise InputError(f"Kind parameter {kind} not compatible "
                             "with n_singlets > 0")
        kind = "singlet"
        n_states = n_singlets
    if n_triplets is not None:
        if not reference_state.restricted:
            raise InputError("The n_triplets parameter may only be employed "
                             "for restricted references")
        if kind not in ["triplet", "any"]:
            raise InputError(f"Kind parameter {kind} not compatible "
                             "with n_triplets > 0")
        kind = "triplet"
        n_states = n_triplets
    if n_spin_flip is not None:
        if reference_state.restricted:
            raise InputError("The n_spin_flip parameter may only be employed "
                             "for unrestricted references")
        if kind not in ["spin_flip", "any"]:
            raise InputError(f"Kind parameter {kind} not compatible "
                             "with n_spin_flip > 0")
        kind = "spin_flip"
        n_states = n_spin_flip

    # Check if there are states to be computed
    if n_states is None or n_states == 0:
        raise InputError("No excited states to be computed. Specify at least "
                         "one of n_states, n_singlets, n_triplets, "
                         "or n_spin_flip")
    if n_states < 0:
        raise InputError("n_states needs to be positive")

    if kind not in ["any", "spin_flip", "singlet", "triplet"]:
        raise InputError("The kind parameter may only take the values 'any', "
                         "'singlet', 'triplet' or 'spin_flip'")
    if kind in ["singlet", "triplet"] and not reference_state.restricted:
        raise InputError("kind==singlet and kind==triplet are only valid for "
                         "ADC calculations in combination with a restricted "
                         "ground state.")
    if kind in ["spin_flip"] and reference_state.restricted:
        raise InputError("kind==spin_flip is only valid for "
                         "ADC calculations in combination with an unrestricted "
                         "ground state.")
    
    return n_states, kind


def diagonalise_adcmatrix(matrix, n_states, kind, eigensolver="davidson",
                          guesses=None, n_guesses=None, n_guesses_doubles=None,
                          conv_tol=None, output=sys.stdout, **solverargs):
    """
    This function seeks appropriate guesses and afterwards proceeds to
    diagonalise the ADC matrix using the specified eigensolver.
    Internal function called from run_adc.
    """
    reference_state = matrix.reference_state

    # Determine default ADC convergence tolerance
    if conv_tol is None:
        conv_tol = max(10 * reference_state.conv_tol, 1e-6)
    if reference_state.conv_tol > conv_tol:
        raise InputError(
            "Convergence tolerance of SCF results "
            f"(== {reference_state.conv_tol}) needs to be lower than ADC "
            f"convergence tolerance parameter conv_tol (== {conv_tol})."
        )

    # Determine explicit_symmetrisation
    explicit_symmetrisation = IndexSymmetrisation
    if kind in ["singlet", "triplet"]:
        explicit_symmetrisation = IndexSpinSymmetrisation(
            matrix, enforce_spin_kind=kind
        )

    # Set some solver-specific parameters
    if eigensolver == "davidson":
        n_guesses_per_state = 2
        callback = setup_solver_printing(
            "Jacobi-Davidson", matrix, kind, davidson.default_print,
            output=output)
        run_eigensolver = davidson.jacobi_davidson
    elif eigensolver == "lanczos":
        n_guesses_per_state = 1
        callback = setup_solver_printing(
            "Lanczos", matrix, kind, lanczos.default_print,
            output=output)
        run_eigensolver = lanczos.lanczos
    else:
        raise InputError(f"Solver {eigensolver} unknown, try 'davidson'.")

    # Obtain or check guesses
    if guesses is None:
        if n_guesses is None:
            n_guesses = estimate_n_guesses(matrix, n_states, n_guesses_per_state)
        guesses = obtain_guesses_by_inspection(matrix, n_guesses, kind, n_guesses_doubles)
    else:
        if len(guesses) < n_states:
            raise InputError("Less guesses provided via guesses (== {}) "
                             "than states to be computed (== {})"
                             "".format(len(guesses), n_states))
        if n_guesses is not None:
            warnings.warn("Ignoring n_guesses parameter, since guesses are "
                          "explicitly provided.")
        if n_guesses_doubles is not None:
            warnings.warn("Ignoring n_guesses_doubles parameter, since guesses "
                          "are explicitly provided.")

    solverargs.setdefault("which", "SA")
    return run_eigensolver(matrix, guesses, n_ep=n_states, conv_tol=conv_tol,
                           callback=callback,
                           explicit_symmetrisation=explicit_symmetrisation,
                           **solverargs)


def estimate_n_guesses(matrix, n_states, singles_only=True,
                       n_guesses_per_state=2):
    """
    Implementation of a basic heuristic to find a good number of guess
    vectors to be searched for using the find_guesses function.
    Internal function called from run_adc.

    matrix             ADC matrix
    n_states           Number of states to be computed
    singles_only       Try to stay withing the singles excitation space
                       with the number of guess vectors.
    n_guesses_per_state  Number of guesses to search for for each state
    """
    # Try to use at least 4 or twice the number of states
    # to be computed as guesses
    n_guesses = n_guesses_per_state * max(2, n_states)

    if singles_only:
        # Compute the maximal number of sensible singles block guesses.
        # This is roughly the number of occupied alpha orbitals
        # times the number of virtual alpha orbitals
        #
        # If the system is core valence separated, then only the
        # core electrons count as "occupied".
        mospaces = matrix.mospaces
        sp_occ = "o2" if matrix.is_core_valence_separated else "o1"
        n_virt_a = mospaces.n_orbs_alpha("v1")
        n_occ_a = mospaces.n_orbs_alpha(sp_occ)
        n_guesses = min(n_guesses, n_occ_a * n_virt_a)

    # Adjust if we overshoot the maximal number of sensible singles block
    # guesses, but make sure we get at least n_states guesses
    return max(n_states, n_guesses)


def obtain_guesses_by_inspection(matrix, n_guesses, kind, n_guesses_doubles=None):  # noqa: E501
    """
    Obtain guesses by inspecting the diagonal matrix elements.
    If n_guesses_doubles is not None, this number is always adhered to.
    Otherwise the number of doubles guesses is adjusted to fill up whatever
    the singles guesses cannot provide to reach n_guesses.
    Internal function called from run_adc.
    """
    if n_guesses_doubles is not None and n_guesses_doubles > 0 \
       and "pphh" not in matrix.axis_blocks:
        raise InputError("n_guesses_doubles > 0 is only sensible if the ADC "
                         "method has a doubles block (i.e. it is *not* ADC(0), "
                         "ADC(1) or a variant thereof.")

    # Determine guess function
    guess_function = {"any": guesses_any, "singlet": guesses_singlet,
                      "triplet": guesses_triplet,
                      "spin_flip": guesses_spin_flip}[kind]

    # Determine number of singles guesses to request
    n_guess_singles = n_guesses
    if n_guesses_doubles is not None:
        n_guess_singles = n_guesses - n_guesses_doubles
    singles_guesses = guess_function(matrix, n_guess_singles, block="ph")

    doubles_guesses = []
    if "pphh" in matrix.axis_blocks:
        # Determine number of doubles guesses to request if not
        # explicitly specified
        if n_guesses_doubles is None:
            n_guesses_doubles = n_guesses - len(singles_guesses)
        if n_guesses_doubles > 0:
            doubles_guesses = guess_function(matrix, n_guesses_doubles,
                                             block="pphh")

    total_guesses = singles_guesses + doubles_guesses
    if len(total_guesses) < n_guesses:
        raise InputError("Less guesses found than requested: {} found, "
                         "{} requested".format(len(total_guesses), n_guesses))
    return total_guesses


def setup_solver_printing(solmethod_name, matrix, kind, default_print,
                          output=None):
    """
    Setup default printing for solvers. Internal function called from run_adc.
    """
    kstr = " "
    if kind != "any":
        kstr = " " + kind
    method_name = f"{matrix}"
    if hasattr(matrix, "method"):
        method_name = matrix.method.name

    if output is not None:
        print(f"Starting {method_name}{kstr} {solmethod_name} ...",
              file=output)

        def inner_callback(state, identifier):
            default_print(state, identifier, output)
        return inner_callback

