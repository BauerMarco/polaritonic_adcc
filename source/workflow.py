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

import adcc
import sys
import numpy as np
import warnings
from qed_matrix_from_diag_adc import qed_matrix_from_diag_adc
from qed_adc_in_std_basis_with_self_en import first_order_qed_matrix
from refstate import refstate
from qed_npadc_exstates import qed_npadc_exstates
from qed_mp import qed_mp
from full_qed_matrix import qed_matrix_full
from solver_functions_from_adcc import (validate_state_parameters,
                                        estimate_n_guesses,
                                        obtain_guesses_by_inspection,
                                        diagonalise_adcmatrix)
from libadcc import set_lt_scalar#set_from_ndarray

__all__ = ["run_qed_adc"]

def run_qed_adc(data_or_matrix, coupl=None, freq=None, qed_hf=True,
                qed_coupl_level=1, n_states=None, kind="any", conv_tol=None,
                eigensolver="davidson", guesses=None, n_guesses=None,
                n_guesses_doubles=None, output=sys.stdout, core_orbitals=None,
                frozen_core=None, frozen_virtual=None, method=None,
                n_singlets=None, n_triplets=None, n_spin_flip=None,
                environment=None, **solverargs):
    # Validation of given parameters
    if method == "adc0":
        raise NotImplementedError("in zeroth order the system is decoupled from the cavity")
    if coupl == None or freq == None:
        raise SyntaxError("coupl and freq can not be set to None")
    if np.linalg.norm(coupl) < 1e-15 or np.linalg.norm(freq) < 1e-15:
        raise NotImplementedError("If you want to do a calculation with zero"
                                  "coupling or zero frequency, you can coduct"
                                  "a standard adcc calculation")
    if "csv" in method:
        warnings.warn("CVS ADC excitations generally do not obey the dipole approximation,"
                      "and therefore should not be used at all with polaritonic_adcc")
    # TODO: Here we always multiply with sqrt(2 * omega), since this
    # eases up the Hamiltonian and is required if you provide a hilbert
    # package QED-HF input. This can differ between QED-HF implementations,
    # e.g. the psi4numpy QED-RHF helper does not do that. Therefore, this
    # factor needs to be adjusted depending on the input.
    coupl_adapted = np.array(coupl) * np.sqrt(2 * np.linalg.norm(np.real(freq)))
    print("coupling parameter in polaritonic_adcc was adapted as sqrt(2 * real(freq)), "
          "because the definition for the coupling parameter here is "
          r"$\lambda = \frac{\epsilon}{\epsilon_0 \epsilon_r V}$")
    # Determining the correct computation from the given input parameters
    #if qed_hf:
    if isinstance(data_or_matrix, refstate):
        qed_groundstate = qed_mp(data_or_matrix, np.linalg.norm(np.real(freq)), qed_hf=qed_hf)
    elif isinstance(data_or_matrix, qed_mp):
        qed_groundstate = data_or_matrix
    elif isinstance(data_or_matrix, qed_matrix_full):
        pass
    else:
        qed_refstate = refstate(data_or_matrix, qed_hf, coupl_adapted, core_orbitals=core_orbitals, frozen_core=frozen_core,
                        frozen_virtual=frozen_virtual)
        qed_groundstate = qed_mp(qed_refstate, np.linalg.norm(np.real(freq)), qed_hf=qed_hf)
    if type(qed_coupl_level) == int and qed_hf:
        exstates = adcc.run_adc(qed_groundstate, n_states=n_states, kind=kind, conv_tol=conv_tol,
                eigensolver=eigensolver, guesses=guesses, n_guesses=n_guesses,
                n_guesses_doubles=n_guesses_doubles, output=output, core_orbitals=core_orbitals,
                frozen_core=frozen_core, frozen_virtual=frozen_virtual, method=method,
                n_singlets=n_singlets, n_triplets=n_triplets, n_spin_flip=n_spin_flip,
                environment=environment, **solverargs)
        qed_exstates = qed_npadc_exstates(exstates, qed_coupl_level)
        qed_matrix_class = qed_matrix_from_diag_adc(qed_exstates, coupl_adapted, freq)
    if not qed_coupl_level:# and qed_hf:
        # full qed at given adc level
        if (method not in ["adc1", "adc2"] and qed_hf) or (method != "adc1" and not qed_hf):
            raise NotImplementedError("full polaritonic ADC is only implemented"
                                      "for ADC(1) and ADC(2) with a polaritonic"
                                      "Hartree-Fock reference and only for ADC(1)"
                                      "with a standard Hartree-Fock reference.")
        if any(np.iscomplex(freq)):
            raise NotImplementedError("full polaritonic ADC is not implemented for lossy cavities")
        if isinstance(data_or_matrix, qed_matrix_full):
            #diag = diagonalizer(data_or_matrix)
            qed_mat_full = data_or_matrix
        elif not isinstance(data_or_matrix, (qed_mp, refstate)):# and qed_hf:
            qed_mat_full = qed_matrix_full(method, qed_groundstate)
            #diag = diagonalizer(qed_mat_full)
        else:
            raise TypeError("full qed adc matrix either needs to be provided or "
                            "build from a qed_mp instance here, which is valid with "
                            "respect to the qed_hf parameter.")
        
        diag = diagonalizer(qed_mat_full, n_states, kind, guesses=guesses, n_guesses=n_guesses,
        n_guesses_doubles=n_guesses_doubles, conv_tol=conv_tol, output=output,
        eigensolver=eigensolver, n_triplets=n_triplets, n_singlets=n_singlets,
        n_spin_flip=n_spin_flip, **solverargs)

        exstates = adcc.ExcitedStates(diag)
    elif qed_coupl_level == 1 and qed_hf:
        # adc at given level with adapted ERIs and 1st order qed coupling
        qed_matrix = qed_matrix_class.first_order_coupling()
        exstates.qed_matrix = qed_matrix
    elif qed_coupl_level == 2 and qed_hf:
        # adc at given level with adapted ERIs and 2st order qed coupling
        if method == "adc1":
            raise NotImplementedError("requesting adc1 with second order coupling is not possible")
        qed_matrix = qed_matrix_class.second_order_coupling()
        exstates.qed_matrix = qed_matrix
    elif qed_coupl_level == 1 and not qed_hf:
        # adc at given level without adapted ERIs and 1st order qed coupling
        # this is the equivalent to using field free states
        exstates = adcc.run_adc(data_or_matrix, n_states=n_states, kind=kind, conv_tol=conv_tol,
                eigensolver=eigensolver, guesses=guesses, n_guesses=n_guesses,
                n_guesses_doubles=n_guesses_doubles, output=output, core_orbitals=core_orbitals,
                frozen_core=frozen_core, frozen_virtual=frozen_virtual, method=method,
                n_singlets=n_singlets, n_triplets=n_triplets, n_spin_flip=n_spin_flip,
                environment=environment, **solverargs)
        qed_matrix = first_order_qed_matrix(exstates, coupl_adapted, freq)
        exstates.qed_matrix = qed_matrix
    else:
        raise NotImplementedError(f"the given combination of qed_coupl_level = "
                                  "{qed_coupl_level} and qed_hf = {qed_hf} is not" 
                                  "implemented")
        
    return exstates


def diagonalizer(matrix, n_states, kind, guesses=None, n_guesses=None,
        n_guesses_doubles=None, conv_tol=None, output=sys.stdout,
        eigensolver="davidson", n_triplets=None, n_singlets=None,
        n_spin_flip=None, **solverargs):
    
    n_states, kind = validate_state_parameters(
        matrix.reference_state, n_states=n_states, n_singlets=n_singlets,
        n_triplets=n_triplets, n_spin_flip=n_spin_flip, kind=kind)
    
    if eigensolver == "davidson":
        n_guesses_per_state = 2
    elif eigensolver == "lanczos":
        n_guesses_per_state = 1
    else:
        raise NotImplementedError(f"eigensolver {eigensolver} is unknown")
    
    n_guesses = estimate_n_guesses(matrix, n_states, n_guesses_per_state=n_guesses_per_state)

    guesses = obtain_guesses_by_inspection_qed(matrix, n_guesses, kind, n_guesses_doubles=n_guesses_doubles)

    diag_result = diagonalise_adcmatrix(matrix, n_states, kind, eigensolver=eigensolver,
                          guesses=guesses, n_guesses=n_guesses, n_guesses_doubles=n_guesses_doubles,
                          conv_tol=conv_tol, output=output, **solverargs)
    return diag_result




def obtain_guesses_by_inspection_qed(matrix, n_guesses, kind, n_guesses_doubles=None):  # noqa: E501
    """
    Obtain guesses for full qed method, by pushing the subblocks
    into obtain_guesses_by_inspection.
    """
    # It seems like qed_adc converges better with all three guesses
    # originating from the purely electronic diagonal. If the actual
    # diagonals are desired, do something like the commented out hacks.
    guesses_elec = obtain_guesses_by_inspection(
        matrix, n_guesses, kind, n_guesses_doubles)
    #matrix.return_diag_as = "phot"  # hack for the guess setup
    guesses_phot = obtain_guesses_by_inspection(
        matrix, n_guesses, kind, n_guesses_doubles)
    #matrix.return_diag_as = "phot2"  # hack for the guess setup
    guesses_phot2 = obtain_guesses_by_inspection(
        matrix, n_guesses, kind, n_guesses_doubles)
    #matrix.return_diag_as = "full"  # resets the hacks from above

    # Usually only few states are requested and most of them are close
    # to pure electronic states, so we initialize the guess vectors
    # as almost purely electric guesses.
    for i in np.arange(n_guesses):
        # TODO: maybe make these values accessible by a keyword, since
        # they can tune the performance. From my experience these work
        # very well though
        guesses_phot[i] *= 0.02
        guesses_phot2[i] *= 0.001
    if n_guesses != len(guesses_phot):
        raise adcc.exceptions.InputError("amount of guesses for electronic and photonic must be "
                         "equal, but are {} electronic and {} photonic "
                         "guesses".format(len(guesses_elec), len(guesses_phot)))

    zero = set_lt_scalar(0.0)
    #zero = set_from_ndarray(np.array([0.0]))

    if hasattr(guesses_elec[0], "pphh"):
        final_guesses = [adcc.AmplitudeVector(**{
            "ph": guesses_elec[guess_index].ph,
            "pphh": guesses_elec[guess_index].pphh,
            "gs1": zero.copy(), "ph1": guesses_phot[guess_index].ph,
            "pphh1": guesses_phot[guess_index].pphh,
            "gs2": zero.copy(), "ph2": guesses_phot2[guess_index].ph,
            "pphh2": guesses_phot2[guess_index].pphh
        }) for guess_index in np.arange(n_guesses)]
    else:
        final_guesses = [adcc.AmplitudeVector(**{
            "ph": guesses_elec[guess_index].ph,
            "gs1": zero.copy(), "ph1": guesses_phot[guess_index].ph,
            "gs2": zero.copy(), "ph2": guesses_phot2[guess_index].ph
        }) for guess_index in np.arange(n_guesses)]

    # TODO: maybe make these values accessible by a keyword, since
    # they can tune the performance. From my experience these work
    # quite well though, but depending on how strong a state couples
    # to the single photon dispersion mode and how many states one
    # requests, adjusting these values can increase the
    # convergence rate
    # for stronger coupling e.g. 2
    final_guesses[n_guesses - 2].gs1.set_from_ndarray(np.array([5]))
    # for stronger coupling e.g. 5
    final_guesses[n_guesses - 1].gs2.set_from_ndarray(np.array([20]))

    return [vec / np.sqrt(vec @ vec) for vec in final_guesses]
