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

__all__ = ["run_qed_adc"]

def run_qed_adc(data_or_matrix, coupl=None, freq=None, qed_hf=False,
                qed_coupl_level=1, method=None, **adcc_args):
    # Validation of given parameters
    if method == "adc0":
        raise NotImplementedError("in zeroth order the system is decoupled from the cavity")
    if coupl == None or freq == None:
        raise SyntaxError("coupl and freq can not be set to None")
    if np.linalg.norm(coupl) < 1e-15 or np.linalg.norm(freq) < 1e-15:
        raise NotImplementedError("If you want to do a calculation with zero"
                                  "coupling or zero frequency, you can coduct"
                                  "a standard adcc calculation")
    if method.startswith("cvs"):
        warnings.warn("CVS ADC excitations generally do not obey the dipole approximation,"
                      "and therefore should not be used at all with polaritonic_adcc")
    # TODO: Here we always multiply with sqrt(2 * omega), since this
    # eases up the Hamiltonian and is required if you provide a hilbert
    # package QED-HF input. This can differ between QED-HF implementations,
    # e.g. the psi4numpy QED-RHF helper does not do that. Therefore, this
    # factor needs to be adjusted depending on the input.
    coupl_adapted = np.array(coupl) * np.sqrt(2 * np.linalg.norm(np.real(freq)))
    # Determining the correct computation from the given input parameters
    if qed_hf:
        #print("double check whether **adcc_args can be given here, or if explicit arguments are required")
        qed_refstate = refstate(data_or_matrix, coupl=coupl_adapted, **adcc_args)
        qed_groundstate = qed_mp(qed_refstate, np.linalg.norm(np.real(freq)), qed_hf=qed_hf)#, **adcc_args)
    if type(qed_coupl_level) == int and qed_hf:
        # the following wont work...best would be building qed_npadc_exstates from the exstates result of adcc
        #matrix = adcc.construct_adcmatrix(qed_mp, method=method, **adcc_args)
        #exstates = adcc.run_adc(matrix, method=method, **adcc_args)
        #diag = diagonalize(matrix)
        exstates = adcc.run_adc(qed_groundstate, method=method, **adcc_args)
        qed_exstates = qed_npadc_exstates(exstates, qed_coupl_level)
        qed_matrix_class = qed_matrix_from_diag_adc(qed_exstates, coupl_adapted, freq)
    if not qed_coupl_level and qed_hf:
        # full qed at given adc level
        if method not in ["adc1", "adc2"]:
            raise NotImplementedError("full polaritonic ADC is only implemented"
                                      "for ADC(1) and ADC(2)")
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
        exstates = adcc.run_adc(data_or_matrix, method=method, **adcc_args)
        qed_matrix = first_order_qed_matrix(exstates, coupl_adapted, freq)
        exstates.qed_matrix = qed_matrix
    else:
        raise NotImplementedError(f"the given combination of qed_coupl_level = "
                                  "{qed_coupl_level} and qed_hf = {qed_hf} is not" 
                                  "implemented")
        
    return exstates



