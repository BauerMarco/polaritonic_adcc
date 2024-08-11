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
import numpy as np
from qed_npadc_s2s_tdm_terms import qed_npadc_s2s_tdm_terms
from adcc.timings import timed_member_call
from adcc.adc_pp import transition_dm
from adcc.OneParticleOperator import product_trace
from adcc import block as b
from adcc.functions import einsum


class qed_npadc_exstates:
    """Output object for the truncated state space methods"""
    def __init__(self, exstates, qed_coupl_level=1, **kw_args):
        self.qed_coupl_level = qed_coupl_level
        self.matrix = exstates.matrix
        self.ground_state = self.matrix.ground_state
        self.reference_state = self.matrix.ground_state.reference_state
        self.operators = self.reference_state.operators
        self.excitation_vector = exstates.excitation_vector
        self.excitation_energy = exstates.excitation_energy
        self.method = exstates.method

    @timed_member_call(timer="_property_timer")
    def transition_dipole_moments_qed(self):
        """
        List of transition dipole moments of all computed states
        to build the QED-matrix in the basis of the diagonal
        purely electric subblock
        """
        dipole_integrals = self.operators.electric_dipole

        def tdm(i, prop_level):
            return transition_dm(prop_level, self.ground_state,
                                    self.excitation_vector[i])

        prop_level = "adc" + str(self.qed_coupl_level - 1)
        return np.array([
            [product_trace(comp, tdm(i, prop_level))
                for comp in dipole_integrals]
            for i in np.arange(len(self.excitation_energy))
        ])

    @timed_member_call(timer="_property_timer")
    def s2s_dipole_moments_qed(self):
        """
        List of s2s transition dipole moments of all computed states
        to build the QED-matrix in the basis of the diagonal
        purely electric subblock
        """
        dipole_integrals = self.operators.electric_dipole
        print("note, that only the z coordinate of the "
                "dipole integrals is calculated")
        n_states = len(self.excitation_energy)

        def s2s(i, f, name):
            vec = self.excitation_vector
            return qed_npadc_s2s_tdm_terms(name, self.ground_state,
                                                vec[i], vec[f])

        def final_block(name):
            return np.array([[[product_trace(comp, s2s(i, j, name))
                                for comp in dipole_integrals]
                                for j in np.arange(n_states)]
                                for i in np.arange(n_states)])

        block_dict = {"qed_adc1_off_diag": final_block("adc1")}

        if self.qed_coupl_level == 2:
            keys = ("qed_adc2_diag", "qed_adc2_edge_couple",
                    "qed_adc2_edge_phot_couple", "qed_adc2_ph_pphh",
                    "qed_adc2_pphh_ph")
            for key in keys:
                block_dict[key] = final_block(key)
        return block_dict

    @timed_member_call(timer="_property_timer")
    def qed_second_order_ph_ph_couplings(self):
        """
        List of blocks containing the expectation value of the perturbation
        of the Hamiltonian for all computed states required
        to build the QED-matrix in the basis of the diagonal
        purely electric subblock
        """
        if self.qed_coupl_level == 2:
            qed_t1 = self.ground_state.qed_t1(b.ov)

            def couple(qed_t1, ul, ur):
                return {
                    b.ooov: einsum("kc,ia,ja->kjic", qed_t1, ul, ur)
                    + einsum("ka,ia,jb->jkib", qed_t1, ul, ur),
                    b.ovvv: einsum("kc,ia,ib->kacb", qed_t1, ul, ur)
                    + einsum("ic,ia,jb->jabc", qed_t1, ul, ur)
                }

            def phot_couple(qed_t1, ul, ur):
                return {
                    b.ooov: einsum("kc,ia,ja->kijc", qed_t1, ul, ur)
                    + einsum("kb,ia,jb->ikja", qed_t1, ul, ur),
                    b.ovvv: einsum("kc,ia,ib->kbca", qed_t1, ul, ur)
                    + einsum("jc,ia,jb->ibac", qed_t1, ul, ur)
                }

            def prod_sum(hf, two_p_op):
                return (einsum("ijka,ijka->", hf.ooov, two_p_op[b.ooov])
                        + einsum("iabc,iabc->", hf.ovvv, two_p_op[b.ovvv]))

            def final_block(func):
                return np.array([
                    [prod_sum(self.reference_state, func(qed_t1, i.ph, j.ph))
                     for i in self.excitation_vector]
                    for j in self.excitation_vector])

            block_dict = {}
            block_dict["couple"] = final_block(couple)
            block_dict["phot_couple"] = final_block(phot_couple)

            return block_dict
        else:
            return 0
