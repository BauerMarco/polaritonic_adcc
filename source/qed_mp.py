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
from adcc import block as b
from adcc.functions import einsum
from adcc.misc import cached_member_function
from adcc import LazyMp, OneParticleOperator

class qed_mp(LazyMp):
    def __init__(self, hf, omega, qed_hf=True, **kwargs):
        """Lazily evaluated (non-)polaritonic MP object"""
        super().__init__(hf)
        self.get_qed_total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)  # noqa: E501
        self.get_qed_total_dip.oo = hf.get_qed_total_dip(b.oo)
        self.get_qed_total_dip.ov = hf.get_qed_total_dip(b.ov)
        self.get_qed_total_dip.vv = hf.get_qed_total_dip(b.vv)
        self.omega = float(omega)
        self.qed_hf = qed_hf


    @cached_member_function
    def qed_t1_df(self, space):
        """
        qed_t1 amplitude times (df+omega)
        """
        total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
        if space == b.oo:
            total_dip.oo = self.get_qed_total_dip.oo
            return total_dip.oo
        elif space == b.ov:
            total_dip.ov = self.get_qed_total_dip.ov
            return total_dip.ov
        elif space == b.vv:
            total_dip.vv = self.get_qed_total_dip.vv
            return total_dip.vv

    @cached_member_function
    def qed_t1(self, space):
        """
        Return new electronic singly excited amplitude in the first
        order correction to the wavefunction for qed
        """
        if space != b.ov:
            raise NotImplementedError("qed_t1 term not implemented "
                                      f"for space {space}.")
        #omega = self.omega
        return self.qed_t1_df(b.ov) / (self.df(b.ov) + self.omega)

    @cached_member_function
    def qed_t0_df(self, space):
        """
        qed_t0 amplitude times df
        """
        total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
        total_dip.oo = self.get_qed_total_dip.oo
        total_dip.ov = self.get_qed_total_dip.ov
        total_dip.vv = self.get_qed_total_dip.vv
        if space == b.ov:
            occ_sum = einsum("ka,ki->ia", total_dip.ov, total_dip.oo)
            virt_sum = einsum("ac,ic->ia", total_dip.vv, total_dip.ov)
        elif space == b.oo:
            occ_sum = einsum("ki,kj->ij", total_dip.oo, total_dip.oo)
            virt_sum = einsum("ic,jc->ij", total_dip.ov, total_dip.ov)
        elif space == b.vv:
            occ_sum = einsum("ka,kb->ab", total_dip.ov, total_dip.ov)
            virt_sum = einsum("ac,bc->ab", total_dip.vv, total_dip.vv)
        return occ_sum - virt_sum

    @cached_member_function
    def qed_t0(self, space):
        """
        Return second new electronic singly excited amplitude in the first
        order correction to the wavefunction for qed from the standard
        HF reference
        """
        if space != b.ov:
            raise NotImplementedError("qed_t0 term not implemented "
                                      f"for space {space}.")
        return self.qed_t0_df(b.ov) / self.df(b.ov)


    def qed_energy_correction(self, level=2):
        """Determines the polaritonic correction to the MP energy."""
        if level >= 3:
            raise NotImplementedError("polaritonic MP energy is not implemented for third order")
        if self.has_core_occupied_space:
            raise NotImplementedError("polaritonic MP energy not implemented for cvs")
        if level == 1:
            if not self.qed_hf:
                qed_mp1_additional_terms = [(0.5, total_dip.ov)]
                return sum(pref * lambda_dip.dot(lambda_dip)
                           for pref, lambda_dip in qed_mp1_additional_terms)
            else:
                return 0
        if level == 2:
            total_dip = OneParticleOperator(self.mospaces, is_symmetric=True)
            omega = self.omega #ReferenceState.get_qed_omega(hf)
            total_dip.ov = self.get_qed_total_dip.ov
            qed_terms = [(omega / 2, total_dip.ov, self.qed_t1(b.ov))]
            qed_mp2_correction_1 = sum(
                -pref * lambda_dip.dot(qed_t)
                for pref, lambda_dip, qed_t in qed_terms
            )
            if self.qed_hf:
                qed_mp2_correction = qed_mp2_correction_1
            else:
                qed_terms_0 = [(1.0, self.qed_t0(b.ov), self.qed_t0_df(b.ov))]
                qed_mp2_correction_0 = sum(
                    -0.25 * pref * ampl_t0.dot(ampl_t0_df)
                    for pref, ampl_t0, ampl_t0_df in qed_terms_0
                )
                qed_mp2_correction = qed_mp2_correction_1 +\
                    qed_mp2_correction_0
            return qed_mp2_correction
    

    def qed_energy(self, level=2):
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory for QED MP.
        """

        # Accumulator for all energy terms
        energies = [self.energy(level)]
        for il in range(1, level + 1):
            energies.append(self.qed_energy_correction(il))
        return sum(energies)
    

