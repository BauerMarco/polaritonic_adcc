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
from .qed_mp import qed_mp
import numpy as np
from adcc.misc import cached_property
from adcc.functions import einsum, direct_sum
import adcc.block as b
from .solver.diis import diis
import warnings


def update_t2_amplitude(ampl_in, hf, df2, omega, qed_t1_df, qed_t1_ampl):
    return (
        hf.oovv
        + 0.5 * einsum('klij,klab->ijab', hf.oooo, ampl_in)
        + 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, ampl_in)
        - einsum('kaic,jkbc->ijab', hf.ovov, ampl_in)
        + einsum('kajc,ikbc->ijab', hf.ovov, ampl_in)
        + einsum('kbic,jkac->ijab', hf.ovov, ampl_in)
        - einsum('kbjc,ikac->ijab', hf.ovov, ampl_in)
        - einsum("ia,jb->ijab", qed_t1_df, qed_t1_ampl).antisymmetrise(0, 1).antisymmetrise(2, 3) * np.sqrt(omega / 2)
    ) / df2

def update_qed_t1_amplitude(qed_t1_ampl_in, t2_ampl, qed_t1_df, hf, df_omega, omega):
    return (
        - qed_t1_df * np.sqrt(omega / 2)  # can be cashed
        - einsum("kc,ikac->ia", qed_t1_df, t2_ampl) * np.sqrt(omega / 2) * 0.5
        - einsum("kaic,kc->ia", hf.ovov, qed_t1_ampl_in)
    ) / df_omega


class qed_ucc(qed_mp):
    def __init__(self, hf, omega, qed_hf=True, conv_tol=1e-5, max_iter=100,
                 max_subspace=None):  # last parameter is an ugly dummy parameter, so **solverargs can be provided here
        super().__init__(hf, omega, qed_hf=qed_hf)
        if self.has_core_occupied_space:
            raise NotImplementedError("UCC is not implemented for CVS.")
        self.conv_tol = conv_tol
        self.max_iter = max_iter

    @cached_property
    def t_ampls(self):
        hf = self.reference_state
        t2 = -1.0 * super().t2(b.oovv).evaluate()  # change sign due to definition of t2 amplitude
        df2 = -1.0 * direct_sum("-i-j+a+b->ijab",
                                hf.foo.diagonal(), hf.foo.diagonal(),
                                hf.fvv.diagonal(), hf.fvv.diagonal())
        df2.evaluate()
        diis_handler_t2 = diis()
        qed_t1 = super().qed_t1(b.ov).evaluate()  # this amplitude doesn't require a sign change
        df_omega = self.df(b.ov) + self.omega
        qed_t1_df = self.qed_t1_df(b.ov)
        diis_handler_qed_t1 = diis()
        print("Niter", "|res_total|", "|res_t2|", "|res_qed_t1|")
        for i in range(self.max_iter):
            # TODO: iterating both amplitudes one step each until both converge
            # seems to be only a good idea, if they converge very robust and
            # monotonous...still the currently applied solver is very crude
            # t2 ampl
            t2old = - 1.0 * t2
            t2new = update_t2_amplitude(t2, hf, df2, self.omega, qed_t1_df, qed_t1).evaluate()
            t2, rnorm_t2 = diis_handler_t2.do_iteration(t2, t2new)

            # qed_t1 ampl
            # here we use t2old to keep the iteration number for the amplitude inputs equal
            qed_t1_new = update_qed_t1_amplitude(qed_t1, t2old, qed_t1_df, hf, df_omega, self.omega).evaluate()
            qed_t1, rnorm_qed_t1 = diis_handler_qed_t1.do_iteration(qed_t1, qed_t1_new)

            # convergence criteria
            rnorm = np.sqrt(rnorm_t2**2 + rnorm_qed_t1**2)
            print(i, rnorm, rnorm_t2, rnorm_qed_t1)
            if rnorm < self.conv_tol:
                # switch sign for compatibility
                return (-1.0 * t2, qed_t1)
        raise ValueError("t amplitudes not converged.")

    def t2(self, space):
        if space != b.oovv:
            raise NotImplementedError("QED-UCC2 t2-amplitudes only implemented for oovv block")
        return self.t_ampls[0]

    def qed_t1(self, space):
        if space != b.ov:
            raise NotImplementedError("QED-UCC2 qed_t1-amplitudes only implemented for ov block")
        return self.t_ampls[1]

