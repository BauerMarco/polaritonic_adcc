#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
from math import sqrt
from collections import namedtuple

from adcc import block as b
from adcc.functions import direct_sum, einsum, zeros_like
from adcc.Intermediates import Intermediates, register_as_intermediate
from adcc.AmplitudeVector import AmplitudeVector
#from adcc.ReferenceState import ReferenceState
from libadcc import set_lt_scalar
#from libadcc import set_from_ndarray
import numpy as np

#__all__ = ["block"]

# TODO One thing one could still do to improve timings is implement a "fast einsum"
#      that does not call opt_einsum, but directly dispatches to libadcc. This could
#      lower the call overhead in the applies for the cases where we have only a
#      trivial einsum to do. For the moment I'm not convinced that is worth the
#      effort ... I suppose it only makes a difference for the cheaper ADC variants
#      (ADC(0), ADC(1), CVS-ADC(0-2)-x), but then on the other hand they are not
#      really so much our focus.


#
# Dispatch routine
#
"""
`apply` is a function mapping an AmplitudeVector to the contribution of this
block to the result of applying the ADC matrix. `diagonal` is an `AmplitudeVector`
containing the expression to the diagonal of the ADC matrix from this block.
"""
AdcBlock = namedtuple("AdcBlock", ["apply", "diagonal"])


def qed_block(ground_state, spaces, order, variant=None, intermediates=None):
    """
    Gets ground state, potentially intermediates, spaces (ph, pphh and so on)
    and the perturbation theory order for the block,
    variant is "cvs" or sth like that.

    It is assumed largely, that CVS is equivalent to mp.has_core_occupied_space,
    while one would probably want in the long run that one can have an "o2" space,
    but not do CVS.
    """
    if isinstance(variant, str):
        variant = [variant]
    elif variant is None:
        variant = []
    reference_state = ground_state.reference_state
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if ground_state.has_core_occupied_space and "cvs" not in variant:
        raise ValueError("Cannot run a general (non-core-valence approximated) "
                         "ADC method on top of a ground state with a "
                         "core-valence separation.")
    if not ground_state.has_core_occupied_space and "cvs" in variant:
        raise ValueError("Cannot run a core-valence approximated ADC method on "
                         "top of a ground state without a "
                         "core-valence separation.")

    fn = "_".join(["block"] + variant + spaces + [str(order)])

    if fn not in globals():
        raise ValueError("Could not dispatch: "
                         f"spaces={spaces} order={order} variant=variant")
    return globals()[fn](reference_state, ground_state, intermediates)


# For QED-ADC (up to double photon dispersion) we build the matrix as follows:
# elec              phot_couple         phot_couple_edge
# elec_couple       phot                phot_couple_inner
# elec_couple_edge  elec_couple_inner   phot2
# where each block is a "standard" ADC matrix itself, including the groundstate
# and the groundstate couplings. However, the gs_ph and gs_gs blocks are merged
# into the ph_ph and ph_gs blocks, respectively, since we calculate matrix vector
# products anyway. Note, that the purely electronic groundstate never appears, since
# it is always zero.


#
# 0th order gs blocks
#

def block_ph_gs_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_ph_gs_0_phot(hf, mp, intermediates):

    def apply(ampl):
        return AmplitudeVector(gs1=mp.omega * ampl.gs1)
    return AdcBlock(apply, AmplitudeVector(gs1=set_lt_scalar(mp.omega)))
    #return AdcBlock(apply, AmplitudeVector(gs1=set_from_ndarray(np.array([mp.omega]))))


def block_ph_gs_0_phot2(hf, mp, intermediates):

    def apply(ampl):
        return AmplitudeVector(gs2=2 * mp.omega * ampl.gs2)
    return AdcBlock(apply, AmplitudeVector(gs2=set_lt_scalar(2 * mp.omega)))
    #return AdcBlock(apply, AmplitudeVector(gs2=set_from_ndarray(np.array([2 * mp.omega]))))


block_ph_gs_0_couple = block_ph_gs_0_phot_couple =\
    block_ph_gs_0_couple_edge = block_ph_gs_0_phot_couple_edge =\
    block_ph_gs_0_couple_inner = block_ph_gs_0_phot_couple_inner =\
    block_ph_gs_0

#
# 0th order main
#

def block_ph_ph_0(hf, mp, intermediates):
    diagonal = AmplitudeVector(ph=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                                hf.foo.diagonal()))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, hf.fvv)
            - einsum("IJ,Ja->Ia", hf.foo, ampl.ph)
        ))

    return AdcBlock(apply, diagonal)


def block_ph_ph_0_couple(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_ph_ph_0_phot_couple = block_ph_ph_0_phot_couple_edge =\
    block_ph_ph_0_phot_couple_inner = block_ph_ph_0_couple_edge =\
    block_ph_ph_0_couple_inner = block_ph_ph_0_couple


def block_ph_ph_0_phot(hf, mp, intermediates):
    diagonal = AmplitudeVector(ph1=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                                hf.foo.diagonal()))

    def apply(ampl):
        return AmplitudeVector(ph1=(
            + einsum("ib,ab->ia", ampl.ph1, hf.fvv)
            - einsum("IJ,Ja->Ia", hf.foo, ampl.ph1)
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_0_phot2(hf, mp, intermediates):
    diagonal = AmplitudeVector(ph2=direct_sum("a-i->ia", hf.fvv.diagonal(),
                                              hf.foo.diagonal()))

    def apply(ampl):
        return AmplitudeVector(ph2=(
            + einsum("ib,ab->ia", ampl.ph2, hf.fvv)
            - einsum("IJ,Ja->Ia", hf.foo, ampl.ph2)
        ))
    return AdcBlock(apply, diagonal)


def diagonal_pphh_pphh_0(hf, to_add):
    # Note: adcman similarly does not symmetrise the occupied indices
    res = direct_sum("-i-J+a+b->iJab",
                        hf.foo.diagonal() + to_add, hf.foo.diagonal() + to_add,
                        hf.fvv.diagonal() + to_add, hf.fvv.diagonal() + to_add)
    return res.symmetrise(2, 3)


def block_pphh_pphh_0(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh).antisymmetrise(0, 1)
        ))
    return AdcBlock(apply, AmplitudeVector(pphh=diagonal_pphh_pphh_0(hf, 0)))


def block_pphh_pphh_0_couple(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_pphh_pphh_0_phot_couple = block_pphh_pphh_0_phot_couple_edge =\
    block_pphh_pphh_0_phot_couple_inner = block_pphh_pphh_0_couple_edge =\
    block_pphh_pphh_0_couple_inner = block_pphh_pphh_0_couple


def block_pphh_pphh_0_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh1=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh1, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh1).antisymmetrise(0, 1)
            + mp.omega * ampl.pphh1
        ))
    return AdcBlock(apply, AmplitudeVector(pphh1=diagonal_pphh_pphh_0(hf, mp.omega)))


def block_pphh_pphh_0_phot2(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh2=(
            + 2 * einsum("ijac,bc->ijab", ampl.pphh2, hf.fvv).antisymmetrise(2, 3)
            - 2 * einsum("ik,kjab->ijab", hf.foo, ampl.pphh2).antisymmetrise(0, 1)
            + 2 * mp.omega * ampl.pphh2
        ))
    return AdcBlock(apply, AmplitudeVector(pphh2=diagonal_pphh_pphh_0(hf, 2 * mp.omega)))


#
# 0th order coupling
#

def block_ph_pphh_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


def block_pphh_ph_0(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_pphh_ph_0_couple = block_pphh_ph_0_phot_couple = block_pphh_ph_0_phot =\
    block_pphh_ph_0_couple_edge = block_pphh_ph_0_couple_inner =\
    block_pphh_ph_0_phot_couple_edge = block_pphh_ph_0_phot_couple_inner =\
    block_pphh_ph_0_phot2 = block_pphh_ph_0

block_ph_pphh_0_couple = block_ph_pphh_0_phot_couple = block_ph_pphh_0_phot =\
    block_ph_pphh_0_couple_edge = block_ph_pphh_0_couple_inner =\
    block_ph_pphh_0_phot_couple_edge = block_ph_pphh_0_phot_couple_inner =\
    block_ph_pphh_0_phot2 = block_ph_pphh_0


#
# 1st order gs blocks
#

block_ph_gs_1_phot = block_ph_gs_0_phot
block_ph_gs_1_phot2 = block_ph_gs_0_phot2


block_ph_gs_1_phot_couple = block_ph_gs_1_phot_couple_edge =\
    block_ph_gs_1_couple_edge = block_ph_gs_1_phot_couple_inner =\
    block_ph_gs_1 = block_ph_gs_0


def block_ph_gs_1_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(gs1=(
            (-1) * sqrt(0.5 * mp.omega) * mp.qed_t1_df(b.ov).dot(ampl.ph)
        ))
    return AdcBlock(apply, 0)


def block_ph_gs_1_couple_inner(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(gs2=(
            (-1) * sqrt(mp.omega) * mp.qed_t1_df(b.ov).dot(ampl.ph1)
        ))
    return AdcBlock(apply, 0)


#
# 1st order main
#

def block_ph_ph_1(hf, mp, intermediates):
    if not hf.qed_hf:
        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())  # order 0
            - einsum("IaIa->Ia", hf.ovov)  # order 1
            + (1 / 2) * direct_sum("i-a->ia", einsum("ii->i", mp.qed_t0_df(b.oo)),
                                   einsum("aa->a", mp.qed_t0_df(b.vv)))
        ))

        def apply(ampl):
            return AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", hf.foo, ampl.ph)     # 0
                - einsum("JaIb,Jb->Ia", hf.ovov, ampl.ph)  # 1
                + (1 / 2) * einsum("ij,ja->ia", mp.qed_t0_df(b.oo), ampl.ph)
                - (1 / 2) * einsum("ib,ab->ia", ampl.ph, mp.qed_t0_df(b.vv))
            ))
    else:
        diagonal = AmplitudeVector(ph=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())  # 0
            - einsum("IaIa->Ia", hf.ovov)  # 1
        ))

        def apply(ampl):
            return AmplitudeVector(ph=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", hf.foo, ampl.ph)     # 0
                - einsum("JaIb,Jb->Ia", hf.ovov, ampl.ph)  # 1
            ))
    return AdcBlock(apply, diagonal)


block_cvs_ph_ph_1 = block_ph_ph_1


def block_ph_ph_1_phot(hf, mp, intermediates):
    if not hf.qed_hf:

        diagonal = AmplitudeVector(ph1=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())  # order 0
            - einsum("IaIa->Ia", hf.ovov)  # order 1
            + (1 / 2) * direct_sum("i-a->ia", einsum("ii->i", mp.qed_t0_df(b.oo)),
                                   einsum("aa->a", mp.qed_t0_df(b.vv)))
            + intermediates.delta_ia_omega
        ))

        def apply(ampl):
            return AmplitudeVector(ph1=(                 # PT order
                + einsum("ib,ab->ia", ampl.ph1, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", hf.foo, ampl.ph1)     # 0
                - einsum("JaIb,Jb->Ia", hf.ovov, ampl.ph1)  # 1
                + (1 / 2) * einsum("ij,ja->ia", mp.qed_t0_df(b.oo), ampl.ph1)
                - (1 / 2) * einsum("ib,ab->ia", ampl.ph1, mp.qed_t0_df(b.vv))
                + mp.omega * ampl.ph1
            ))
    else:
        diagonal = AmplitudeVector(ph1=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())  # 0
            - einsum("IaIa->Ia", hf.ovov)  # 1
            + intermediates.delta_ia_omega
        ))

        def apply(ampl):
            return AmplitudeVector(ph1=(                  # PT order
                + einsum("ib,ab->ia", ampl.ph1, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", hf.foo, ampl.ph1)     # 0
                - einsum("JaIb,Jb->Ia", hf.ovov, ampl.ph1)  # 1
                + mp.omega * ampl.ph1
            ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_1_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph1=(
            sqrt(mp.omega / 2) * (
                - einsum("ib,ab->ia", ampl.ph, mp.qed_t1_df(b.vv))
                + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph))
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_1_phot_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            sqrt(mp.omega / 2) * (
                - einsum("ib,ab->ia", ampl.ph1, mp.qed_t1_df(b.vv))
                + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph1)
                - mp.qed_t1_df(b.ov) * ampl.gs1.to_ndarray())
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_1_couple_edge(hf, mp, intermediates):
    return AdcBlock(lambda ampl: 0, 0)


block_ph_ph_1_phot_couple_edge = block_ph_ph_1_couple_edge


def block_ph_ph_1_couple_inner(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph2=(
            sqrt(mp.omega) * (- einsum("ib,ab->ia", ampl.ph1, mp.qed_t1_df(b.vv))
                            + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph1))
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_1_phot_couple_inner(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph1=(
            sqrt(mp.omega) * (- einsum("ib,ab->ia", ampl.ph2, mp.qed_t1_df(b.vv))
                            + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph2)
                            - mp.qed_t1_df(b.ov) * ampl.gs2.to_ndarray())
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_1_phot2(hf, mp, intermediates):
    if not hf.qed_hf:

        diagonal = AmplitudeVector(ph2=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())  # order 0
            - einsum("IaIa->Ia", hf.ovov)  # order 1
            + (1 / 2) * direct_sum("i-a->ia", einsum("ii->i", mp.qed_t0_df(b.oo)),
                                   einsum("aa->a", mp.qed_t0_df(b.vv)))
            + intermediates.delta_ia_omega * 2
        ))

        def apply(ampl):
            return AmplitudeVector(ph2=(                  # PT order
                + einsum("ib,ab->ia", ampl.ph2, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", hf.foo, ampl.ph2)     # 0
                - einsum("JaIb,Jb->Ia", hf.ovov, ampl.ph2)  # 1
                + (1 / 2) * einsum("ij,ja->ia", mp.qed_t0_df(b.oo), ampl.ph2)
                - (1 / 2) * einsum("ib,ab->ia", ampl.ph2, mp.qed_t0_df(b.vv))
                + 2 * mp.omega * ampl.ph2
            ))
    else:
        diagonal = AmplitudeVector(ph2=(
            + direct_sum("a-i->ia", hf.fvv.diagonal(), hf.foo.diagonal())  # 0
            - einsum("IaIa->Ia", hf.ovov)  # 1
            + intermediates.delta_ia_omega * 2
        ))

        def apply(ampl):
            return AmplitudeVector(ph2=(                  # PT order
                + einsum("ib,ab->ia", ampl.ph2, hf.fvv)  # 0
                - einsum("IJ,Ja->Ia", hf.foo, ampl.ph2)     # 0
                - einsum("JaIb,Jb->Ia", hf.ovov, ampl.ph2)  # 1
                + 2 * mp.omega * ampl.ph2
            ))
    return AdcBlock(apply, diagonal)


#
# 1st order coupling
#

def block_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh)
            + einsum("ijbc,jabc->ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph1=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh1)
            + einsum("ijbc,jabc->ia", ampl.pphh1, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_phot2(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph2=(
            + einsum("jkib,jkab->ia", hf.ooov, ampl.pphh2)
            + einsum("ijbc,jabc->ia", ampl.pphh2, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_cvs_ph_pphh_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(
            + sqrt(2) * einsum("jKIb,jKab->Ia", hf.occv, ampl.pphh)
            - 1 / sqrt(2) * einsum("jIbc,jabc->Ia", ampl.pphh, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + einsum("ic,jcab->ijab", ampl.ph, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_phot(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh1=(
            + einsum("ic,jcab->ijab", ampl.ph1, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_phot2(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh2=(
            + einsum("ic,jcab->ijab", ampl.ph2, hf.ovvv).antisymmetrise(0, 1)
            - einsum("ijka,kb->ijab", hf.ooov, ampl.ph2).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_cvs_pphh_ph_1(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            + sqrt(2) * einsum("jIKb,Ka->jIab",
                               hf.occv, ampl.ph).antisymmetrise(2, 3)
            - 1 / sqrt(2) * einsum("Ic,jcab->jIab", ampl.ph, hf.ovvv)
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph1=(
            2 * sqrt(mp.omega / 2) * einsum("kc,ikac->ia", mp.qed_t1_df(b.ov), ampl.pphh)  # noqa: E501
        ))
    return AdcBlock(apply, 0)


def block_ph_pphh_1_couple_inner(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph2=(
            2 * sqrt(mp.omega) * einsum("kc,ikac->ia", mp.qed_t1_df(b.ov), ampl.pphh1)
        ))
    return AdcBlock(apply, 0)


block_pphh_ph_1_couple = block_pphh_ph_1_couple_inner =\
    block_pphh_ph_1_phot_couple_edge = block_pphh_ph_1_couple_edge =\
    block_pphh_ph_0_couple

block_ph_pphh_1_phot_couple = block_ph_pphh_1_phot_couple_inner =\
    block_ph_pphh_1_phot_couple_edge = block_ph_pphh_1_couple_edge =\
    block_ph_pphh_0_phot_couple


def block_pphh_ph_1_phot_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh=(
            2 * sqrt(mp.omega / 2) * einsum(
                "jb,ia->ijab", mp.qed_t1_df(b.ov),
                ampl.ph1).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


def block_pphh_ph_1_phot_couple_inner(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(pphh1=(
            2 * sqrt(mp.omega) * einsum(
                "jb,ia->ijab", mp.qed_t1_df(b.ov),
                ampl.ph2).antisymmetrise(0, 1).antisymmetrise(2, 3)
        ))
    return AdcBlock(apply, 0)


#
# 2nd order gs blocks
#

block_ph_gs_2_phot_couple = block_ph_gs_2_phot_couple_edge =\
    block_ph_gs_2_couple_edge = block_ph_gs_2_phot_couple_inner =\
    block_ph_gs_2 = block_ph_gs_0


def block_ph_gs_2_couple(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(gs1=(sqrt(mp.omega / 2) * einsum(
            "jkbc,kc->jb", mp.t2oo, mp.qed_t1_df(b.ov)).dot(ampl.ph)
            - sqrt(0.5 * mp.omega) * mp.qed_t1_df(b.ov).dot(ampl.ph)))  # 1. order
    return AdcBlock(apply, 0)


block_ph_gs_2_phot = block_ph_gs_0_phot
block_ph_gs_2_phot2 = block_ph_gs_0_phot2


def block_ph_gs_2_couple_inner(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(gs2=(sqrt(mp.omega) * einsum(
            "jkbc,kc->jb", mp.t2oo, mp.qed_t1_df(b.ov)).dot(ampl.ph1)
            - sqrt(mp.omega) * mp.qed_t1_df(b.ov).dot(ampl.ph1)))  # 1. order
    return AdcBlock(apply, 0)


#
# 2nd order main
#

def block_ph_ph_2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2

    term_t2_eri = intermediates.term_t2_eri

    qed_i1 = intermediates.adc2_qed_i1
    qed_i2 = intermediates.adc2_qed_i2
    diagonal = AmplitudeVector(ph=(
        + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
        - einsum("IaIa->Ia", hf.ovov)
        - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
        + (-mp.omega / 2) * (
            - direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
            + (1 / 2) * 2 * einsum(
                "ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)
            )
        )
    ))

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph, i1)
            - einsum("ij,ja->ia", i2, ampl.ph)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph)    # 1
            - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph)  # 2
            + (-mp.omega / 2) * (
                - einsum("ib,ab->ia", ampl.ph, qed_i1)
                - einsum("ij,ja->ia", qed_i2, ampl.ph)
                + (1 / 2) * (
                    mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph)
                    + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph)
                )
            )
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_couple(hf, mp, intermediates):
    qed_i1 = intermediates.adc2_qed_couple_i1
    qed_i2 = intermediates.adc2_qed_couple_i2

    def apply(ampl):
        return AmplitudeVector(ph1=(
            + einsum("ib,ab->ia", ampl.ph, qed_i1)
            + einsum("ij,ja->ia", qed_i2, ampl.ph)
            + sqrt(mp.omega / 2) * (
                + einsum("ka,jkib,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph)
                + einsum("ic,jabc,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph))
            + sqrt(mp.omega / 2) * (
                - einsum("ib,ab->ia", ampl.ph, mp.qed_t1_df(b.vv))  # 1. order
                + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph))  # 1. order
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_2_phot_couple(hf, mp, intermediates):
    gs_part = intermediates.adc2_qed_ph_ph_2_phot_couple_gs_part
    qed_i1 = intermediates.adc2_qed_phot_couple_i1
    qed_i2 = intermediates.adc2_qed_phot_couple_i2

    def apply(ampl):
        return AmplitudeVector(ph=(
            + einsum("ib,ab->ia", ampl.ph1, qed_i1)
            + einsum("ij,ja->ia", qed_i2, ampl.ph1)
            + sqrt(mp.omega / 2) * (
                + einsum("kb,ikja,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph1)
                + einsum("jc,ibac,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph1))
            + gs_part * ampl.gs1.to_ndarray()
            + sqrt(mp.omega / 2) * (
                - einsum("ib,ab->ia", ampl.ph1, mp.qed_t1_df(b.vv))  # 1. order
                + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph1)  # 1. order
                - mp.qed_t1_df(b.ov) * ampl.gs1.to_ndarray())        # 1. order
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_2_phot(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2

    term_t2_eri = intermediates.term_t2_eri

    qed_i1 = intermediates.adc2_qed_i1
    qed_i2 = intermediates.adc2_qed_i2

    diagonal = AmplitudeVector(ph1=(
        + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
        - einsum("IaIa->Ia", hf.ovov)
        - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
        + (-mp.omega / 2) * 2 * (
            - direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
            + einsum("ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))
        + intermediates.delta_ia_omega  # 0. order
    ))

    def apply(ampl):
        return AmplitudeVector(ph1=(
            + einsum("ib,ab->ia", ampl.ph1, i1)
            - einsum("ij,ja->ia", i2, ampl.ph1)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph1)    # 1
            - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph1)  # 2
            + (-mp.omega / 2) * 2 * (
                - einsum("ib,ab->ia", ampl.ph1, qed_i1)
                - einsum("ij,ja->ia", qed_i2, ampl.ph1)
                + 0.5 * (mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph1)
                         + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph1)))
            + mp.omega * ampl.ph1  # 0. order
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_phot2(hf, mp, intermediates):
    i1 = intermediates.adc2_i1
    i2 = intermediates.adc2_i2

    term_t2_eri = intermediates.term_t2_eri

    qed_i1 = intermediates.adc2_qed_i1
    qed_i2 = intermediates.adc2_qed_i2

    diagonal = AmplitudeVector(ph2=(
        + direct_sum("a-i->ia", i1.diagonal(), i2.diagonal())
        - einsum("IaIa->Ia", hf.ovov)
        - einsum("ikac,ikac->ia", mp.t2oo, hf.oovv)
        + (-mp.omega / 2) * 3 * (
            - direct_sum("a+i->ia", qed_i1.diagonal(), qed_i2.diagonal())
            + einsum("ia,ia->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))
        + intermediates.delta_ia_omega * 2  # 0. order
    ))

    def apply(ampl):
        return AmplitudeVector(ph2=(
            + einsum("ib,ab->ia", ampl.ph2, i1)
            - einsum("ij,ja->ia", i2, ampl.ph2)
            - einsum("jaib,jb->ia", hf.ovov, ampl.ph2)    # 1
            - 0.5 * einsum("ikac,kc->ia", term_t2_eri, ampl.ph2)  # 2
            + (-mp.omega / 2) * 3 * (
                - einsum("ib,ab->ia", ampl.ph2, qed_i1)
                - einsum("ij,ja->ia", qed_i2, ampl.ph2)
                + 0.5 * (mp.qed_t1(b.ov) * mp.qed_t1_df(b.ov).dot(ampl.ph2)
                         + mp.qed_t1_df(b.ov) * mp.qed_t1(b.ov).dot(ampl.ph2)))
            + 2 * mp.omega * ampl.ph2  # 0. order
        ))
    return AdcBlock(apply, diagonal)


def block_ph_ph_2_couple_inner(hf, mp, intermediates):
    qed_i1 = intermediates.adc2_qed_couple_i1
    qed_i2 = intermediates.adc2_qed_couple_i2

    def apply(ampl):
        return AmplitudeVector(ph2=(
            + sqrt(2) * einsum("ib,ab->ia", ampl.ph1, qed_i1)
            + sqrt(2) * einsum("ij,ja->ia", qed_i2, ampl.ph1)
            + sqrt(mp.omega) * (
                + einsum("ka,jkib,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph1)
                + einsum("ic,jabc,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph1))
            + sqrt(mp.omega) * (
                - einsum("ib,ab->ia", ampl.ph1, mp.qed_t1_df(b.vv))  # 1. order
                + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph1))  # 1. order
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_2_phot_couple_inner(hf, mp, intermediates):
    gs_part = intermediates.adc2_qed_ph_ph_2_phot_couple_inner_gs_part
    qed_i1 = intermediates.adc2_qed_phot_couple_i1
    qed_i2 = intermediates.adc2_qed_phot_couple_i2

    def apply(ampl):
        return AmplitudeVector(ph1=(
            + sqrt(2) * einsum("ib,ab->ia", ampl.ph2, qed_i1)
            + sqrt(2) * einsum("ij,ja->ia", qed_i2, ampl.ph2)
            + sqrt(mp.omega) * (
                + einsum("kb,ikja,jb->ia", mp.qed_t1(b.ov), hf.ooov, ampl.ph2)
                + einsum("jc,ibac,jb->ia", mp.qed_t1(b.ov), hf.ovvv, ampl.ph2))
            + gs_part * ampl.gs2.to_ndarray()
            + sqrt(mp.omega) * (
                - einsum("ib,ab->ia", ampl.ph2, mp.qed_t1_df(b.vv))  # 1. order
                + einsum("ij,ja->ia", mp.qed_t1_df(b.oo), ampl.ph2)  # 1. order
                - mp.qed_t1_df(b.ov) * ampl.gs2.to_ndarray())        # 1. order
        ))
    return AdcBlock(apply, 0)


def block_ph_ph_2_couple_edge(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph2=(- (mp.omega / 2) * sqrt(2) * (
            einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph
            - einsum("ka,kb,ib->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph)
            - einsum("ic,jc,ja->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph)
        )))
    return AdcBlock(apply, 0)


def block_ph_ph_2_phot_couple_edge(hf, mp, intermediates):
    def apply(ampl):
        return AmplitudeVector(ph=(- (mp.omega / 2) * sqrt(2) * (
            einsum("kc,kc->", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)) * ampl.ph2
            - einsum("kb,ka,ib->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
            - einsum("jc,ic,ja->ia", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov), ampl.ph2)
        )))
    return AdcBlock(apply, 0)

#
# Intermediates
#

@register_as_intermediate
def adc2_i1(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.fvv term.
    return hf.fvv + 0.5 * einsum("ijac,ijbc->ab", mp.t2oo, hf.oovv).symmetrise()


@register_as_intermediate
def adc2_i2(hf, mp, intermediates):
    # This definition differs from libadc. It additionally has the hf.foo term.
    return hf.foo - 0.5 * einsum("ikab,jkab->ij", mp.t2oo, hf.oovv).symmetrise()


# qed intermediates for adc2, without the factor of (omega/2),
# which is added in the actual matrix builder
@register_as_intermediate
def adc2_qed_i1(hf, mp, intermediates):  # maybe do this with symmetrise
    return (1 / 2) * (einsum("kb,ka->ab", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov))
                      + einsum("ka,kb->ab", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_i2(hf, mp, intermediates):  # maybe do this with symmetrise
    return (1 / 2) * (einsum("jc,ic->ij", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov))
                      + einsum("ic,jc->ij", mp.qed_t1(b.ov), mp.qed_t1_df(b.ov)))


@register_as_intermediate
def qed_adc2_ph_gs_intermediate(hf, mp, intermediates):
    return (0.5 * einsum("jkib,jkab->ia", hf.ooov, mp.t2oo)
            + 0.5 * einsum("ijbc,jabc->ia", mp.t2oo, hf.ovvv))


@register_as_intermediate
def term_t2_eri(hf, mp, intermediates):
    return (einsum("ijab,jkbc->ikac", mp.t2oo, hf.oovv)
            + einsum("ijab,jkbc->ikac", hf.oovv, mp.t2oo))


@register_as_intermediate
def adc2_qed_ph_ph_2_phot_couple_gs_part(hf, mp, intermediates):
    return sqrt(mp.omega / 2) * (einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_ph_ph_2_phot_couple_inner_gs_part(hf, mp, intermediates):
    return sqrt(mp.omega) * (einsum("ikac,kc->ia", mp.t2oo, mp.qed_t1_df(b.ov)))


@register_as_intermediate
def adc2_qed_couple_i1(hf, mp, intermediates):
    return (sqrt(mp.omega / 2) * (einsum("kc,kacb->ab", mp.qed_t1(b.ov), hf.ovvv)))


@register_as_intermediate
def adc2_qed_couple_i2(hf, mp, intermediates):
    return (sqrt(mp.omega / 2) * (einsum("kc,kjic->ij", mp.qed_t1(b.ov), hf.ooov)))


@register_as_intermediate
def adc2_qed_phot_couple_i1(hf, mp, intermediates):
    return (sqrt(mp.omega / 2) * (einsum("kc,kbca->ab", mp.qed_t1(b.ov), hf.ovvv)))


@register_as_intermediate
def adc2_qed_phot_couple_i2(hf, mp, intermediates):
    return (sqrt(mp.omega / 2) * (einsum("kc,kijc->ij", mp.qed_t1(b.ov), hf.ooov)))


@register_as_intermediate
def delta_ia_omega(hf, mp, intermediates):
    # Build two Kronecker deltas
    d_oo = zeros_like(hf.foo)
    d_vv = zeros_like(hf.fvv)
    d_oo.set_mask("ii", 1.0)
    d_vv.set_mask("aa", 1.0)

    return einsum("ii,aa->ia", d_oo, d_vv) * mp.omega
