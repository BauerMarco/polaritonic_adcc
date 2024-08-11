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
from adcc.LazyMp import LazyMp
#from adcc.AdcMethod import AdcMethod
from adcc.functions import einsum
#from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleOperator import OneParticleOperator

def s2s_tdm_adc0(mp, amplitude_l, amplitude_r):#, intermediates):
    #check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    dm = OneParticleOperator(mp, is_symmetric=False)
    dm.oo = -einsum('ja,ia->ij', ul1, ur1)
    dm.vv = einsum('ia,ib->ab', ul1, ur1)
    return dm


def s2s_tdm_qed_adc2_diag_part(mp, amplitude_l, amplitude_r):#, intermediates):
    #check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_adc0(mp, amplitude_l, amplitude_r)#, intermediates)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph
    p0_oo = dm.oo.evaluate()
    p0_vv = dm.vv.evaluate()

    dm_new = OneParticleOperator(mp, is_symmetric=False)

    dm_new.ov = (
        - einsum("kb,ab->ka", mp.qed_t1(b.ov), p0_vv)
        + einsum("ji,ic->jc", p0_oo, mp.qed_t1(b.ov))
        + ul1.dot(mp.qed_t1(b.ov)) * ur1
    ) / 2

    dm_new.vo = (
        - einsum("kb,ba->ak", mp.qed_t1(b.ov), p0_vv)
        + einsum("ij,ic->cj", p0_oo, mp.qed_t1(b.ov))
        + ur1.dot(mp.qed_t1(b.ov)) * ul1.transpose()
    ) / 2

    return dm_new


def s2s_tdm_qed_adc2_edge_part_couple(mp, amplitude_l, amplitude_r):#, intermediates):
    #check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_adc0(mp, amplitude_l, amplitude_r)#, intermediates)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph
    p0_oo = dm.oo.evaluate()
    p0_vv = dm.vv.evaluate()

    dm_new = OneParticleOperator(mp, is_symmetric=False)

    dm_new.ov = (
        mp.qed_t1(b.ov) * ul1.dot(ur1)
        - einsum("kb,ab->ka", mp.qed_t1(b.ov), p0_vv)
        + einsum("ji,ic->jc", p0_oo, mp.qed_t1(b.ov))
    )

    return dm_new


def s2s_tdm_qed_adc2_edge_part_phot_couple(mp, amplitude_l, amplitude_r):#, intermediates):  # noqa: E501
    #check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_adc0(mp, amplitude_l, amplitude_r)#, intermediates)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph
    p0_oo = dm.oo.evaluate()
    p0_vv = dm.vv.evaluate()

    dm_new = OneParticleOperator(mp, is_symmetric=False)

    dm_new.vo = (
        einsum("ia->ai", mp.qed_t1(b.ov)) * ul1.dot(ur1)
        - einsum("kb,ba->ak", mp.qed_t1(b.ov), p0_vv)
        + einsum("ij,ic->cj", p0_oo, mp.qed_t1(b.ov))
    )

    return dm_new


def s2s_tdm_qed_adc2_ph_pphh_coupl_part(mp, amplitude_l, amplitude_r):#, intermediates):  # noqa: E501
    #check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.ph
    ur2 = amplitude_r.pphh

    dm = OneParticleOperator(mp, is_symmetric=False)

    dm.ov = -2 * einsum("jb,ijab->ia", ul1, ur2)

    return dm


def s2s_tdm_qed_adc2_pphh_ph_phot_coupl_part(mp, amplitude_l, amplitude_r):#, intermediates):  # noqa: E501
    #check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    ul2 = amplitude_l.pphh
    ur1 = amplitude_r.ph

    dm = OneParticleOperator(mp, is_symmetric=False)

    dm.vo = -2 * einsum("ijab,jb->ai", ul2, ur1)

    return dm


DISPATCH = {"adc0": s2s_tdm_adc0,
            "adc1": s2s_tdm_adc0,       # same as ADC(0)
            # The following qed terms are not the actual s2s densities,
            # but those required for the approx function
            "qed_adc2_diag": s2s_tdm_qed_adc2_diag_part,
            "qed_adc2_edge_couple": s2s_tdm_qed_adc2_edge_part_couple,
            "qed_adc2_edge_phot_couple": s2s_tdm_qed_adc2_edge_part_phot_couple,
            "qed_adc2_ph_pphh": s2s_tdm_qed_adc2_ph_pphh_coupl_part,
            "qed_adc2_pphh_ph": s2s_tdm_qed_adc2_pphh_ph_phot_coupl_part,
            #"adc2": s2s_tdm_adc2,
            #"adc2x": s2s_tdm_adc2,      # same as ADC(2)
            }


def qed_npadc_s2s_tdm_terms(key, ground_state, amplitude_from,
                              amplitude_to):#, intermediates=None):
    """
    Compute the state to state transition density matrix and related
    intermediates required for the QED matrix in the truncated state
    space in the MO basis using the intermediate-states representation.

    Parameters
    ----------
    method : str, AdcMethod
        The method to use for the computation (e.g. "adc2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude_from : AmplitudeVector
        The amplitude vector of the state to start from
    amplitude_to : AmplitudeVector
        The amplitude vector of the state to excite to
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    #if not isinstance(method, AdcMethod):
    #    method = AdcMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    #if intermediates is None:
    #    intermediates = Intermediates(ground_state)

    if key not in DISPATCH:
        raise NotImplementedError(f"contribution {key} unknown")
    
    ret = DISPATCH[key](ground_state, amplitude_to, amplitude_from)
    return ret.evaluate()

