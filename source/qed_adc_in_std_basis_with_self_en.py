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
from adcc.OneParticleOperator import product_trace
from adcc.adc_pp import state2state_transition_dm
#import scipy.linalg as sp

def first_order_qed_matrix(state, coupl, frequency):
    """Build polaritonic matrix in a truncated state space on a standard SCF."""
    freq = np.linalg.norm(np.real(frequency))
    imag_freq = np.linalg.norm(np.imag(frequency))

    # we also need mp2 dipole moment for ground state contribution of squared term
    mp2_dip = state.ground_state.dipole_moment(state.property_method.level) - state.ground_state.dipole_moment(level=1)

    # get tdm block 
    tdm_block = state.transition_dipole_moment

    # build s2s block, but only z coordinate
    dipole_integrals = state.operators.electric_dipole
    n_states = len(state.excitation_energy)

    def s2s_density(i, f):
        vec = state.excitation_vector
        return state2state_transition_dm(
            state.method, state.ground_state, vec[i], vec[f]
            )

    s2s_block = np.array([[[product_trace(
                            comp, s2s_density(i, j)
                            )
                            for comp in dipole_integrals]
                            for j in np.arange(n_states)]
                            for i in np.arange(n_states)])

    # dot product coupling strength with tdm and s2s
    tdm_block = np.array([np.dot(i, coupl) for i in tdm_block])
    s2s_block = np.array([[np.dot(s2s_ij, coupl) for s2s_ij in s2s_i]
                                                 for s2s_i in s2s_block])
    mp2_dip = np.array([np.dot(mp2_dip, coupl)])

    # prepare actual dipole terms of cavity Hamiltonian
    # if you wish to neglect the squared dipole contribution, use the terms,
    # which are currently commented out
    #dip_square_s2s = np.zeros_like(s2s_block)
    #dip_square_tdm = np.zeros_like(tdm_block)
    dip_square_s2s = 0.5 * s2s_block ** 2
    dip_square_tdm = 0.5 * tdm_block ** 2
    dip_s2s = np.sqrt(0.5 * freq) * s2s_block
    dip_tdm = np.sqrt(0.5 * freq) * tdm_block
    #mp2_dip_square = np.zeros_like(mp2_dip)
    mp2_dip_square = 0.5 * mp2_dip ** 2
    mp2_dip_off_diag = np.sqrt(0.5 * freq) * mp2_dip

    # build actual matrix blocks
    # now we introduce the actual shifted omega for lossy cavities
    freq += imag_freq
    gs_gs_block = mp2_dip_square
    elec_block = np.diag(state.excitation_energy) + dip_square_s2s
    phot_block = np.diag(state.excitation_energy + freq) + dip_square_s2s
    phot2_block = np.diag(state.excitation_energy + 2 * freq) + dip_square_s2s
    gs_coupl_01 = dip_tdm
    gs_coupl_12 = np.sqrt(2) * dip_tdm
    coupl_01 = dip_s2s
    coupl_12 = np.sqrt(2) * dip_s2s
    gs_coupl = dip_square_tdm
    gs_off_diag_01 = mp2_dip_off_diag
    gs_off_diag_12 = np.sqrt(2) * mp2_dip_off_diag
    omega = np.array([freq])

    # build the full matrix
    # Note, that we also need to include the electronic ground state in the photonic vacuum
    gs = np.concatenate((gs_gs_block, gs_coupl, gs_off_diag_01, gs_coupl_01, np.zeros(n_states + 1)))
    elec = np.vstack((gs_coupl.reshape((1, n_states)), elec_block, gs_coupl_01.reshape((1, n_states)), coupl_01, np.zeros((n_states + 1, n_states))))
    gs_1 = np.concatenate((gs_off_diag_01, gs_coupl_01, omega + gs_gs_block, gs_coupl, gs_off_diag_12, gs_coupl_12))
    phot_1 = np.vstack((gs_coupl_01.reshape((1, n_states)), coupl_01, gs_coupl.reshape((1, n_states)), phot_block, gs_coupl_12.reshape((1, n_states)), coupl_12))
    gs_2 = np.concatenate((np.zeros(n_states + 1), gs_off_diag_12, gs_coupl_12, 2 * omega + gs_gs_block, gs_coupl))
    phot_2 = np.vstack((np.zeros((n_states + 1, n_states)), gs_coupl_12.reshape((1, n_states)), coupl_12, gs_coupl.reshape((1, n_states)), phot2_block))

    matrix = np.hstack((gs.reshape((len(gs), 1)), elec, gs_1.reshape((len(gs), 1)), phot_1, gs_2.reshape((len(gs), 1)), phot_2))

    # diagonalize
    #if any(np.iscomplex(frequency)):
    #    return sp.eig(matrix)
    #else:
    #    return sp.eigh(matrix)

    return matrix

