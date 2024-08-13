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
from qed_matrix_working_equations import qed_block
import numpy as np
from qed_mp import qed_mp

class qed_matrix_full(adcc.AdcMatrix):
    def __init__(self, method, hf_or_mp, block_orders=None, intermediates=None,
        diagonal_precomputed=None):
        """
        Initialise an polaritonic ADC matrix.
        Parameters
        ----------
        method : str or AdcMethod
            Method to use.
        hf_or_mp : polaritonic_adcc.qed_mp
            HF reference or QED MP ground state
        block_orders : optional
            PT order for each matrix block. (if None, ADC default is chosen)
        intermediates : adcc.Intermediates or NoneType
            Allows to pass intermediates to re-use to this class.
        diagonal_precomputed: adcc.AmplitudeVector
            Allows to pass a pre-computed diagonal, for internal use only.
        """
        if not isinstance(hf_or_mp, qed_mp):
            raise TypeError("hf_or_mp is not a qed_mp object.")

        if not isinstance(method, adcc.AdcMethod):
            method = adcc.AdcMethod(method)

        if diagonal_precomputed:
            if not isinstance(diagonal_precomputed, adcc.AmplitudeVector):
                raise TypeError("diagonal_precomputed needs to be"
                                " an AmplitudeVector.")
            if diagonal_precomputed.needs_evaluation:
                raise ValueError("diagonal_precomputed must already"
                                 " be evaluated.")

        self.timer = adcc.timings.Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = False
        self.ndim = 2
        self.extra_terms = []
        self.return_diag_as = "full"

        self.intermediates = intermediates
        if self.intermediates is None:
            self.intermediates = adcc.Intermediates.Intermediates(self.ground_state)

        # Determine orders of PT in the blocks
        if block_orders is None:
            block_orders = self.default_block_orders[method.base_method.name].copy()
            block_orders["ph_gs"] = block_orders["ph_ph"]
        else:
            tmp_orders = self.default_block_orders[method.base_method.name].copy()
            tmp_orders.update(block_orders)
            block_orders = tmp_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("ph_gs", "ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be None if ph_pphh isn't.")
        self.block_orders = block_orders

        self.qed_dispatch_dict = {
            "elec": "",
            "elec_couple": "_couple", "phot_couple": "_phot_couple",
            "phot": "_phot", "phot2": "_phot2",
            "elec_couple_inner": "_couple_inner",
            "elec_couple_edge": "_couple_edge",
            "phot_couple_inner": "_phot_couple_inner",
            "phot_couple_edge": "_phot_couple_edge"
        }

        def get_pp_blocks(disp_str):
            """
            Extraction of blocks from the adc_pp matrix
            ----------
            disp_str : string
                key specifying block to return
            """
            return {  # TODO Rename to self.block in 0.16.0
                block: qed_block(self.ground_state, block.split("_"),
                                 order=str(order) +\
                                 self.qed_dispatch_dict[disp_str],
                                 intermediates=self.intermediates,
                                 variant=variant)
                for block, order in block_orders.items() if order is not None
            }

        # Build the blocks and diagonals

        with self.timer.record("build"):
            variant = None
            blocks = {
                bl + "_" + key: get_pp_blocks(key)[bl]
                for bl, order in block_orders.items()
                if order is not None
                for key in self.qed_dispatch_dict
            }

            if diagonal_precomputed:
                self.__diagonal = diagonal_precomputed
            else:
                self.__diagonal = sum(bl.diagonal for bl in blocks.values()
                                      if bl.diagonal)
                self.__diagonal.evaluate()
            self.__init_qed_space_data(self.__diagonal)

            self.blocks_ph = {bl: blocks[bl].apply for bl in blocks}


    def __init_qed_space_data(self, diagonal):
        """Update the cached data regarding the spaces of the QED ADC matrix"""
        self.axis_spaces = {}
        self.axis_lengths = {}
        for block in diagonal.blocks_ph:
            if "gs" in block:
                # Either include g1 in whole libadcc backend, or use this
                # approach for now, which is only required for functionalities,
                # which should not be used with the full qed matrix yet anyway
                self.axis_spaces[block] = ['g1']
                self.axis_lengths[block] = 1
            else:
                self.axis_spaces[block] = getattr(diagonal, block).subspaces
                self.axis_lengths[block] = np.prod([
                    self.mospaces.n_orbs(sp) for sp in self.axis_spaces[block]
                ])
        self.shape = (sum(self.axis_lengths.values()),
                      sum(self.axis_lengths.values()))
        

    def diagonal(self):
        """Return the diagonal of the full QED-ADC matrix"""
        if self.return_diag_as == "full":
            return self.__diagonal
    #    # the following options are a hack for the guess setup
    #    elif self.return_diag_as == "phot":
    #        if hasattr(self.__diagonal, "pphh"):
    #            diag = adcc.AmplitudeVector(**{
    #            "ph": self.__diagonal.ph1,
    #            "pphh": self.__diagonal.pphh1})
    #        else:
    #            diag = adcc.AmplitudeVector(**{
    #            "ph": self.__diagonal.ph1})
    #        return diag
    #    elif self.return_diag_as == "phot2":
    #        if hasattr(self.__diagonal, "pphh"):
    #            diag = adcc.AmplitudeVector(**{
    #            "ph": self.__diagonal.ph2,
    #            "pphh": self.__diagonal.pphh2})
    #        else:
    #            diag = adcc.AmplitudeVector(**{
    #            "ph": self.__diagonal.ph2})
    #        return diag
    #    else:
    #        raise NotImplementedError(f"option {self.return_diag_as} is unknown")

        
    
