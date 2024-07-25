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
import numpy as np
import libadcc
from backends import import_qed_scf_result

#__all__ = ["refstate"]

class refstate(adcc.ReferenceState):
    def __init__(self, hfdata, qed_hf, coupl, core_orbitals=None, frozen_core=None,
                         frozen_virtual=None, symmetry_check_on_import=False,
                         import_all_below_n_orbs=10):
        if not isinstance(hfdata, libadcc.HartreeFockSolution_i):
            hfdata = import_qed_scf_result(hfdata)
        super().__init__(hfdata, core_orbitals=core_orbitals, frozen_core=frozen_core,
                         frozen_virtual=frozen_virtual, symmetry_check_on_import=symmetry_check_on_import,
                         import_all_below_n_orbs=import_all_below_n_orbs)
        self.coupl = coupl
        self.qed_hf = qed_hf

    def __getattr__(self, attr):
        b = adcc.block

        if attr.startswith("f"):
            return self.fock(b.__getattr__(attr[1:]))
        elif attr.startswith("get_qed_total_dip"):
            return self.get_qed_total_dip(b.__getattr__(attr))
        else:
            return self.eri(b.__getattr__(attr))
        
    @adcc.misc.cached_member_function
    def get_qed_total_dip(self, block):
        """
        Return qed coupling strength times dipole operator
        """
        dips = self.operators.electric_dipole
        #couplings = self.coupling
        #freqs = self.frequency
        total_dip = adcc.OneParticleOperator(self.mospaces, is_symmetric=True)
        for coupling, dip in zip(self.coupl, dips):
            total_dip += coupling * dip
        total_dip.evaluate()
        return total_dip[block]

    #@cached_member_function
    #def get_qed_omega(self):
    #    """
    #    Return the cavity frequency
    #    """
    #    if self.is_qed:
    #        freqs = self.frequency
    #        return np.linalg.norm(freqs)

    #@adcc.misc.cached_member_function
    def qed_D_object(self, block):
        """
        Return the object, which is added to the ERIs in a PT QED calculation
        """
        b = adcc.block
        einsum = adcc.functions.einsum
        total_dip = adcc.OneParticleOperator(self.mospaces, is_symmetric=True)
        total_dip.oo = self.get_qed_total_dip(b.oo)
        total_dip.ov = self.get_qed_total_dip(b.ov)
        total_dip.vv = self.get_qed_total_dip(b.vv)
        # We have to define all the blocks from the
        # D_{pqrs} = d_{pr} d_{qs} - d_{ps} d_{qr} object, which has the
        # same symmetry properties as the ERI object
        # Actually in b.ovov: second term: ib,ja would be ib,aj ,
        # but d_{ia} = d_{ai}
        ds = {
            b.oooo: einsum('ik,jl->ijkl', total_dip.oo, total_dip.oo)
            - einsum('il,jk->ijkl', total_dip.oo, total_dip.oo),
            b.ooov: einsum('ik,ja->ijka', total_dip.oo, total_dip.ov)
            - einsum('ia,jk->ijka', total_dip.ov, total_dip.oo),
            b.oovv: einsum('ia,jb->ijab', total_dip.ov, total_dip.ov)
            - einsum('ib,ja->ijab', total_dip.ov, total_dip.ov),
            b.ovvv: einsum('ib,ac->iabc', total_dip.ov, total_dip.vv)
            - einsum('ic,ab->iabc', total_dip.ov, total_dip.vv),
            b.ovov: einsum('ij,ab->iajb', total_dip.oo, total_dip.vv)
            - einsum('ib,ja->iajb', total_dip.ov, total_dip.ov),
            b.vvvv: einsum('ac,bd->abcd', total_dip.vv, total_dip.vv)
            - einsum('ad,bc->abcd', total_dip.vv, total_dip.vv),
        }
        return ds[block]

    @adcc.misc.cached_member_function
    def eri(self, block):
        b = adcc.block
        #from .functions import einsum
        einsum = adcc.functions.einsum
        # Since there is no TwoParticleOperator object,
        # we initialize it like this
        ds_init = adcc.OneParticleOperator(self.mospaces, is_symmetric=True)
        ds = {
            b.oooo: einsum('ik,jl->ijkl', ds_init.oo, ds_init.oo),
            b.ooov: einsum('ik,ja->ijka', ds_init.oo, ds_init.ov),
            b.oovv: einsum('ia,jb->ijab', ds_init.ov, ds_init.ov),
            b.ovvv: einsum('ib,ac->iabc', ds_init.ov, ds_init.vv),
            b.ovov: einsum('ij,ab->iajb', ds_init.oo, ds_init.vv),
            b.vvvv: einsum('ac,bd->abcd', ds_init.vv, ds_init.vv),
        }
        ds[block] = self.qed_D_object(block)
        return super().eri(block) + ds[block]
