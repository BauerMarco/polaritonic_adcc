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
import libadcc
from .backends import import_qed_scf_result

#__all__ = ["refstate"]

class refstate(adcc.ReferenceState):
    def __init__(self, hfdata, qed_hf, coupl, core_orbitals=None, frozen_core=None,
                 frozen_virtual=None, symmetry_check_on_import=False,
                 import_all_below_n_orbs=10):
        """
        Lazily evaluated (non-)polaritonic HF object

        Orbital subspace selection: In order to specify `frozen_core`,
        `core_orbitals` and `frozen_virtual`, adcc allows a range of
        specifications including

           a. A number: Just put this number of alpha orbitals and this
              number of beta orbitals into the respective space. For frozen
              core and core orbitals these are counted from below, for
              frozen virtual orbitals, these are counted from above. If both
              frozen core and core orbitals are specified like this, the
              lowest-energy, occupied orbitals will be put into frozen core.
           b. A range: The orbital indices given by this range will be put
              into the orbital subspace.
           c. An explicit list of orbital indices to be placed into the
              subspace.
           d. A pair of (a) to (c): If the orbital selection for alpha and
              beta orbitals should differ, a pair of ranges, or a pair of
              index lists or a pair of numbers can be specified.

        Parameters
        ----------
        hfdata
            Object with Hartree-Fock data (e.g. a molsturm scf state, a pyscf
            SCF object or any class implementing the
            :py:class:`adcc.HartreeFockProvider` interface or in fact any python
            object representing a pointer to a C++ object derived off
            the :cpp:class:`adcc::HartreeFockSolution_i`.
        
        qed_hf : bool
            Specify, whether the hfdata object is a polaritonic or standard
            SCF reference.
        
        coupl : list or tuple or numpy array of length 3
            x, y, z vector containing the coupling strengths to the cavity photon.

        core_orbitals : int or list or tuple, optional
            The orbitals to be put into the core-occupied space. For ways to
            define the core orbitals see the description above. Note, that you
            don't want to make use of them for a polaritonic calculation,
            as they are usually connected to core-valence separated calculations,
            which break the dipole approximation! It you want to read more on this
            see the paper linked in the polaritonic_adcc documentation.

        frozen_core : int or list or tuple, optional
            The orbitals to be put into the frozen core space. For ways to
            define the core orbitals see the description above. For an automatic
            selection of the frozen core space one may also specify
            ``frozen_core=True``.

        frozen_virtuals : int or list or tuple, optional
            The orbitals to be put into the frozen virtual space. For ways to
            define the core orbitals see the description above.

        symmetry_check_on_import : bool, optional
            Should symmetry of the imported objects be checked explicitly during
            the import process. This massively slows down the import and has a
            dramatic impact on memory usage. Thus one should enable this only
            for debugging (e.g. for testing import routines from the host
            programs). Do not enable this unless you know what you are doing.

        import_all_below_n_orbs : int, optional
            For small problem sizes lazy make less sense, since the memory
            requirement for storing the ERI tensor is neglibile and thus the
            flexiblity gained by having the full tensor in memory is
            advantageous. Below the number of orbitals specified by this
            parameter, the class will thus automatically import all ERI tensor
            and Fock matrix blocks.

        Examples
        --------
        To start a calculation with the 2 lowest alpha and beta orbitals
        in the core occupied space, construct the class as

        >>> ReferenceState(hfdata, core_orbitals=2)

        or

        >>> ReferenceState(hfdata, core_orbitals=range(2))

        or

        >>> ReferenceState(hfdata, core_orbitals=[0, 1])

        or

        >>> ReferenceState(hfdata, core_orbitals=([0, 1], [0, 1]))

        There is no restriction to choose the core occupied orbitals
        from the bottom end of the occupied orbitals. For example
        to select the 2nd and 3rd orbital setup the class as

        >>> ReferenceState(hfdata, core_orbitals=range(1, 3))

        or

        >>> ReferenceState(hfdata, core_orbitals=[1, 2])

        If different orbitals should be placed in the alpha and
        beta orbitals, this can be achievd like so

        >>> ReferenceState(hfdata, core_orbitals=([1, 2], [0, 1]))

        which would place the 2nd and 3rd alpha and the 1st and second
        beta orbital into the core space.
        """
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
        Return the scalar product between coupl and the dipole operator
        """
        dips = self.operators.electric_dipole
        #couplings = self.coupling
        #freqs = self.frequency
        total_dip = adcc.OneParticleOperator(self.mospaces, is_symmetric=True)
        for coupling, dip in zip(self.coupl, dips):
            total_dip += coupling * dip
        total_dip.evaluate()
        return total_dip[block]

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
        """
        Return the electron repulsion integrals, to which the polaritonic
        two partilce correction is added, which obeys the same symmetry.
        """
        b = adcc.block
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
