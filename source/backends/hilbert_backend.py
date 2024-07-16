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
import psi4
import numpy as np

# sadly this doesnt work, because this class is not accessible from adcc
parent_class = "adcc.backends.psi4.Psi4HFProvider"

class HilbertProvider(parent_class):
        def __init__(self, wfn):
            #self.wfn = wfn
            super().__init__(wfn)

        def get_restricted(self):
            if isinstance(self.wfn, (psi4.core.RHF, psi4.core.ROHF)):
                return True
            elif isinstance(self.wfn, psi4.core.UHF):
                return False
            else:
                # The hilbert package returns a more basic object, which does
                # not provide a restricted indicator, so we determine it here
                orben_a = np.asarray(self.wfn.epsilon_a())
                orben_b = np.asarray(self.wfn.epsilon_b())
                # TODO Maybe give a small tolerance here
                return all(orben_a == orben_b)
            
        def fill_occupation_f(self, out):
            if not hasattr(self.wfn, "occupation_a"):  # hilbert package
                num_of_orbs = self.wfn.nmo()
                nalpha_elec = self.wfn.nalpha()
                nbeta_elec = self.wfn.nbeta()
                occ_array_a = occ_array_b = np.zeros(num_of_orbs)
                np.put(occ_array_a, np.arange(nalpha_elec), 1)
                np.put(occ_array_b, np.arange(nbeta_elec), 1)
                out[:] = np.hstack((occ_array_a, occ_array_b))
            else:
                out[:] = np.hstack((
                    np.asarray(self.wfn.occupation_a()),
                    np.asarray(self.wfn.occupation_b())
                ))


def hilbert_scf_import(wfn):
    if wfn.nirrep() > 1:
        raise adcc.exceptions.InvalidReference("The passed Psi4 wave function object needs to "
                                               "have exactly one irrep, i.e. be of C1 symmetry.")
    return HilbertProvider

