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


# Best case would be, that all polaritonic HF providers would not differ
# from their non-polaritonic counterparts, but that is not true e.g. for
# the hilbert plugin to psi4. The adapted imports are handled here.

import adcc
from .hilbert_backend_no_inherit import hilbert_scf_import

#__all__ = ["import_qed_scf_result"]

def import_qed_scf_result(scf):
    hilbert = False
    try:
        import psi4
        if isinstance(scf, psi4.core.Wavefunction):
            return hilbert_scf_import(scf)
        else:
            adcc.backends.import_scf_result(scf)
    except ModuleNotFoundError:
        adcc.backends.import_scf_result(scf)


