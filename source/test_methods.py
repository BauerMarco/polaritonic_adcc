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
import unittest

from adcc.misc import expand_test_templates

from numpy.testing import assert_allclose

from .testdata.yml_loader import qed_energies_psi4
from .testdata.psi4_scf_provider import get_psi4_wfn
from .workflow import run_qed_adc

import itertools

from scipy.linalg import eigh

# In principle one could also test the approx method against the full
# method, by expanding them to the full matrix dimension. The smallest
# example would be HF sto-3g, since it contains only one virtual orbital,
# which keeps the matrix dimension low. This is important since we
# build most of the matrix in the approx method from properties, so this
# is very slow for a lot of states, compared to the standard method.
# However, even this test case would take quite some time, so we leave
# it out for now.

testcases = ["hf_sto-3g", "h2o_sto-3g", "pyrrole_sto-3g"]
testcases_reduced = ["h2o_sto-3g", "pyrrole_sto-3g"]
adc1 = ["adc1"]
adc2 = ["adc2"]


@expand_test_templates(list(itertools.product(testcases, adc1 + adc2)))
class qed_test_qedhf(unittest.TestCase):
    """Setup method testing with polaritonic SCF"""
    def get_qed_adc(self, case, method, full):
        wfn = get_psi4_wfn(case)  # for larger test suites this should be cached
        if "pyrrole" in case:
            freq = [0., 0., 0.3]
        else:
            freq = [0., 0., 0.5]
        if full:
            level = False
        else:
            level = int(method[-1])
        return run_qed_adc(wfn, method=method, coupl=[0., 0., 0.1], freq=freq,
                             qed_hf=True, qed_coupl_level=level, n_singlets=5, conv_tol=1e-7)

    def template_approx_qedhf(self, case, method):
        """Test approx method with polaritonic SCF"""
        approx = self.get_qed_adc(case, method, False)
        eigvals = eigh(approx.qed_matrix)[0]

        ref_name = f"{case}_{method}_approx_qedhf"
        approx_ref = qed_energies_psi4[ref_name]

        assert_allclose(eigvals, approx_ref, atol=1e-6, rtol=1e-2)


    def template_full_qedhf(self, case, method):
        """Test full method with polaritonic SCF"""
        full = self.get_qed_adc(case, method, True)
        eigvals = full.excitation_energy

        ref_name = f"{case}_{method}_full_qedhf"
        full_ref = qed_energies_psi4[ref_name]

        assert_allclose(eigvals, full_ref, atol=1e-6)


@expand_test_templates(list(itertools.product(testcases_reduced, adc1 + adc2)))
class qed_test_approx_hf(unittest.TestCase):
    """Setup method testing with standard SCF for the approx method"""
    def get_qed_adc(self, case, method):
        wfn = get_psi4_wfn(case)  # for larger test suites this should be cached
        if "pyrrole" in case:
            freq = [0., 0., 0.3]
        else:
            freq = [0., 0., 0.5]
        #if full:
        #    level = False
        #else:
        #    level = int(method[-1])
        return run_qed_adc(wfn, method=method, coupl=[0., 0., 0.1], freq=freq,
                             qed_hf=False, qed_coupl_level=1, n_singlets=5, conv_tol=1e-7)

    def template_approx_hf(self, case, method):
        """Test approx method with standard SCF"""
        approx = self.get_qed_adc(case, method)
        eigvals = eigh(approx.qed_matrix)[0]

        ref_name = f"{case}_{method}_approx_hf"
        approx_ref = qed_energies_psi4[ref_name]

        assert_allclose(eigvals, approx_ref, atol=1e-6, rtol=1e-2)


@expand_test_templates(list(itertools.product(testcases, adc1)))
class qed_test_full_hf(unittest.TestCase):
    """Setup method testing with standard SCF for the approx method"""
    def get_qed_adc(self, case, method):
        wfn = get_psi4_wfn(case)  # for larger test suites this should be cached
        if "pyrrole" in case:
            freq = [0., 0., 0.3]
        else:
            freq = [0., 0., 0.5]
        #if full:
        #    level = False
        #else:
        #    level = int(method[-1])
        return run_qed_adc(wfn, method=method, coupl=[0., 0., 0.1], freq=freq,
                             qed_hf=False, qed_coupl_level=False, n_singlets=5, conv_tol=1e-7)

    def template_full_hf(self, case, method):
        """Test full method with standard SCF"""
        full = self.get_qed_adc(case, method)
        eigvals = full.excitation_energy

        ref_name = f"{case}_{method}_full_hf"
        full_ref = qed_energies_psi4[ref_name]

        assert_allclose(eigvals, full_ref, atol=1e-6)


@expand_test_templates(list(itertools.product(testcases, adc2)))
class qed_test_qedhf_ucc(unittest.TestCase):
    """Setup method testing with polaritonic SCF"""
    def get_qed_adc(self, case, method):
        wfn = get_psi4_wfn(case)  # for larger test suites this should be cached
        if "pyrrole" in case:
            freq = [0., 0., 0.3]
        else:
            freq = [0., 0., 0.5]
        #if full:
        #    level = False
        #else:
        #    level = int(method[-1])
        return run_qed_adc(wfn, method=method, coupl=[0., 0., 0.1], freq=freq, gs="ucc",
                             qed_hf=True, qed_coupl_level=False, n_singlets=5, conv_tol=1e-7)


    def template_full_qedhf_ucc(self, case, method):
        """Test full method with polaritonic SCF"""
        full = self.get_qed_adc(case, method)
        eigvals = full.excitation_energy

        ref_name = f"{case}_{method}_full_qedhf_ucc"
        full_ref = qed_energies_psi4[ref_name]

        assert_allclose(eigvals, full_ref, atol=1e-6)
