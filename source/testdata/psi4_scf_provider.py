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
import psi4
import adcc


def calc(mol_str, basis):
    """Build the wfn object for the three supported test molecules using psi4"""
    if mol_str == "pyrrole":
        mol = psi4.geometry(f"""
            0 1
            N   -0.92281471284584  0.58821402111512 -0.33781594435227
            C    0.30043862859482  1.14552542500874 -0.04548363647026
            C    1.15592491009446  0.10790051720692  0.32517145421854
            C    0.41009737446365 -1.10616275981017  0.24854431680376
            C   -0.88005056619207 -0.77608828168333 -0.16676764758713
            H    0.46488928756668  2.21923339825349 -0.12229705905370
            H    2.19985554468969  0.22126115123613  0.61703825932396
            H    0.76642802914261 -2.11210759889606  0.46976975933510
            H   -1.75611678490561 -1.39620709785452 -0.35048350701150
            H   -1.73865171060838  1.10843122542367 -0.63767599520649
            units angstrom
            symmetry c1
            no_reorient
            """)
    elif mol_str == "hf":
        mol = psi4.geometry(f"""
            0 1
            H 0.0 0.0 0.0
            F 0.0 0.0 0.917
            units angstrom
            symmetry c1
            no_reorient
            """)
    elif mol_str == "h2o":
        mol = psi4.geometry(f"""
            0 1
            O          0.00000        0.00000        0.11779
            H          0.00000        0.75545       -0.47116
            H          0.00000       -0.75545       -0.47116
            units angstrom
            symmetry c1
            no_reorient
            """)
    else:
        raise NotImplementedError(f"mol_str {mol_str} unknown")

    # set the number of cores equal to the auto-determined value from
    # the adcc ThreadPool
    psi4.set_num_threads(adcc.get_n_threads())
    psi4.core.be_quiet()
    psi4.set_options({'basis': basis,
                      #'scf_type': 'df',
                      'd_convergence': 1e-10,
                      'e_convergence': 1e-12})
    #psi4.set_module_options('hilbert', {'n_photon_states': 1,
    #                  'cavity_frequency': '{}'.format(freq),
    #                  'cavity_coupling_strength': '{}'.format(coupl)})
    #scf_e, wfn = psi4.energy('polaritonic-uhf', return_wfn=True)
    scf_e, wfn = psi4.energy('hf', return_wfn=True)
    return wfn


def get_psi4_wfn(testcase):
    """Dispatch and pipeline to calc"""
    mol_str, basis = testcase.split("_")
    return calc(mol_str, basis)

