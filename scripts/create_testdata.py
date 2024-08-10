import psi4
from polaritonic_adcc.source import run_qed_adc
import hilbert
import adcc
from scipy.linalg import eigh
from polaritonic_adcc.source.testdata.psi4_scf_provider import get_psi4_wfn

adcc.set_n_threads(4)

#case = "hf_sto-3g"
#method = "adc1"
cases = ["hf_sto-3g", "h2o_sto-3g", "pyrrole_sto-3g"]
methods = ["adc2"]#, "adc2"]

res = {}

for case in cases:
    for method in methods:
        wfn = get_psi4_wfn(case)

        freq = [0., 0., 0.5]
        if case.startswith("pyrrole"):
            freq = [0., 0., 0.3]

        #exstates = run_qed_adc(wfn, method=method, coupl=[0., 0., 0.1], freq=freq,
        #                       qed_hf=False, qed_coupl_level=False, n_singlets=5, conv_tol=1e-7)
        exstates = run_qed_adc(wfn, method=method, coupl=[0., 0., 0.1], freq=freq,
                               qed_hf=False, qed_coupl_level=1, n_singlets=5, conv_tol=1e-8)
        key_str = case + "_" + method + "_approx_hf:"
        eigs = list(eigh(exstates.qed_matrix)[0])
        #eigs = list(exstates.excitation_energy)
        res[key_str] = eigs

for key in res:
    print(key, res[key])
