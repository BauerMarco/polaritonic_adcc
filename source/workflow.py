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
import sys
import numpy as np
import warnings
from .qed_matrix_from_diag_adc import qed_matrix_from_diag_adc
from .qed_adc_in_std_basis_with_self_en import first_order_qed_matrix
from .refstate import refstate
from .qed_npadc_exstates import qed_npadc_exstates
from .qed_mp import qed_mp
from .qed_ucc import qed_ucc
from .full_qed_matrix import qed_matrix_full
from .qed_amplitude_vec import qed_amplitude_vec
from adcc import run_adc, ExcitedStates
from adcc.exceptions import InputError
from adcc.workflow import (validate_state_parameters, estimate_n_guesses,
                           setup_solver_printing, obtain_guesses_by_inspection)
from .solver.lanczos import lanczos
from adcc.solver.lanczos import default_print as lanczos_print
from .solver.davidson import jacobi_davidson
from adcc.solver.davidson import default_print as davidson_print
from adcc.solver.explicit_symmetrisation import (IndexSpinSymmetrisation,
                                             IndexSymmetrisation)

#__all__ = ["run_qed_adc"]

def run_qed_adc(data_or_matrix, coupl=None, freq=None, qed_hf=True, gs="mp",
                qed_coupl_level=1, n_states=None, kind="any", conv_tol=None,
                eigensolver="davidson", guesses=None, n_guesses=None,
                n_guesses_doubles=None, output=sys.stdout, core_orbitals=None,
                frozen_core=None, frozen_virtual=None, method=None,
                n_singlets=None, n_triplets=None, n_spin_flip=None,
                environment=None, **solverargs):
    """Run an ADC calculation.

    Main entry point to run an ADC calculation. The reference to build the ADC
    calculation upon is supplied using the `data_or_matrix` argument.
    `adcc` is pretty flexible here. Possible options include:

        a. Hartree-Fock data from a host program, e.g. a molsturm SCF
           state, a pyscf SCF object or any class implementing the
           :py:class:`adcc.HartreeFockProvider` interface. From this data all
           objects mentioned in (b) to (d) will be implicitly created and will
           become available in the returned state.
        b. A :py:class:`polaritonic_adcc.refstate` object
        c. A :py:class:`polaritonic_adcc.qed_mp` object
        d. A :py:class:`polaritonic_adcc.qed_matrix_full` object

    Parameters
    ----------
    data_or_matrix
        Data containing the SCF reference
    n_states : int, optional
    kind : str, optional
    n_singlets : int, optional
    n_triplets : int, optional
    n_spin_flip : int, optional
        Specify the number and kind of states to be computed. Possible values
        for kind are "singlet", "triplet", "spin_flip" and "any", which is
        the default. For unrestricted references clamping spin-pure
        singlets/triplets is currently not possible and kind has to remain as
        "any". For restricted references `kind="singlets"` or `kind="triplets"`
        may be employed to enforce a particular excited states manifold.
        Specifying `n_singlets` is equivalent to setting `kind="singlet"` and
        `n_states=5`. Similarly for `n_triplets` and `n_spin_flip`.
        `n_spin_flip` is only valid for unrestricted references.

    conv_tol : float, optional
        Convergence tolerance to employ in the iterative solver for obtaining
        the ADC vectors (default: `1e-6` or 10 * SCF tolerance,
        whatever is larger)

    eigensolver : str, optional
        The eigensolver algorithm to use.

    n_guesses : int, optional
        Total number of guesses to compute. By default only guesses derived from
        the singles block of the ADC matrix are employed. See
        `n_guesses_doubles` for alternatives. If no number is given here
        `n_guesses = min(4, 2 * number of excited states to compute)`
        or a smaller number if the number of excitation is estimated to be less
        than the outcome of above formula.

    n_guesses_doubles : int, optional
        Number of guesses to derive from the doubles block. By default none
        unless n_guesses as explicitly given or automatically determined is
        larger than the number of singles guesses, which can be possibly found.

    guesses : list, optional
        Provide the guess vectors to be employed for the ADC run. Takes
        preference over `n_guesses` and `n_guesses_doubles`, such that these
        parameters are ignored.

    output : stream, optional
        Python stream to which output will be written. If `None` all output
        is disabled.

    core_orbitals : int or list or tuple, optional
        The orbitals to be put into the core-occupied space. For ways to
        define the core orbitals see the description in
        :py:class:`adcc.ReferenceState`.
        Required if core-valence separation is applied and the input data is
        given as data from the host program (i.e. option (a) discussed above)

    frozen_core : int or list or tuple, optional
        The orbitals to select as frozen core orbitals (i.e. inactive occupied
        orbitals for both the MP and ADC methods performed). For ways to define
        these see the description in :py:class:`adcc.ReferenceState`.

    frozen_virtual : int or list or tuple, optional
        The orbitals to select as frozen virtual orbitals (i.e. inactive
        virtuals for both the MP and ADC methods performed). For ways to define
        these see the description in :py:class:`adcc.ReferenceState`.

    environment : bool or list or dict, optional
        The keywords to specify how coupling to an environment model,
        e.g. PE, is treated.
    
    coupl : list or tuple or numpy array of length 3, optional
        x, y, z coupling vector to the cavity photon. Use the definition
        1 / sqrt(2 * freq * eps_0 * eps_r * V)!

    freq : list or tuple or numpy array or int, optional
        Energy of the cavity photon.

    qed_hf : bool, optional
        Specify, whether a standard or polaritonic SCF result is provided.

    gs : str, optional
        Which ground state reference to use

    qed_coupl_level : bool or int, optional
        Specify, whether to calculate the full matrix (False), or provide
        the perturbative level to which polaritonic coupling shall be included
        into a truncated state space approach (1 or 2).

    Other parameters
    ----------------
    max_subspace : int, optional
        Maximal subspace size
    max_iter : int, optional
        Maximal number of iterations

    Returns
    -------
    ExcitedStates
        An :class:`adcc.ExcitedStates` object or
        :class:`polaritonic_adcc.qed_npadc_exstates` object, which inherits
        from the previous object, containing the
        :class:`polaritonic_adcc.full_qed_matrix`, the
        :class:`polaritonic_adcc.qed_mp` ground state and the
        :class:`polaritonic_adcc.refstate` as well as computed eigenpairs.

    Examples
    --------

    Run an ADC(3) calculation on top of a non-polaritonic `pyscf`
    RHF reference of hydrogen flouride, building the qed matrix
    in a truncated state basis.

    >>> from pyscf import gto, scf
    ... mol = gto.mole.M(atom="H 0 0 0; F 0 0 1.1", basis="sto-3g")
    ... mf = scf.RHF(mol)
    ... mf.conv_tol_grad = 1e-8
    ... mf.kernel()
    ...
    ... state = run_qed_adc(mf, method="adc3", n_singlets=3, freq=[0., 0., 0.5],
                            coupl=[0., 0., 0.1], qed_hf=True)

    Run an ADC(2) calculation of O2 with a polaritonic `psi4` reference,
    building the full polaritonic matrix.

    >>> import hilbert
    ... mol = psi4.geometry(f'''
    ...     0 1
    ...     O 0.0 0.0 0.0
    ...     O 0.0 0.0 1.2
    ...     units angstrom
    ...     symmtery c1
    ...     no_reorient
    ... ''')
    ... psi4.core.be_quiet()
    ... psi4.set_options({'basis': 'sto-3g', 'scf_type': 'df',
                          'e_convergence': 1e-10})
    ... psi4.set_module_options('hilbert': {'cavity_frequency': [0.0, 0.0, 0.4],
                                            'cavity_coupling_strength': [0.0, 0.0, 0.1]})
    ... scf_e, wfn = psi4.energy('scf', return_wfn=True)
    ... state = run_qed_adc(mf, n_singlets=3, freq=[0., 0., 0.4], coupl=[0., 0., 0.1],
                            qed_coupl_level=False)
    """
    # Validation of given parameters
    if method == "adc0":
        raise NotImplementedError("in zeroth order the system is decoupled from the cavity")
    if coupl == None or freq == None:
        raise SyntaxError("coupl and freq can not be set to None")
    if np.linalg.norm(coupl) < 1e-15 or np.linalg.norm(freq) < 1e-15:
        raise NotImplementedError("If you want to do a calculation with zero"
                                  "coupling or zero frequency, you can coduct"
                                  "a standard adcc calculation")
    if "csv" in method:
        warnings.warn("CVS ADC excitations generally do not obey the dipole approximation,"
                      "and therefore should not be used at all with polaritonic_adcc")
    # TODO: Here we always multiply with sqrt(2 * omega), since this
    # eases up the Hamiltonian and is required if you provide a hilbert
    # package QED-HF input. This can differ between QED-HF implementations,
    # e.g. the psi4numpy QED-RHF helper does not do that. Therefore, this
    # factor needs to be adjusted depending on the input.
    coupl_adapted = np.array(coupl) * np.sqrt(2 * np.linalg.norm(np.real(freq)))
    warnings.warn("coupling parameter in polaritonic_adcc was adapted as sqrt(2 * real(freq)), "
                  "because the definition for the coupling parameter here is "
                  r"$\lambda = \frac{\epsilon}{\epsilon_0 \epsilon_r V}$")
    # Determining the correct computation from the given input parameters
    if gs == "mp":
        ground_state = qed_mp
    elif gs == "ucc":
        if qed_hf == False and qed_coupl_level == False:
            raise NotImplementedError("UCC is only implemented for a polaritonic HF reference."
                                      "However, you can requested the truncated matrix built"
                                      "with a standard HF reference.")
        if qed_coupl_level != False:
            warnings.warn("From a theoretical perspective using qed_ucc with the truncated"
                          "matrix built is not quite correct, since the t2 and qed_t1"
                          "amplitudes are coupled. Hence, it is recommended to use"
                          "qed_mp for QED_npADC calculations.")
        ground_state = qed_ucc
    else:
        raise NotImplementedError(f"ground state option {gs} unknown")

    if isinstance(data_or_matrix, refstate):
        qed_groundstate = ground_state(data_or_matrix, np.linalg.norm(np.real(freq)), qed_hf=qed_hf)
    elif isinstance(data_or_matrix, qed_mp):  # this check for gs type might cause problems
        qed_groundstate = data_or_matrix
    elif isinstance(data_or_matrix, qed_matrix_full):
        pass
    else:
        qed_refstate = refstate(data_or_matrix, qed_hf, coupl_adapted, core_orbitals=core_orbitals, frozen_core=frozen_core,
                        frozen_virtual=frozen_virtual)
        qed_groundstate = ground_state(qed_refstate, np.linalg.norm(np.real(freq)), qed_hf=qed_hf)
    if type(qed_coupl_level) == int and qed_hf:
        exstates = run_adc(qed_groundstate, n_states=n_states, kind=kind, conv_tol=conv_tol,
                eigensolver=eigensolver, guesses=guesses, n_guesses=n_guesses,
                n_guesses_doubles=n_guesses_doubles, output=output, core_orbitals=core_orbitals,
                frozen_core=frozen_core, frozen_virtual=frozen_virtual, method=method,
                n_singlets=n_singlets, n_triplets=n_triplets, n_spin_flip=n_spin_flip,
                environment=environment, **solverargs)
        qed_exstates = qed_npadc_exstates(exstates, qed_coupl_level)
        qed_matrix_class = qed_matrix_from_diag_adc(qed_exstates, coupl_adapted, freq)
    if not qed_coupl_level:
        # full qed at given adc level
        if (method not in ["adc1", "adc2"] and qed_hf) or (method != "adc1" and not qed_hf):
            raise NotImplementedError("full polaritonic ADC is only implemented"
                                      "for ADC(1) and ADC(2) with a polaritonic"
                                      "Hartree-Fock reference and only for ADC(1)"
                                      "with a standard Hartree-Fock reference.")
        if any(np.iscomplex(freq)):
            raise NotImplementedError("full polaritonic ADC is not implemented for lossy cavities")
        if isinstance(data_or_matrix, qed_matrix_full):
            qed_mat_full = data_or_matrix
        elif not isinstance(data_or_matrix, qed_matrix_full) and isinstance(qed_groundstate, qed_mp):
            qed_mat_full = qed_matrix_full(method, qed_groundstate)
        else:
            raise TypeError("full qed adc matrix either needs to be provided or "
                            "build from a qed_mp instance here, which is valid with "
                            "respect to the qed_hf parameter.")
        
        diag = diagonalizer(qed_mat_full, n_states, kind, guesses=guesses, n_guesses=n_guesses,
        n_guesses_doubles=n_guesses_doubles, conv_tol=conv_tol, output=output,
        eigensolver=eigensolver, n_triplets=n_triplets, n_singlets=n_singlets,
        n_spin_flip=n_spin_flip, **solverargs)

        exstates = ExcitedStates(diag)
    elif qed_coupl_level == 1 and qed_hf:
        # adc at given level with adapted ERIs and 1st order qed coupling
        qed_matrix = qed_matrix_class.first_order_coupling()
        exstates.qed_matrix = qed_matrix
    elif qed_coupl_level == 2 and qed_hf:
        # adc at given level with adapted ERIs and 2st order qed coupling
        if method == "adc1":
            raise NotImplementedError("requesting adc1 with second order coupling is not possible")
        qed_matrix = qed_matrix_class.second_order_coupling()
        exstates.qed_matrix = qed_matrix
    elif qed_coupl_level == 1 and not qed_hf:
        # adc at given level without adapted ERIs and 1st order qed coupling
        # this is the equivalent to using field free states
        exstates = run_adc(data_or_matrix, n_states=n_states, kind=kind, conv_tol=conv_tol,
                eigensolver=eigensolver, guesses=guesses, n_guesses=n_guesses,
                n_guesses_doubles=n_guesses_doubles, output=output, core_orbitals=core_orbitals,
                frozen_core=frozen_core, frozen_virtual=frozen_virtual, method=method,
                n_singlets=n_singlets, n_triplets=n_triplets, n_spin_flip=n_spin_flip,
                environment=environment, **solverargs)
        qed_matrix = first_order_qed_matrix(exstates, coupl_adapted, freq)
        exstates.qed_matrix = qed_matrix
    else:
        raise NotImplementedError(f"the given combination of qed_coupl_level = "
                                  "{qed_coupl_level} and qed_hf = {qed_hf} is not" 
                                  "implemented")
        
    return exstates


def diagonalizer(matrix, n_states, kind, guesses=None, n_guesses=None,
        n_guesses_doubles=None, conv_tol=None, output=sys.stdout,
        eigensolver="davidson", n_triplets=None, n_singlets=None,
        n_spin_flip=None, **solverargs):
    """Diagonalize the provided full polaritonic matrix."""
    
    if conv_tol is None:
        conv_tol = max(10 * matrix.reference_state.conv_tol, 1e-6)
    if matrix.reference_state.conv_tol > conv_tol:
        raise InputError(
            "Convergence tolerance of SCF results "
            f"(== {matrix.reference_state.conv_tol}) needs to be lower than ADC "
            f"convergence tolerance parameter conv_tol (== {conv_tol})."
        )
    
    explicit_symmetrisation = IndexSymmetrisation
    if kind in ["singlet", "triplet"]:
        explicit_symmetrisation = IndexSpinSymmetrisation(
            matrix, enforce_spin_kind=kind
        )

    n_states, kind = validate_state_parameters(
        matrix.reference_state, n_states=n_states, n_singlets=n_singlets,
        n_triplets=n_triplets, n_spin_flip=n_spin_flip, kind=kind)
    
    if eigensolver == "davidson":
        n_guesses_per_state = 2
        callback = setup_solver_printing(
            "Jacobi-Davidson", matrix, kind, davidson_print,
            output=output)
        run_eigensolver = jacobi_davidson
    elif eigensolver == "lanczos":
        n_guesses_per_state = 1
        callback = setup_solver_printing(
            "Lanczos", matrix, kind, lanczos_print,
            output=output)
        run_eigensolver = lanczos
    else:
        raise NotImplementedError(f"eigensolver {eigensolver} is unknown")
    
    n_guesses = estimate_n_guesses(matrix, n_states, n_guesses_per_state=n_guesses_per_state)

    guesses = obtain_guesses_by_inspection_qed(matrix, n_guesses, kind, n_guesses_doubles=n_guesses_doubles)

    solverargs.setdefault("which", "SA")

    # the following could be used, if one could set up a libtensor scalar,
    # i.e. that the entire functions file, as well as all solver functionalities
    # regarding davidson and lanczos could be left out, but the new function
    # for libtensor won't be accepted, since it's not a core functionality
    # of the adcc package itself.
    #diag_result = adcc.diagonalise_adcmatrix(matrix, n_states, kind, eigensolver=eigensolver,
    #                      guesses=guesses, n_guesses=n_guesses, n_guesses_doubles=n_guesses_doubles,
    #                      conv_tol=conv_tol, output=output, **solverargs)
    #return diag_result
    return run_eigensolver(matrix, guesses, n_ep=n_states, conv_tol=conv_tol,
                           callback=callback,
                           explicit_symmetrisation=explicit_symmetrisation,
                           **solverargs)



def obtain_guesses_by_inspection_qed(matrix, n_guesses, kind, n_guesses_doubles=None):  # noqa: E501
    """
    Obtain guesses for full qed method, by pushing the subblocks
    into obtain_guesses_by_inspection.
    """
    # It seems like qed_adc converges better with all three guesses
    # originating from the purely electronic diagonal. If the actual
    # diagonals are desired, do something like the commented out hacks.
    guesses_elec = obtain_guesses_by_inspection(
        matrix, n_guesses, kind, n_guesses_doubles)
    #matrix.return_diag_as = "phot"  # hack for the guess setup
    guesses_phot = obtain_guesses_by_inspection(
        matrix, n_guesses, kind, n_guesses_doubles)
    #matrix.return_diag_as = "phot2"  # hack for the guess setup
    guesses_phot2 = obtain_guesses_by_inspection(
        matrix, n_guesses, kind, n_guesses_doubles)
    #matrix.return_diag_as = "full"  # resets the hacks from above

    # Usually only few states are requested and most of them are close
    # to pure electronic states, so we initialize the guess vectors
    # as almost purely electric guesses.
    for i in np.arange(n_guesses):
        # TODO: maybe make these values accessible by a keyword, since
        # they can tune the performance. From my experience these work
        # very well though
        guesses_phot[i] *= 0.02
        guesses_phot2[i] *= 0.001
    if n_guesses != len(guesses_phot):
        raise InputError("amount of guesses for electronic and photonic must be "
                         "equal, but are {} electronic and {} photonic "
                         "guesses".format(len(guesses_elec), len(guesses_phot)))

    zero = np.array([0.0])

    if hasattr(guesses_elec[0], "pphh"):
        final_guesses = [qed_amplitude_vec(**{
            "ph": guesses_elec[guess_index].ph,
            "pphh": guesses_elec[guess_index].pphh,
            "gs1": zero.copy(), "ph1": guesses_phot[guess_index].ph,
            "pphh1": guesses_phot[guess_index].pphh,
            "gs2": zero.copy(), "ph2": guesses_phot2[guess_index].ph,
            "pphh2": guesses_phot2[guess_index].pphh
        }) for guess_index in np.arange(n_guesses)]
    else:
        final_guesses = [qed_amplitude_vec(**{
            "ph": guesses_elec[guess_index].ph,
            "gs1": zero.copy(), "ph1": guesses_phot[guess_index].ph,
            "gs2": zero.copy(), "ph2": guesses_phot2[guess_index].ph
        }) for guess_index in np.arange(n_guesses)]

    # TODO: maybe make these values accessible by a keyword, since
    # they can tune the performance. From my experience these work
    # quite well though, but depending on how strong a state couples
    # to the single photon dispersion mode and how many states one
    # requests, adjusting these values can increase the
    # convergence rate
    # for stronger coupling e.g. 2
    final_guesses[n_guesses - 2].gs1 = np.array([5])
    # for stronger coupling e.g. 5
    final_guesses[n_guesses - 1].gs2 = np.array([20])

    return [vec / np.sqrt(vec @ vec) for vec in final_guesses]
