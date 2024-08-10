.. _usage:

Usage
=====


Get started
-----------

Polaritonic_adcc can has a main workflow function called run_qed_adc,
which only takes an extended and not altered set of arguments as run_adc
from the adcc. Hence, you should also get familiar on how to
`run an adcc calculation <https://adc-connect.org/v0.15.13/calculations.html>`_ .

Making use of the psi4 backend as a non-polaritonic SCF provider a calculation
can be carried out as follows:

.. code-block:: shell

   import psi4
   import adcc
 
   adcc.set_n_threads(4)

   # psi4 SCF calculation
   mol = psi4.geometry(f"""
       0 1
       O          0.00000        0.00000        0.11779
       H          0.00000        0.75545       -0.47116
       H          0.00000       -0.75545       -0.47116
       units angstrom
       symmetry c1
       no_reorient
    """)

    # set the number of cores equal to the auto-determined value from
    # the adcc ThreadPool
    psi4.set_num_threads(adcc.get_n_threads())
    psi4.core.be_quiet()
    psi4.set_options({'basis': basis,
                      'd_convergence': 1e-10,
                      'e_convergence': 1e-12})
    scf_e, wfn = psi4.energy('hf', return_wfn=True)

   # polaritonic_adcc calculation
   exstates = run_qed_adc(wfn, method="adc1", coupl=[0., 0., 0.1], freq=[0., 0., 0.5],
       qed_hf=False, qed_coupl_level=False, n_singlets=5, conv_tol=1e-7
   )
   # since this calculates a full polaritonic matrix, the result is already
   # diagonalized and e.g. the energies can be obtained as follows
   print(exstates.excitation_energy)


Using hilbert as a polaritonic SCF provider currently requires using the psi4
executable, due to a bug in hilbert, with which a calculation can be carried
out as follows

.. code-block:: shell

   import hilbert
   import adcc
   from scipy.linalg import eigh
   
   adcc.set_n_threads(4)

   # hilbert SCF calculation
   molecule h2o {
   O          0.00000        0.00000        0.11779
   H          0.00000        0.75545       -0.47116
   H          0.00000       -0.75545       -0.47116
   units angstrom
   symmetry c1
   no_reorient
   }   

   set {
     basis sto-3g
     scf_type df
     df_basis_scf cc-pv5z-jkfit
     e_convergence 1e-12
     d_convergence 1e-10
   }

   set hilbert {
     n_photon_states          2
     cavity_frequency         [0.0, 0.0, 0.5]
     cavity_coupling_strength [0.0, 0.0, 0.1]
   }

   scf_e, wfn = psi4.energy('polaritonic-uhf', return_wfn=True)

   # polaritonic_adcc calculation
   exstates = run_qed_adc(wfn, method="adc2", coupl=[0., 0., 0.1], freq=[0., 0., 0.5],
       qed_hf=True, qed_coupl_level=1, n_singlets=5, conv_tol=1e-7
   )
   # This calculation builds the polaritonic matrix in the truncated state space
   # of the 5 singlets requested. Since this yields a compact matrix, the full
   # can be build and for maximum flexibility for the user is returned as such.
   # This is often helpful for dynamics models, where e.g. a simple 2x2 model is
   # desired, which can then be extracted from the returned matrix in numpy array
   # format. In case you only want the energies, you can simply diagonalize as 
   print(eigh(exstates.qed_matrix)[0])
   # The polaritonic MP ground state energy can be accessed via
   print(exstate.ground_state.energy())


.. attention::
   As can be seen the polaritonic results need to be accessed in two different
   ways, depending on whether ``qed_coupl_level`` is set to False or an integer.
   To ensure maximum flexibility for the user, the ``exstates.excitation_energy``
   attribute is also populated with ``qed_coupl_level`` set to an integer, but
   these refer to the state basis, from which the polaritonic matrix was build
   from.

.. attention::
   Always double check how the SCF provider defines the coupling strength!!!
   Usually one out of two common definitions is used, which differ by a factor
   of two times the energy of the cavity photon. Currently polaritonic_adcc
   uses the same definition than the hilbert package, but e.g.
   `psi4numpy <https://github.com/psi4/psi4numpy/tree/master/Polaritonic-Quantum-Chemistry>`_ 
   , which can also be used as a polaritonic SCF provider, uses the other definition,
   which needs to be accounted for, when setting the coupling parameter!
  

Basic usage
-----------

For a detailed description of the ranges allowed for the parameters in
``run_qed_adc`` check the :ref:`genindex`.

In a recent study we showed that using the truncated state approach is
usually sufficient up to first order for reasonable isolation from other
states, which also couple to the cavity photon (unpublished results).
This promotes the following recommendations:

1. Unless you want to simulate rather weak couplings, always use a polaritonic
   SCF reference.
2. If the density of states and which also couple to the cavity is large,
   use the full matrix ansatz, but if not, use the first order coupling,
   which is the default, as it is much faster with similar accuracy. 
3. Use at least ADC(2) or ADC(3).


Advanced usage
--------------

The truncated ansatz can account for lossy cavities, by simply including
the energetical damping as imaginary contribution to the cavity photon
energy.

Use tight convergence criteria, as small variations in energy can lead
to large differences near the resonance region of the cavity photon.

All polaritonic methods explicitly include up to double photon excitation,
which means that the returned matrix from the truncated state space
method has the dimension-size of three times the number of states requested
plus the ground state with a singly and a doubly populated photonic contribution.
With a non-polaritonic reference, the excited states are not orthogonal
to the ground state in the photonic vacuum anymore, which is why it needs
to be included into the matrix and therefore appends the size of each
dimension by one.

.. warning::
   Going to larger systems you might experience trouble with psi4 based
   SCF providers, as the export of the wfn object in psi4 is very slow.
   This can simply be circumvented using an other SCF provider and is
   not an issue of polaritonic_adcc!

