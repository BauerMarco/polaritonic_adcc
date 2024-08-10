.. Polaritonic_adcc documentation master file

Welcome to Polaritonic_adcc's documentation!
============================================

Polaritonic_adcc provides a wide variety of methodologies to determine
the electronic structure of molecules and clusters, which are polaritonically
coupled to a photon. A calculation can be carried out using an SCF provider,
exporting the wfn object, and then executing polaritonic_adcc on top of that,
as is done in the following example.


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


As the name already suggests, this project is based on
`adcc <https://adc-connect.org/v0.15.13/index.html>`_ and adapts its
functionalities, so you can for instance freeze ortbials or request
only certain spin configurations. 

Polaritonic_adcc is also very flexible with its in- and outputs. You can
feed it a standard or a polaritonic SCF result, and request the states
and energies of the full polaritonic matrix, or the whole matrix, if the
matrix is build in a truncated space of states. The latter representation
is also quasi-diabatic, making it a great tool for subsequent quantum dynamics
simulations.

Sounds interesting? Then check out the :ref:`installation` and
:ref:`usage` sections of this documentation or get in touch via
`GitHub <https://github.com/BauerMarco/polaritonic_adcc>`_ .


.. toctree::
   :maxdepth: 2
   
   installation
   usage
   theory
   troubleshooting


* :ref:`genindex`

