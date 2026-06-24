# Carter Notes: qDRIFT UQK Runtime Strategy

The broad goal is to understand when stochastic qDRIFT UQK gives results that are close enough to the full-Trotter UQK reference to be scientifically useful, while using circuits that are meaningfully shallower. The main comparison should be against full Trotterization first, not directly against FCI, because we want to separate qDRIFT error from Krylov-subspace error, Trotter error, and noise. For H2 and H4, try to get a feeling for how the overlap matrix \(S\), the correlation values \(C_k\), and the final UQK energy estimates change as you vary `QDRIFT_SEGMENT_COUNT_ND`, `DT`, `KRYLOV_DIMENSION`, `SHOTS_PER_MFE_EXPERIMENT`, and `STOCHASTIC_INSTANCES_PER_CORRELATION`.

A useful way to think about the problem is that there are several competing limits. Larger `N_d` should make the qDRIFT channel closer to the intended time evolution, but it also makes each sampled circuit deeper. More stochastic instances should reduce random qDRIFT sampling noise, while more shots should reduce ordinary MFE measurement noise. Smaller `DT` usually makes time evolution easier to approximate, but may give a less informative Krylov basis; larger `KRYLOV_DIMENSION` may improve the subspace but can make the overlap matrix less well-conditioned and requires longer-time correlations. The interesting part is not just finding "the biggest parameters we can afford", but finding where the tradeoffs actually become reasonable.

I would also like you to look for ways to reduce the primitive qDRIFT block depth. Some possibilities are fairly practical: compare Qiskit transpiler optimization levels, compare `EVOLUTION_METHOD = "pauli_evolution_gate"` against `manual_ladder`, inspect whether the deepest or most frequently sampled groups have repeated Pauli-string structure that could be synthesized more efficiently, and see whether IBM-targeted transpilation changes the depth/gate-count picture. More exploratory possibilities include testing Bravyi-Kitaev instead of Jordan-Wigner, coefficient thresholding of tiny Pauli terms, or eventually symmetry tapering. None of these are guaranteed wins, so the useful result may simply be a clear plot showing that a tempting idea does not help for H4.

After getting a noiseless picture, repeat the most promising parameter choices with the noisy backends, especially `local_noisy_simple` and `local_noisy_ibm_model`. My guess is that the noiseless optimum and noisy optimum may differ: in a noisy simulation, a smaller `N_d` might sometimes beat a more accurate but deeper qDRIFT approximation. Try to identify whether there is a practical "sweet spot" where the overlap matrix and energy estimates are still recognizable while the circuit depth is low enough that the noisy simulation is not completely washed out.

Useful entry points are the two workflow notebooks:

- `notebooks/01_qforte_build_molecule_and_jw_blocks.ipynb`
- `notebooks/02_qiskit_uqk_overlap_and_energy.ipynb`

Useful implementation references are:

- `scripts/qiskit/build_uqk_overlap_matrix.py`
- `scripts/qiskit/solve_uqk_gep.py`
- `notebooks/workflow_helpers/qiskit_uqk_helpers.py`
- `notebooks/workflow_helpers/qforte_export_helpers.py`
- `notes/stochastic_srqk.tex`

Good plots to make include overlap-matrix error versus qDRIFT depth, UQK energy error versus qDRIFT depth, noisy versus noiseless energy estimates, and primitive grouped-circuit depth histograms. A compact table of the best few H2 and H4 parameter choices would also be very helpful.
