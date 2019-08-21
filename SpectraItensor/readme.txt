GenEigsSolver Ported from https://github.com/yixuan/spectra
An additional CplxGenEigsSolver based on GenEigsSolver.
GenEigsSolver working with itensor LocalMPO class, it can only return Ground state of a LocalMPO object (both H and psi must be real). For complex H or psi, use cplxdmrg which calls CplxGenEigsSolver.
Note: DMRG may not convergent for some non-Hermitian systems (conditions).
