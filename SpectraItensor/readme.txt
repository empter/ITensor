GenEigsSolver Ported from https://github.com/yixuan/spectra
An additional CplxGenEigsSolver based on GenEigsSolver.
GenEigsSolver working with itensor LocalMPO class, it can only return Ground state of a LocalMPO object (both H and psi must be real). For complex H or psi, use cplxdmrg which calls CplxGenEigsSolver.
Note: DMRG may not convergent for non-Hermitian systems.
Notes: itensor useing matrix notation which is not common with usual notation, if you need right eigenvector of H, prepare AutoMPO from dag(H) instead of H itself.