#ifndef __ITENSOR_ARNOLDI_H
#define __ITENSOR_ARNOLDI_H
#include <math.h>
#include "itensor/all_mps.h"
#include "GenEigsSolver.h"


namespace itensor {

//
// Use the Arnoldi algorithm to find the 
// eigenvector of the Hermitian matrix A with minimal eigenvalue.
// (BigMatrixT objects must implement the methods product, size and diag.)
// Returns the minimal eigenvalue lambda such that
// A phi = lambda phi.
//
template <class BigMatrixT>
Real 
arnoldi(BigMatrixT const& A, 
         ITensor& phi,
         Args const& args = Args::global())
    {
    auto maxiter_ = args.getSizeT("MaxIter",2);
    auto errgoal_ = args.getReal("ErrGoal",1E-6);
    int krydim_ = (int) args.getSizeT("MaxKrylov",6);
    int maxsize = (int) A.size();
    int krydim = (int) std::min(krydim_,maxsize-1);
    if(krydim < 3) krydim = 3;
    
    Spectra::GenEigsSolver<double, Spectra::SMALLEST_REAL, BigMatrixT> eigs(&A, 1, krydim);
    eigs.init(phi);
    eigs.compute(maxiter_,errgoal_,Spectra::SMALLEST_REAL);
    // std::cout << "Info: " << eigs.info() << std::endl;
    auto eig = eigs.eigenvalues();
    phi = eigs.eigenvectors();

    if(std::fabs(eig.imag()) > 1E-14) printf("   Arnoldi reports energy.imag = %.4e\n",eig.imag());
    if(isComplex(phi))
    {
      auto ha = args.getSizeT("DMRGh",0);
      auto b = args.getSizeT("DMRGb",0);
      printf("   Arnoldi ComplexVector (%d, %d), take real part.\n",ha,b);
      phi.takeReal();
      phi /= norm(phi);
    }
    return eig.real();
    }

} //namespace itensor

#endif
