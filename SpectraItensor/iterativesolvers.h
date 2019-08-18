#ifndef __ITENSOR_ARNOLDIR_H
#define __ITENSOR_ARNOLDIR_H
#include <math.h>
#include "itensor/all_mps.h"
#include "GenEigsSolver.h"
#include "CplxGenEigsSolver.h"


namespace itensor {

//
// Use the Arnoldi algorithm to find the 
// eigenvector of the nonHermitian matrix A with minimal eigenvalue.
// (BigMatrixT objects must implement the methods product, size and diag.)
// Returns the minimal eigenvalue lambda such that
// A phi = lambda phi.
//
template <class BigMatrixT>
void
arnoldiR(BigMatrixT const& A, 
         ITensor& phi,
         Cplx& eig,
         Args const& args = Args::global())
    {
    auto maxiter_ = args.getSizeT("MaxIter",4);
    auto errgoal_ = args.getReal("ErrGoal",1E-6);
    int krydim_ = (int) args.getSizeT("MaxKrylov",6);
    int maxsize = (int) A.size();
    int krydim = (int) std::min(krydim_,maxsize-1);
    if(krydim < 2) krydim = 2;

    Spectra::CplxGenEigsSolver<double, Spectra::SMALLEST_REAL, BigMatrixT> eigs(&A, 1, krydim);
    eigs.init(phi);
    eigs.compute(maxiter_,errgoal_,Spectra::SMALLEST_REAL);
    eig = eigs.eigenvalues();
    phi = eigs.eigenvectors();

    auto ha = args.getSizeT("DMRGh",0);
    auto b = args.getSizeT("DMRGb",0);
    printf("   --------------------MoreInfo--------------------\n");
    printfln("   Loop(%d, %d) energy: ",ha,b,eig);
    eigs.moreinfo();
    // if(ha==1&&b==6) eigs.moreinfo();
    printf("   --------------------EndMInfo--------------------\n");
    }

template <class BigMatrixT>
void 
arnoldiR(BigMatrixT const& A, 
         ITensor& phi,
         Real& eig,
         Args const& args = Args::global())
    {
    auto maxiter_ = args.getSizeT("MaxIter",4);
    auto errgoal_ = args.getReal("ErrGoal",1E-6);
    int krydim_ = (int) args.getSizeT("MaxKrylov",6);
    int maxsize = (int) A.size();
    int krydim = (int) std::min(krydim_,maxsize-1);
    if(krydim < 3) krydim = 3;
    
    Spectra::GenEigsSolver<double, Spectra::SMALLEST_REAL, BigMatrixT> eigs(&A, 1, krydim);
    eigs.init(phi);
    eigs.compute(maxiter_,errgoal_,Spectra::SMALLEST_REAL);
    auto eigo = eigs.eigenvalues();
    phi = eigs.eigenvectors();

    auto ha = args.getSizeT("DMRGh",0);
    auto b = args.getSizeT("DMRGb",0);
    printfln("   Loop(%d, %d) energy: ",ha,b,eig);
    if(eigo.imag() != 0.)
    {
      printf("   --------------------MoreInfo--------------------\n");
      // std::cout << "   Info: " << eigs.info() << std::endl;
      printf("   Arnoldi(%d, %d) reports energy.imag = %.4e\n",ha,b,eigo.imag());
      printf("   Arnoldi take conjugate pair.\n");
      eigs.moreinfo();
      printf("   --------------------EndMInfo--------------------\n");
    }
    eig = eigo.real();
    }
} //namespace itensor

#endif
