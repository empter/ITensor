#ifndef __ITENSOR_EXPAPPLYH_H
#define __ITENSOR_EXPAPPLYH_H

#include "itensor/all_basic.h"
#include <Eigen/Dense>
// #include <unsupported/Eigen/MatrixFunctions>

namespace Eigen {
  // Chebyshev expansion ExpM U = e^(tau*H)
  template<typename T>
  void
  expm_small(int N, int M, T* Hptr, T* Uptr, T tau)
  {
    std::vector<double>
    bessj = {1.266065877752008,1.13031820798497,
             2.714953395340766,4.43368498486638,
             5.474240442093732,5.429263119139439,
             4.497732295429515,3.19843646240199,
             1.992124806672796,1.103677172551734,
             5.505896079673748,2.497956616984982,
             1.03915223067857,3.991263356414401,
             1.423758010825657,4.740926102561494,
             1.480180057208297,4.349919494944169,
             1.207428927279753,3.175356737059445};
    std::vector<double>
    bessjf = {1.,1.,1E-1,1E-2,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-10,
              1E-11,1E-12,1E-14,1E-15,1E-17,1E-18,1E-20,1E-21,1E-23};
    Map<Matrix<T,Dynamic,Dynamic>> hmat(Hptr, N, N);
    Matrix<T,Dynamic,Dynamic> umat = Matrix<T,Dynamic,Dynamic>(M, M);
    Matrix<T,Dynamic,Dynamic> t0 = Matrix<T,Dynamic,Dynamic>::Identity(M,M);
    Matrix<T,Dynamic,Dynamic> t1 = hmat.block(0,0,M,M) * tau;
    auto nrm = t1.norm();
    int pn = (int) std::ceil(std::log2(nrm));
    if(pn >3) std::cout << "too big norm: " << nrm << std::endl << std::endl;
    if(pn > 0) t1 /= std::pow(2,pn);
    Matrix<T,Dynamic,Dynamic> m = 2.0 * t1;
    umat = t0 * bessj[0];
    umat += t1 * bessj[1];
    for(int i = 2; i < 19; ++i)
    {
      Matrix<T,Dynamic,Dynamic> tk = m * t1 - t0;
      Matrix<T,Dynamic,Dynamic> ire = (tk * bessj[i]) * bessjf[i];
      umat += ire;
      // if((ire.nrm() < 1E-16))  break;
      t0 = t1;
      t1 = tk;
    }
    if(pn > 0)
    {
      for(int i = 0; i < pn; ++i) umat = umat * umat;
    }
    for(int i = 0; i < M; ++i, ++Uptr) *Uptr = umat(i,0);
  }

  // template<typename T>
  // void
  // expm_small(int N, int M, T* Hptr, T* Uptr, T tau)
  // {
  //   Map<Matrix<T,Dynamic,Dynamic>> hmat(Hptr, N, N);
  //   Matrix<T,Dynamic,Dynamic> umat = Matrix<T,Dynamic,Dynamic>(M, M);
  //   umat = (tau * hmat.block(0,0,M,M)).exp();
  //   for(int i = 0; i < M; ++i, ++Uptr) *Uptr = umat(i,0);
  //   // std::cout.precision(12);
  //   // std::cout << "h=" << hmat.block(0,0,M,M) << std::endl << std::endl;
  //   // std::cout << "u=" << umat.block(0,0,M,1) << std::endl << std::endl;
  // }
}


namespace itensor{
ITensor
matdot(ITensor& A, ITensor& B, Index& ind)
{
  auto AB = A * prime(B);
  AB = swapTags(AB,"2","1");
  // AB = permute(AB,ind,prime(ind));
  return AB;
}

// Chebyshev expansion e^(H)
ITensor
ChebyshevExpm(ITensor& H)
{
  auto inds = stdx::reserve_vector<Index>(order(H)/2);
  for(auto& i : H.inds()) {if(i.primeLevel() == 0) inds.push_back(i);}
  auto [comb,cind] = combiner(std::move(inds));
  H = comb * H;
  auto combP = dag(prime(comb));
  H = combP * H;
  if(H.order() != 2)
      {
      Error("H must be matrix-like (order 2)");
      }

    const  std::vector<Real>
    bessj = {1.266065877752008,1.13031820798497,
             2.714953395340766,4.43368498486638,
             5.474240442093732,5.429263119139439,
             4.497732295429515,3.19843646240199,
             1.992124806672796,1.103677172551734,
             5.505896079673748,2.497956616984982,
             1.03915223067857,3.991263356414401,
             1.423758010825657,4.740926102561494,
             1.480180057208297,4.349919494944169,
             1.207428927279753,3.175356737059445};
    const std::vector<Real>
    bessjf = {1.,1.,1E-1,1E-2,1E-3,1E-4,1E-5,1E-6,1E-7,1E-8,1E-10,
              1E-11,1E-12,1E-14,1E-15,1E-17,1E-18,1E-20,1E-21,1E-23};

  auto t0 = H;
  t0.fill(0.);
  for(auto i : range1(dim(cind))) t0.set(i,i,1.0);
  auto nrm = norm(H);
  int pn = (int) std::ceil(std::log2(nrm));
  // Print(pn);
  if(pn > 4) printfln("ITensor::ChebyshevExpm: matrix norm %.8f too big.",nrm);
  if(pn > 0) H *= 1./std::pow(2,pn);
  auto t1 = H;
  auto res = t0 * bessj[0];
  res += t1 * bessj[1];

  H *= 2.0;
  for(int i = 2; i < 19; ++i)
  {
    auto tk = matdot(H,t1,cind) - t0;
    res += (tk * bessj[i]) * bessjf[i];
    t0 = t1;
    t1 = tk;
  }
  if(pn > 0)
  {
    for(auto i : range(pn)) res = matdot(res,res,cind);
  }
  res *= dag(comb);
  res *= dag(combP);
  return res;
}
// 
// void inline
// eltRC(ITensor const& A, long r, long c, Real* ptr)
//     {
//     *ptr = elt(A,r,c);
//     }
// 
// void inline
// eltRC(ITensor const& A, long r, long c, Cplx* ptr)
//     {
//     *ptr = eltC(A,r,c);
//     }

// // return U = e^(tau*H(0,M,0,M)) for small M
// template<typename T>
// void
// expm_small(int N, int M, T* Hptr, T* Uptr, T tau)
// {
//   // Eigen::expm_small(N,M,Hptr,Uptr,tau);
//   // return;
//   auto i = Index(M);
//   auto H = ITensor(i,prime(i));
//   for(auto j : range1(N))
//     for(auto i : range1(N))
//       {if(i <= M && j <= M) H.set(i,j,*Hptr); ++Hptr;}
//   H *= tau;
//   auto Hm = ChebyshevExpm(H,i);
//   for(auto i : range1(M))
//     {eltRC(Hm,i,1,Uptr);++Uptr;}
// }

//infinity norm: max(abs(list))
template<typename T>
Real
norm_inf(std::vector<T> const& list)
{
  Real norminf = 0.0;
  for(auto i : range(list.size()))
    {
      Real abs = std::abs(list[i]);
      if(abs > norminf) norminf = abs;
    }
  return norminf;
}

void inline
dot(ITensor const& A, ITensor const& B, Real& res)
    {
    res = elt(dag(A)*B);
    }

void inline
dot(ITensor const& A, ITensor const& B, Cplx& res)
    {
    res = eltC(dag(A)*B);
    }

// Krylov subspace: phi = exp(tau*A.localh)*phi
template<typename T, typename BigMatrixT>
void
expApplyHImpl(BigMatrixT const& A, ITensor& phi, T tau, Direction dir, Args& args)
{
  auto maxm = args.getInt("MaxKrylov",40);
  auto tol = args.getReal("ErrGoal",1E-12);
  auto debug_level_ = args.getInt("DebugLevel",-1);
  auto len = 1;
  for(auto& I : inds(phi)) len *= dim(I);
  
  // exact expH when dim is not large
  if(len < 40)
  {
    if(debug_level_ > 0)
        println("expApplyHImpl: use exact expm.");
    ITensor heff;
    if(dir == NoDir) A.localh(heff);
    else A.localhnext(heff,dir);
    heff *= tau;
    phi = noPrime(ChebyshevExpm(heff)*phi);
    return;
  }
  
  std::vector<T> H;// using column major
  H.resize((maxm+1)*(maxm+1),0.0);
  std::vector<ITensor> V(maxm+1);
  auto pnorm = norm(phi);
  V[0] = phi/pnorm;

  std::vector<T> S = {tau/100.0, tau/3.0, tau*2.0/3.0, tau};
  // std::vector<T> S = {tau/100.0, tau/2.0, tau};
  std::vector<T> jnorm;
  jnorm.resize(S.size(),0.0);
  std::vector<T> u;
  auto curm = 0;
  
  for(auto j : range(maxm))
  {
    ITensor w;
    if(dir == NoDir) A.product(V[j],w);
    else A.productnext(V[j],w,dir);
    for(auto i : range(j+1))
    {
      T helt;
      dot(w,V[i],helt);
      H[(maxm+1)*j+i] = helt;
      w -= helt * V[i];
    }
    auto wnorm = norm(w);
    H[(maxm+1)*j+j+1] = wnorm;
    
    for(auto i : range(S.size()))
    {
      u.resize(j+1,0.0);
      Eigen::expm_small(maxm+1,j+1,H.data(),u.data(),S[i]);
      jnorm[i] = - wnorm * u.back();
    }
    auto resnorm = norm_inf(jnorm);
    if(debug_level_ > 0)
        println("residual norm = ", resnorm);
    curm = j;

    if(resnorm < tol) break;
    V[j+1] = w/wnorm;
  }

  // phi = pnorm*V*u
  phi = u[curm] * V[curm];
  for(auto i : range(curm)) phi += u[i] * V[i];
  phi = pnorm * phi;

  if(curm == maxm-1)
  {
    args.add("KrylovNoCover",true);
    if(debug_level_ > 0)
        println("expApplyH no convergence. Try raising 'MaxKrylov'.");
  }
}

template<typename BigMatrixT>
void
expApplyH(BigMatrixT const& A,
      ITensor& phi,
      Cplx tau,
      Direction dir,
      Args& args)
    {
    auto debug_level_ = args.getInt("DebugLevel",-1);

    if(debug_level_ > 0)
        println("Calling complex version of expApplyHImpl()");
    expApplyHImpl<Cplx>(A,phi,tau,dir,args);
    }

template<typename BigMatrixT>
void
expApplyH(BigMatrixT const& A,
      ITensor& phi,
      Real tau,
      Direction dir,
      Args& args)
    {
    auto debug_level_ = args.getInt("DebugLevel",-1);
    
    ITensor Ax;
    if(dir == NoDir) A.product(phi, Ax);
    else A.productnext(phi, Ax, dir);
    if(isComplex(phi) || isComplex(Ax))
    {
      if(debug_level_ > 0)
          println("Calling complex version of expApplyHImpl()");
      Cplx tauc = tau;
      expApplyHImpl<Cplx>(A,phi,tauc,dir,args);
    }
    else
    {
      if(debug_level_ > 0)
          println("Calling real version of expApplyHImpl()");
      expApplyHImpl<Real>(A,phi,tau,dir,args);
    }
    }
}//namespace itensor

#endif
