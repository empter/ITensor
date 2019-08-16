// Copyright (C) 2016-2019 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CPLX_UPPER_HESSENBERG_QR_H
#define CPLX_UPPER_HESSENBERG_QR_H

#include <Eigen/Core>
#include <cmath>      // std::sqrt
#include <algorithm>  // std::fill, std::copy
#include <stdexcept>  // std::logic_error
#include <complex>

namespace std {
  using LAPACK_INT = int;
  using LAPACK_REAL = double;
  using cplx = complex<double>;
  typedef struct
  {
  LAPACK_REAL real, imag;
  } LAPACK_COMPLEX;
  
  extern "C" void zlartg_(LAPACK_COMPLEX* f, LAPACK_COMPLEX* g, LAPACK_REAL* cs, LAPACK_COMPLEX* sn, LAPACK_COMPLEX* r);
  
  void callzlattg(cplx& f, cplx& g, double& c, cplx& s, cplx& r)
  {
    auto pf = reinterpret_cast<LAPACK_COMPLEX*>(&f);
    auto pg = reinterpret_cast<LAPACK_COMPLEX*>(&g);
    auto ps = reinterpret_cast<LAPACK_COMPLEX*>(&s);
    auto pr = reinterpret_cast<LAPACK_COMPLEX*>(&r);
    zlartg_(pf,pg,&c,ps,pr);
  }
}

namespace Spectra {

template <typename Scalar = double>
class CplxUpperHessenbergQR
{
private:
    typedef Eigen::Index Index;
    typedef std::complex<Scalar> Complex;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
    typedef Eigen::Ref<ComplexMatrix> GenericMatrix;
    typedef const Eigen::Ref<const ComplexMatrix> ConstGenericMatrix;

protected:
    Index m_n;
    Index cnt;
    // Gi = [ cos[i]    sin[i]]
    //      [-sin[i]^*  cos[i]]
    // Q = G1 * G2 * ... * G_{n-1}
    Complex m_shift;
    bool m_computed;
    ComplexMatrix m_mat_T;

public:
    ///
    /// Constructor to preallocate memory. Computation can
    /// be performed later by calling the compute() method.
    ///
    CplxUpperHessenbergQR(Index size) :
        m_n(size),
        cnt(0),
        m_computed(false)
    {}

    ///
    /// Virtual destructor.
    ///
    virtual ~CplxUpperHessenbergQR() {}

    const ComplexMatrix& updatedH() const { return m_mat_T; }

    virtual void compute(ConstGenericMatrix& mat, ComplexMatrix& q, const Complex& shift = Scalar(0))
    {
        m_n = mat.rows();
        if(m_n != mat.cols())
            throw std::invalid_argument("CplxUpperHessenbergQR: matrix must be square");
        if(m_n != q.cols() || m_n != q.rows())
            throw std::invalid_argument("CplxUpperHessenbergQR: Q not appropriate");

        m_shift = shift;
        m_mat_T.resize(m_n, m_n);
        cnt++;

        // Make a copy of mat
        std::copy(mat.data(), mat.data() + mat.size(), m_mat_T.data());

        Complex s, r;
        Scalar c;
        Complex h11 = m_mat_T(0,0);
        Complex h21 = m_mat_T(1,0);
        Complex f = h11 - m_shift;
        Complex g = h21;
        for(Index i = 0; i < m_n-1; i++)
        {
          // Construct the plane rotation
          std::callzlattg(f,g,c,s,r);
          if(i > 0)
          {
            m_mat_T(i,i-1) = r;
            m_mat_T(i+1,i-1) = Complex(0,0);
          }
          
          // Apply rotation to the left of H
          for(Index j = i; j < m_n; j++)
          {
            Complex tmp = c*m_mat_T(i,j) + s*m_mat_T(i+1,j);
            m_mat_T(i+1,j) = -conj(s)*m_mat_T(i,j) + c*m_mat_T(i+1,j);
            m_mat_T(i,j) = tmp;
          }
          
          // Apply rotation to the right of H
          for(Index j = 0; j <= std::min(i+2,m_n-1); j++)
          {
            Complex tmp = c*m_mat_T(j,i) + conj(s)*m_mat_T(j,i+1);
            m_mat_T(j,i+1) = -s*m_mat_T(j,i) + c*m_mat_T(j,i+1);
            m_mat_T(j,i) = tmp;
          }
          
          //Accumulate the rotation in the matrix Q
          for(Index j = 0; j <= std::min(i+cnt,m_n-1); j++)
          {
            Complex tmp = c*q(j,i) + conj(s)*q(j,i+1);
            q(j,i+1) = -s*q(j,i) + c*q(j,i+1);
            q(j,i) = tmp;
          }
          
          // Prepare for next rotation
          if(i < m_n-2)
          {
            f = m_mat_T(i+1,i);
            g = m_mat_T(i+2,i);
          }
        }

        m_computed = true;
    }

};

} // namespace Spectra

#endif // CPLX_UPPER_HESSENBERG_QR_H
