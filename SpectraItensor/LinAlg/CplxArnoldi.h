// Copyright (C) 2018-2019 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CPLX_ARNOLDI_H
#define CPLX_ARNOLDI_H

#include <Eigen/Core>
#include <cmath>      // std::sqrt
#include <stdexcept>  // std::invalid_argument
#include <sstream>    // std::stringstream
#include <complex>

#include "../Util/TypeTraits.h"

namespace Spectra {


// Arnoldi factorization A * V = V * H + f * e'
// A: n x n
// V: n x k
// H: k x k
// f: n x 1
// e: [0, ..., 0, 1]
// V and H are allocated of dimension m, so the maximum value of k is m
template <typename Scalar, typename OpType>
class CplxArnoldi
{
private:
    typedef Eigen::Index Index;
    typedef std::complex<Scalar> Complex;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic> ComplexMatrix;
    typedef Eigen::Matrix<Complex, Eigen::Dynamic, 1> ComplexVector;
    typedef Eigen::Map<ComplexVector> MapVec;
    typedef Eigen::Map<const ComplexVector> MapConstVec;
    typedef const Eigen::Ref<const ComplexMatrix> ConstGenericMatrix;
    typedef std::vector<itensor::ITensor> Tv;
    typedef itensor::ITensor Ten;

protected:
    const OpType& m_op;       // Operators for the Arnoldi factorization

    const Index m_n;          // dimension of A
    const Index m_m;          // maximum dimension of subspace V
    Index       m_k;          // current dimension of subspace V

    Tv m_fac_V;           // V matrix in the Arnoldi factorization
    ComplexMatrix m_fac_H;    // H matrix in the Arnoldi factorization
    Ten m_fac_f;           // residual in the Arnoldi factorization
    Scalar m_beta;            // ||f||, B-norm of f

    const Scalar m_near_0;    // a very small value, but 1.0 / m_near_0 does not overflow
                              // ~= 1e-307 for the "double" type
    const Scalar m_eps;       // the machine precision, ~= 1e-16 for the "double" type

    // Given orthonormal basis functions V, find a nonzero vector f such that V'Bf = 0
    // Assume that f has been properly allocated
    void expand_basis(Tv const& V, const Index lenv, Ten& f, Scalar& fnorm)
    {
        using std::sqrt;

        const Scalar thresh = m_eps * sqrt(Scalar(m_n));
        ComplexVector Vf(lenv);
        for(Index iter = 0; iter < 5; iter++)
        {
            // Randomly generate a new vector and orthogonalize it against V
            f.randomize();
            // f <- f - V * V'Bf, so that f is orthogonal to V in B-norm
            for(Index i = 0; i < lenv; i++)
            {
              Vf[i] = eltC(dag(V[i])*f);
            }
            for(Index i = 0; i < lenv; i++)
            {
              f -= V[i] * Vf[i];
            }
            // fnorm <- ||f||
            fnorm = norm(f);

            // If fnorm is too close to zero, we try a new random vector,
            // otherwise return the result
            if(fnorm >= thresh)
                return;
        }
    }

public:
    CplxArnoldi(OpType const* op, Index m) :
        m_op(*op), m_n(m_op.size()), m_m(m), m_k(0),
        m_near_0(TypeTraits<Scalar>::min() * Scalar(10)),
        m_eps(Eigen::NumTraits<Scalar>::epsilon())
    {}

    virtual ~CplxArnoldi() {}

    // Const-reference to internal structures
    const Tv& matrix_V() const { return m_fac_V; }
    const ComplexMatrix& matrix_H() const { return m_fac_H; }
    const Ten& vector_f() const { return m_fac_f; }
    Scalar f_norm() const { return m_beta; }
    Index subspace_dim() const { return m_k; }
    Index size() const { return m_n; }

    // Initialize with an operator and an initial vector
    void init(const Ten& v0, Index& op_counter)
    {
        m_fac_V.resize(m_m);
        m_fac_H.resize(m_m, m_m);
        m_fac_H.setZero();

        // Verify the initial vector
        const Scalar v0norm = norm(v0);
        if(v0norm < m_near_0)
            throw std::invalid_argument("initial residual vector cannot be zero");

        // Normalize
        m_fac_V[0] = v0 / v0norm;

        // Compute H and f
        m_op.product(m_fac_V[0], m_fac_f);
        op_counter++;

        m_fac_H(0, 0) = eltC((dag(m_fac_V[0])*m_fac_f));
        m_fac_f -= m_fac_V[0] * m_fac_H(0, 0);

        // In some cases f is zero in exact arithmetics, but due to rounding errors
        // it may contain tiny fluctuations. When this happens, we force f to be zero
        Scalar max_mag = Scalar(0);
        if(isComplex(m_fac_f))
        {
          auto maxComp = [&max_mag](Complex r)
          {
            if(std::abs(r) > max_mag) max_mag = std::abs(r);
          };
          m_fac_f.visit(maxComp);
        }
        else
        {
          auto maxComp = [&max_mag](Scalar r)
          {
            if(std::fabs(r) > max_mag) max_mag = std::fabs(r);
          };
          m_fac_f.visit(maxComp);
        }
        
        if(max_mag < m_eps)
        {
            m_fac_f *= 0.;
            m_beta = Scalar(0);
        } else {
            m_beta = norm(m_fac_f);
        }

        // Indicate that this is a step-1 factorization
        m_k = 1;
    }

    // Arnoldi factorization starting from step-k
    virtual void factorize_from(Index from_k, Index to_m, Index& op_counter)
    {
        using std::sqrt;

        if(to_m <= from_k) return;

        if(from_k > m_k)
        {
            std::stringstream msg;
            msg << "CplxArnoldi: from_k (= " << from_k <<
                   ") is larger than the current subspace dimension (= " <<
                   m_k << ")";
            throw std::invalid_argument(msg.str());
        }

        const Scalar beta_thresh = m_eps * sqrt(Scalar(m_n));

        // Pre-allocate vectors
        ComplexVector Vf(to_m);
        Ten w;

        // Keep the upperleft k x k submatrix of H and set other elements to 0
        m_fac_H.rightCols(m_m - from_k).setZero();
        m_fac_H.block(from_k, 0, m_m - from_k, from_k).setZero();

        for(Index i = from_k; i <= to_m - 1; i++)
        {
            bool restart = false;
            // If beta = 0, then the next V is not full rank
            // We need to generate a new residual vector that is orthogonal
            // to the current V, which we call a restart
            if(m_beta < m_near_0)
            {
                expand_basis(m_fac_V, i, m_fac_f, m_beta); // The first i columns of m_fac_V
                restart = true;
            }

            // v <- f / ||f||
            if(m_beta > m_near_0)
            {
              m_fac_V[i] = m_fac_f / m_beta; // The (i+1)-th column
            }
            else // expand_basis may failed in some cases
            {
              m_fac_V[i] = Scalar(0) * m_fac_f;
              m_beta = Scalar(0);
            }  

            // Note that H[i+1, i] equals to the unrestarted beta
            m_fac_H(i, i - 1) = restart ? Scalar(0) : m_beta;

            // w <- A * v, v = m_fac_V.col(i)
            m_op.product(m_fac_V[i], w);
            op_counter++;

            const Index i1 = i + 1;
            // h = m_fac_H(0:i, i)
            MapVec h(&m_fac_H(0, i), i1);
            // h <- V'Bw
            for(Index hl = 0; hl < i1; hl++) // First i+1 columns of m_fac_V
            {
              h[hl] = eltC(dag(m_fac_V[hl])*w);
            }

            // f <- w - V * h
            m_fac_f = w - m_fac_V[0] * h[0];
            for(Index hl = 1; hl < i1; hl++)
            {
              m_fac_f -= m_fac_V[hl] * h[hl];
            }
            m_beta = norm(m_fac_f);

            if(m_beta > Scalar(0.717) * h.norm())
                continue;

            // f/||f|| is going to be the next column of V, so we need to test
            // whether V'B(f/||f||) ~= 0
            for(Index hl = 0; hl < i1; hl++)
            {
              Vf[hl] = eltC(dag(m_fac_V[hl])*m_fac_f);
            }
            Scalar ortho_err = Vf.head(i1).cwiseAbs().maxCoeff();
            // If not, iteratively correct the residual
            int count = 0;
            while(count < 5 && ortho_err > m_eps * m_beta)
            {
                // There is an edge case: when beta=||f|| is close to zero, f mostly consists
                // of noises of rounding errors, so the test [ortho_err < eps * beta] is very
                // likely to fail. In particular, if beta=0, then the test is ensured to fail.
                // Hence when this happens, we force f to be zero, and then restart in the
                // next iteration.
                if(m_beta < beta_thresh)
                {
                    m_fac_f *= Scalar(0);
                    m_beta = Scalar(0);
                    break;
                }

                // f <- f - V * Vf
                for(Index hl = 0; hl < i1; hl++)
                {
                  m_fac_f -= m_fac_V[hl] * Vf[hl];
                }
                // h <- h + Vf
                h.noalias() += Vf.head(i1);
                // beta <- ||f||
                m_beta = norm(m_fac_f);

                for(Index hl = 0; hl < i1; hl++)
                {
                  Vf[hl] = eltC(dag(m_fac_V[hl])*m_fac_f);
                }
                ortho_err = Vf.head(i1).cwiseAbs().maxCoeff();
                count++;
            }
        }

        // Indicate that this is a step-m factorization
        m_k = to_m;
    }

    // Update modified H -> Q'HQ from an upper Hessenberg QR decomposition
    void update_H(ConstGenericMatrix& uph)
    {
        if(uph.size() != m_fac_H.size()) throw std::invalid_argument("CplxArnoldi: update matrix error!");
        std::copy(uph.data(), uph.data() + uph.size(), m_fac_H.data());
        m_k--;
    }

    // Apply V -> VQ and compute the new f.
    // Should be called after compress_H(), since m_k is updated there.
    // Only need to update the first k+1 columns of V
    // The first (m - k + i) elements of the i-th column of Q are non-zero,
    // and the rest are zero
    void compress_V(const ComplexMatrix& Q)
    {
        Tv Vs(m_k + 1);
        for(Index i = 0; i < m_k; i++)
        {
            const Index nnz = m_m - m_k + i + 1;
            MapConstVec q(&Q(0, i), nnz);
            Vs[i] = m_fac_V[0] * q[0];
            for(Index j = 1; j < nnz; j++)
            {
              Vs[i] += m_fac_V[j] * q[j];
            }
        }
        Vs[m_k] = m_fac_V[0] * Q(0,m_k);
        for(Index j = 1; j < m_m; j++)
        {
          Vs[m_k] += m_fac_V[j] * Q(j,m_k);
        }
        for(Index j = 0; j < m_k+1; j++)
        {
          m_fac_V[j] = Vs[j];
        }

        m_fac_f = m_fac_f * Q(m_m - 1, m_k - 1) + m_fac_V[m_k] * m_fac_H(m_k, m_k - 1);
        m_beta = norm(m_fac_f);
    }
};


} // namespace Spectra

#endif // CPLX_ARNOLDI_H
