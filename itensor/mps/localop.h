//
// Distributed under the ITensor Library License, Version 1.2
//    (See accompanying LICENSE file.)
//
#ifndef __ITENSOR_LOCAL_OP
#define __ITENSOR_LOCAL_OP
//#include "itensor/mps/mpo.h"//cannot include it, since mpo.cc and mps.cc include localop.h
#include "itensor/iqtensor.h"
//#include "itensor/util/print_macro.h"

namespace itensor {

//
// The LocalOp class represents
// an MPO or other operator that
// has been projected into the
// reduced Hilbert space of 
// two sites (default) of an MPS.
//
//   .-              -.
//   |    |      |    |
//   L - Op1 -- Op2 - R
//   |    |      |    |
//   '-              -'
//
// (Note that L, Op1, Op2, (Op3) and R
//  are not required to have this
//  precise structure. L and R
//  can even be null in which case
//  they will not be used.)
//

//  To prevent the danger of accessing 
//  the uninitialized Op3 for the 
//  two-site LocalOp,
//  I change the separate OpX_'s to 
//  an array or vector whose size and memory 
//  is dynamically allocated during runtime.
//  nc_ is passed from the size of the vector.
//  Currently, nc_ cannot be larger than 3.


template <class Tensor>
class LocalOp
    {
    std::vector<Tensor const*> Ops_;// not std:vector<Tensor>* Ops_, since Op1_, Op2_, ... are not necessarily stored in a continuous memory
//    Tensor const* Op1_;
//    Tensor const* Op2_;
    Tensor const* L_;
    Tensor const* R_;
    int nc_;
    mutable long size_;
    public:

    using IndexT = typename Tensor::index_type;

    //
    // Constructors
    //

    LocalOp();

    LocalOp(Tensor const& Op1, 
            Tensor const& Op2,
            Args const& args = Global::args());

    LocalOp(Tensor const& Op1, 
            Tensor const& Op2,
            Tensor const& Op3,
            Args const& args = Global::args());

//    LocalOp(const MPOt<Tensor>& Ops,
//            Args const& args = Global::args());

    LocalOp(Tensor const& Op1, 
            Tensor const& Op2, 
            Tensor const& L, 
            Tensor const& R,
            Args const& args = Global::args());

    LocalOp(Tensor const& Op1, 
            Tensor const& Op2,
            Tensor const& Op3, 
            Tensor const& L, 
            Tensor const& R,
            Args const& args = Global::args());

//    LocalOp(const MPOt<Tensor>& Ops,
//            Tensor const& L, 
//            Tensor const& R,
//            Args const& args = Global::args());

    //
    // Sparse Matrix Methods
    //

    void
    product(Tensor const& phi, Tensor & phip) const;

    Real
    expect(Tensor const& phi) const;

    Tensor
    deltaRho(Tensor const& rho, 
             Tensor const& combine, 
             Direction dir) const;

    Tensor
    diag() const;

    long
    size() const;

    //
    // Accessor Methods
    //

    void
    update(Tensor const& Op1, Tensor const& Op2);

    void
    update(Tensor const& Op1, Tensor const& Op2, Tensor const& Op3);

//    void
//    update(const MPOt<Tensor>& Ops);

    void
    update(Tensor const& Op1, 
           Tensor const& Op2, 
           Tensor const& L, 
           Tensor const& R);

    void
    update(Tensor const& Op1, 
           Tensor const& Op2, 
           Tensor const& Op3,
           Tensor const& L, 
           Tensor const& R);

//    void
//    update(const MPOt<Tensor>& Ops,
//           Tensor const& L, 
//           Tensor const& R);

    Tensor const&
    Op1() const 
        { 
        if(!(*this)) Error("LocalOp is default constructed");
        return *(Ops_[0]);
        }

    Tensor const&
    Op2() const 
        { 
        if(!(*this)) Error("LocalOp is default constructed");
        if(nc_ < 2) Error("LocalOp has less than 2 Ops");
        return *(Ops_[1]);
        }

    Tensor const&
    Op3() const 
        { 
        if(!(*this)) Error("LocalOp is default constructed");
        if(nc_ < 3) Error("LocalOp has less than 3 Ops");
        return *(Ops_[2]);
        }

    int
    numCenter() const { return nc_; }
    void
    numCenter(int val)
    	{
        if(val<1) Error("numCenter must be set >= 1");
        if(val != nc_)
            {
	        nc_ = val;
	        Ops_.resize(val,nullptr);
	        }
	}

    Tensor const&
    L() const 
        { 
        if(!(*this)) Error("LocalOp is default constructed");
        return *L_;
        }

    Tensor const&
    R() const 
        { 
        if(!(*this)) Error("LocalOp is default constructed");
        return *R_;
        }

    explicit operator bool() const { return bool(Ops_[0]); }

    bool
    LIsNull() const;

    bool
    RIsNull() const;


    };


template <class Tensor>
inline LocalOp<Tensor>::
LocalOp()
    :
    Ops_(2,nullptr),
    L_(nullptr),
    R_(nullptr),
    nc_(2),
    size_(-1)
    { 
    }

template <class Tensor>
inline LocalOp<Tensor>::
LocalOp(const Tensor& Op1, const Tensor& Op2,
        const Args& args)
    : 
    Ops_(2,nullptr),
    L_(nullptr),
    R_(nullptr),
    nc_(2),
    size_(-1)
    {
    update(Op1,Op2);
    }

template <class Tensor>
inline LocalOp<Tensor>::
LocalOp(const Tensor& Op1, const Tensor& Op2, const Tensor& Op3,
        const Args& args)
    : 
    Ops_(3,nullptr),
    L_(nullptr),
    R_(nullptr),
    nc_(3),
    size_(-1)
    {
    update(Op1,Op2,Op3);
    }

//template <class Tensor>
//inline LocalOp<Tensor>::
//LocalOp(const MPOt<Tensor>& Ops,
//        const Args& args)
//    : 
//    Ops_(Ops.N(),nullptr),
//    L_(nullptr),
//    R_(nullptr),
//    nc_(Ops.N()),
//    size_(-1)
//    {
//    update(Ops);
//    }

template <class Tensor>
inline LocalOp<Tensor>::
LocalOp(const Tensor& Op1, const Tensor& Op2, 
        const Tensor& L, const Tensor& R,
        const Args& args)
    : 
    Ops_(2,nullptr),
    L_(nullptr),
    R_(nullptr),
    nc_(2),
    size_(-1)
    {
    update(Op1,Op2,L,R);
    }

template <class Tensor>
inline LocalOp<Tensor>::
LocalOp(const Tensor& Op1, const Tensor& Op2, const Tensor& Op3,
        const Tensor& L, const Tensor& R,
        const Args& args)
    : 
    Ops_(3,nullptr),
    L_(nullptr),
    R_(nullptr),
    nc_(3),
    size_(-1)
    {
    update(Op1,Op2,Op3,L,R);
    }

//template <class Tensor>
//inline LocalOp<Tensor>::
//LocalOp(const MPOt<Tensor>& Ops,
//        const Tensor& L, const Tensor& R,
//        const Args& args)
//    : 
//    Ops_(Ops.N(),nullptr),
//    L_(nullptr),
//    R_(nullptr),
//    nc_(Ops.N()),
//    size_(-1)
//    {
//    update(Ops,L,R);
//    }


template <class Tensor>
void inline LocalOp<Tensor>::
update(const Tensor& Op1, const Tensor& Op2)
    {
    if(nc_ != 2) Error("Number of Ops does not match");
    Ops_[0] = &Op1;
    Ops_[1] = &Op2;
    L_ = nullptr;
    R_ = nullptr;
    size_ = -1;
    }

template <class Tensor>
void inline LocalOp<Tensor>::
update(const Tensor& Op1, const Tensor& Op2, const Tensor& Op3)
    {
    if(nc_ != 3) Error("Number of Ops does not match");
    Ops_[0] = &Op1;
    Ops_[1] = &Op2;
    Ops_[2] = &Op3;
    L_ = nullptr;
    R_ = nullptr;
    size_ = -1;
    }

//template <class Tensor>
//void inline LocalOp<Tensor>::
//update(const MPOt<Tensor>& Ops)
//    {
//    if(Ops.N() > 3) Error("Do not support number of Ops > 3");
//    if(nc_ != Ops.N()) Error("Number of Ops does not match");
//    for(int i = 0; i < Ops.N() ; ++i)
//        Ops_[i] = &(Ops.A(i));
//    L_ = nullptr;
//    R_ = nullptr;
//    size_ = -1;
//    }

template <class Tensor>
void inline LocalOp<Tensor>::
update(const Tensor& Op1, const Tensor& Op2, 
       const Tensor& L, const Tensor& R)
    {
    update(Op1,Op2);
    L_ = &L;
    R_ = &R;
    }

template <class Tensor>
void inline LocalOp<Tensor>::
update(const Tensor& Op1, const Tensor& Op2, const Tensor& Op3,
       const Tensor& L, const Tensor& R)
    {
    update(Op1,Op2,Op3);
    L_ = &L;
    R_ = &R;
    }

//template <class Tensor>
//void inline LocalOp<Tensor>::
//update(const MPOt<Tensor>& Ops,
//       const Tensor& L, const Tensor& R)
//    {
//    update(Ops);
//    L_ = &L;
//    R_ = &R;
//    }


template <class Tensor>
bool inline LocalOp<Tensor>::
LIsNull() const
    {
    if(L_ == nullptr) return true;
    return !bool(*L_);
    }

template <class Tensor>
bool inline LocalOp<Tensor>::
RIsNull() const
    {
    if(R_ == nullptr) return true;
    return !bool(*R_);
    }

template <class Tensor>
void inline LocalOp<Tensor>::
product(Tensor const& phi, 
        Tensor      & phip) const
    {
    if(!(*this)) Error("LocalOp is null");

//    auto& Op1 = *Op1_;
//    auto& Op2 = *Op2_;

    if(LIsNull())
        {
        phip = phi;

        if(!RIsNull()) 
            phip *= R(); //m^3 k d
        
	for(auto& Opi : Ops_)
            phip *= *Opi;
//        phip *= Op2; //m^2 k^2
//        phip *= Op1; //m^2 k^2
        }
    else
        {
        phip = phi * L(); //m^3 k d

	for(auto& Opi : Ops_)
	    phip *= *Opi;
//        phip *= Op1; //m^2 k^2
//        phip *= Op2; //m^2 k^2

        if(!RIsNull()) 
            phip *= R();
        }

    phip.mapprime(1,0);
    }

template <class Tensor>
Real inline LocalOp<Tensor>::
expect(const Tensor& phi) const
    {
    Tensor phip;
    product(phi,phip);
    return (dag(phip) * phi).real();
    }

template <class Tensor>
Tensor inline LocalOp<Tensor>::
deltaRho(Tensor const& AA, 
         Tensor const& combine, 
         Direction dir) const
    {
    if(nc_ != 2) Error("Only support Noise term for two-site in the current version!");
    
    auto drho = AA;
    if(dir == Fromleft)
        {
        if(!LIsNull()) drho *= L();
        drho *= (*(Ops_[0]));
        }
    else //dir == Fromright
        {
        if(!RIsNull()) drho *= R();
        drho *= (*(Ops_[1]));
        }
    drho.noprime();
    drho = combine * drho;
    auto ci = commonIndex(combine,drho);
    drho *= dag(prime(drho,ci));

    //Expedient to ensure drho is Hermitian
    drho = drho + dag(swapPrime(drho,0,1));
    drho /= 2.;

    return drho;
    }


template <class Tensor>
Tensor inline LocalOp<Tensor>::
diag() const
    {
    if(!(*this)) Error("LocalOp is null");

//    auto& Op1 = *Op1_;
//    auto& Op2 = *Op2_;

    //lambda helper function:
    auto findIndPair = [](Tensor const& T) {
        for(auto& s : T.inds())
            {
            if(s.primeLevel() == 0 && hasindex(T,prime(s))) 
                {
                return s;
                }
            }
        return IndexT();
        };

    auto toTie = noprime(findtype(*(Ops_[0]),Site));
    auto Diag = (*(Ops_[0])) * delta(toTie,prime(toTie),prime(toTie,2));
    Diag.noprime();

    // Might not the optimal way to contract tensors for number of center sites > 2
    for(int i = 1;i < Ops_.size();++i)
        {
    	toTie = noprime(findtype(*(Ops_[i]),Site));
    	Diag *= noprime((*(Ops_[i])) * delta(toTie,prime(toTie),prime(toTie,2)));
	}

    if(!LIsNull())
        {
        toTie = findIndPair(L());
        if(toTie)
            {
            auto DiagL = L() * delta(toTie,prime(toTie),prime(toTie,2));
            Diag *= noprime(DiagL);
            }
        else
            {
            Diag *= L();
            }
        }

    if(!RIsNull())
        {
        toTie = findIndPair(R());
        if(toTie)
            {
            auto DiagR = R() * delta(toTie,prime(toTie),prime(toTie,2));
            Diag *= noprime(DiagR);
            }
        else
            {
            Diag *= R();
            }
        }

    Diag.dag();
    //Diag must be real since operator assumed Hermitian
    Diag.takeReal();

    return Diag;
    }

template <class Tensor>
long inline LocalOp<Tensor>::
size() const
    {
    if(!(*this)) Error("LocalOp is default constructed");
    if(size_ == -1)
        {
        //Calculate linear size of this 
        //op as a square matrix
        size_ = 1;
        if(!LIsNull()) 
            {
            for(auto& I : L().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= I.m();
                    break;
                    }
                }
            }
        if(!RIsNull()) 
            {
            for(auto& I : R().inds())
                {
                if(I.primeLevel() > 0)
                    {
                    size_ *= I.m();
                    break;
                    }
                }
            }

	for(auto& Opi : Ops_)
	    size_ *= findtype(*Opi,Site).m();
        }

    return size_;
    }

} //namespace itensor

#endif
