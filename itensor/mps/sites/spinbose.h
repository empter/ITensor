#ifndef __ITENSOR_SPINBOSEMIX__H
#define __ITENSOR_SPINBOSEMIX__H
#include "itensor/mps/siteset.h"
#include <cmath>

namespace itensor
{
    class SpinBoseMix;
    using SpinBose=BasicSiteSet<SpinBoseMix>;

// odd sites SpinHalf operator and even sites Bose operators

class SpinBoseMix
{
    IQIndex s;

public:
    SpinBoseMix() {}
    SpinBoseMix(IQIndex I): s(I) {}
    SpinBoseMix(int n, Args const& args=Args::global())
    {
        if(n%2==1) // SpinHalf
        {
          s = IQIndex{nameint("S=1/2 ",n),
                 Index(nameint("Up ",n),1,Site),QN("Sz=",+1),
                 Index(nameint("Dn ",n),1,Site),QN("Sz=",-1)};
        }
        else // boson (m boson/site at most)
        {
          int m = args.getInt("BosonCut",3);
          auto sind = stdx::reserve_vector<IndexQN>(m+1);
          for(int i = 0; i <= m; ++i)
          {
            sind.emplace_back(Index(nameint(format("B%d_",i),n),1,Site),QN("Nb=",+i));
          }
          s = IQIndex(nameint("site=",n),std::move(sind));
        }
    }

    IQIndex index() const {return s;}

    IQIndexVal state(std::string const& state)
    {
        if (state == "Up" || state=="Em")
        {
            return s(1);
        }
        else if (state=="Dn" || state=="B1")
        {
            return s(2);
        }
        for(int j = 2; j <= 100; ++j)
        {
          if(state == format("B%d",j)) return s(j+1);
        }

        Error("State " + state + " not recognized");
        return IQIndexVal{};
    }

    IQTensor op(std::string const& opname, Args const& args) const
    {
        auto sP=prime(s);
        auto m=s.nblock();
        std::vector<IQIndexVal> ind{};
        std::vector<IQIndexVal> indP{};

        for(int i = 1; i <= m; ++i)
        {
          ind.push_back(s(i));
          indP.push_back(sP(i));
        }

        IQTensor Op(dag(s),sP); // operator with 1 arrow in and 1 arrow out

        // SpinHalf single-site operator
        if(opname == "Sz")
            {
            Op.set(ind[0],indP[0],+0.5);
            Op.set(ind[1],indP[1],-0.5);
            }
        else
        if(opname == "Sx")
            {
            //mixedIQTensor call needed here
            //because as an IQTensor, Op would
            //not have a well defined QN flux
            Op = mixedIQTensor(dag(s),sP);
            Op.set(ind[0],indP[1],+0.5);
            Op.set(ind[1],indP[0],+0.5);
            }
        else
        if(opname == "ISy")
            {
            //mixedIQTensor call needed here
            //because as an IQTensor, Op would
            //not have a well defined QN flux
            Op = mixedIQTensor(dag(s),sP);
            Op.set(ind[0],indP[1],-0.5);
            Op.set(ind[1],indP[0],+0.5);
            }
        else
        if(opname == "Sy")
            {
            //mixedIQTensor call needed here
            //because as an IQTensor, Op would
            //not have a well defined QN flux
            Op = mixedIQTensor(dag(s),sP);
            Op.set(ind[0],indP[1],+0.5*Cplx_i);
            Op.set(ind[1],indP[0],-0.5*Cplx_i);
            }
        else
        if(opname == "Sp" || opname == "S+")
            {
            Op.set(ind[1],indP[0],1);
            }
        else
        if(opname == "Sm" || opname == "S-")
            {
            Op.set(ind[0],indP[1],1);
            }
        else
        if(opname == "projUp")
            {
            Op.set(ind[0],indP[0],1);
            }
        else
        if(opname == "projDn")
            {
            Op.set(ind[1],indP[1],1);
            }
        else
        if(opname == "S2")
            {
            Op.set(ind[0],indP[0],0.75);
            Op.set(ind[1],indP[1],0.75);
            }
        // Boson single-site operator
        else if (opname=="Nb")
        {
          for(int i = 0; i < m; ++i)
          {
            Op.set(ind[i],indP[i],i);
          }
        }
        else if (opname=="Adagb")
        {
          for(int i = 1; i < m; ++i)
          {
            Op.set(ind[i-1],indP[i],std::sqrt(i));
          }
        }
        else if (opname=="Ab")
        {
          for(int i = 1; i < m; ++i)
          {
            Op.set(ind[i],indP[i-1],std::sqrt(i));
          }
        }
        else
        {
          Error("Operator " + opname + " name not recognized !");
        }

        return Op;
    }

};
}
#endif
