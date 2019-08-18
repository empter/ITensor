#ifndef __ITENSOR_DMRGS_H
#define __ITENSOR_DMRGS_H

#include "iterativesolvers.h"
#include "itensor/mps/localmposet.h"
#include "itensor/mps/localmpo_mps.h"
#include "itensor/mps/sweeps.h"
#include "itensor/mps/DMRGObserver.h"
#include "itensor/util/cputime.h"


namespace itensor {

template<class LocalOpT>
Cplx
DMRGWorker3s(MPS & psi,
           LocalOpT & PH,
           Sweeps const& sweeps,
           Args const& args = Args::global());

template<class LocalOpT>
Cplx
DMRGWorker3s(MPS & psi,
           LocalOpT & PH,
           Sweeps const& sweeps,
           DMRGObserver & obs,
           Args args = Args::global());

//
// Available DMRG methods:
//

//
//DMRG with an MPO
//
Cplx inline
dmrg3s(MPS & psi, 
     MPO const& H, 
     Sweeps const& sweeps,
     Args const& args = Args::global())
    {
    LocalMPO PH(H,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,args);
    return energy;
    }

//
//DMRG with an MPO
//Version that takes a starting guess MPS
//and returns the optimized MPS
//
std::tuple<Cplx,MPS> inline
dmrg3s(MPO const& H,
     MPS const& psi0,
     Sweeps const& sweeps,
     Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg3s(psi,H,sweeps,args);
    return std::tuple<Cplx,MPS>(energy,psi);
    }

//
//DMRG with an MPO and custom DMRGObserver
//
Cplx inline
dmrg3s(MPS& psi, 
     MPO const& H, 
     Sweeps const& sweeps, 
     DMRGObserver & obs,
     Args const& args = Args::global())
    {
    LocalMPO PH(H,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,obs,args);
    return energy;
    }

//
//DMRG with an MPO and custom DMRGObserver
//Version that takes a starting guess MPS
//and returns the optimized MPS
//
std::tuple<Cplx,MPS> inline
dmrg3s(MPO const& H,
     MPS const& psi0,
     Sweeps const& sweeps,
     DMRGObserver & obs,
     Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg3s(psi,H,sweeps,obs,args);
    return std::tuple<Cplx,MPS>(energy,psi);
    }

//
//DMRG with an MPO and boundary tensors LH, RH
// LH - H1 - H2 - ... - HN - RH
//(ok if one or both of LH, RH default constructed)
//
Cplx inline
dmrg3s(MPS& psi, 
     MPO const& H, 
     ITensor const& LH, 
     ITensor const& RH,
     Sweeps const& sweeps,
     Args const& args = Args::global())
    {
    LocalMPO PH(H,LH,RH,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,args);
    return energy;
    }

//
//DMRG with an MPO and boundary tensors LH, RH
// LH - H1 - H2 - ... - HN - RH
//(ok if one or both of LH, RH default constructed)
//Version that takes a starting guess MPS
//and returns the optimized MPS
//
std::tuple<Cplx,MPS> inline
dmrg3s(MPO const& H,
     ITensor const& LH,
     ITensor const& RH,
     MPS const& psi0,
     Sweeps const& sweeps,
     Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg3s(psi,H,LH,RH,sweeps,args);
    return std::tuple<Cplx,MPS>(energy,psi);
    }

//
//DMRG with an MPO and boundary tensors LH, RH
//and a custom observer
//
Cplx inline
dmrg3s(MPS& psi, 
     MPO const& H, 
     ITensor const& LH, 
     ITensor const& RH,
     Sweeps const& sweeps, 
     DMRGObserver& obs,
     Args const& args = Args::global())
    {
    LocalMPO PH(H,LH,RH,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,obs,args);
    return energy;
    }

std::tuple<Cplx,MPS> inline
dmrg3s(MPO const& H,
     ITensor const& LH,
     ITensor const& RH,
     MPS const& psi0,
     Sweeps const& sweeps,
     DMRGObserver& obs,
     Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg3s(psi,H,LH,RH,sweeps,obs,args);
    return std::tuple<Cplx,MPS>(energy,psi);
    }

// //
// //DMRG with a set of MPOs (lazily summed)
// //(H vector is 0-indexed)
// //
// Cplx inline
// dmrg3s(MPS& psi, 
//      std::vector<MPO> const& Hset, 
//      Sweeps const& sweeps,
//      Args const& args = Args::global())
//     {
//     LocalMPOSet PH(Hset,{args,"NumCenter",1});
//     Cplx energy = DMRGWorker3s(psi,PH,sweeps,args);
//     return energy;
//     }
// 
// std::tuple<Cplx,MPS> inline
// dmrg3s(std::vector<MPO> const& Hset,
//      MPS const& psi0,
//      Sweeps const& sweeps,
//      Args const& args = Args::global())
//     {
//     auto psi = psi0;
//     auto energy = dmrg3s(psi,Hset,sweeps,args);
//     return std::tuple<Cplx,MPS>(energy,psi);
//     }
// 
// //
// //DMRG with a set of MPOs and a custom DMRGObserver
// //(H vector is 0-indexed)
// //
// Cplx inline
// dmrg3s(MPS& psi, 
//      std::vector<MPO> const& Hset, 
//      Sweeps const& sweeps, 
//      DMRGObserver& obs,
//      Args const& args = Args::global())
//     {
//     LocalMPOSet PH(Hset,{args,"NumCenter",1});
//     Cplx energy = DMRGWorker3s(psi,PH,sweeps,obs,args);
//     return energy;
//     }
// 
// std::tuple<Cplx,MPS> inline
// dmrg3s(std::vector<MPO> const& Hset,
//      MPS const& psi0,
//      Sweeps const& sweeps,
//      DMRGObserver& obs,
//      Args const& args = Args::global())
//     {
//     auto psi = psi0;
//     auto energy = dmrg3s(psi,Hset,sweeps,obs,args);
//     return std::tuple<Cplx,MPS>(energy,psi);
//     }

//
//DMRG with a single Hamiltonian MPO and a set of 
//MPS to orthogonalize against
//(psis vector is 0-indexed)
//Named Args recognized:
// Weight - real number w > 0; calling dmrg(psi,H,psis,sweeps,Args("Weight",w))
//          sets the effective Hamiltonian to be
//          H + w * (|0><0| + |1><1| + ...) where |0> = psis[0], |1> = psis[1]
//          etc.
//
Cplx inline
dmrg3s(MPS& psi, 
     MPO const& H, 
     std::vector<MPS> const& psis, 
     Sweeps const& sweeps, 
     Args const& args = Args::global())
    {
    LocalMPO_MPS PH(H,psis,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,args);
    return energy;
    }

std::tuple<Cplx,MPS> inline
dmrg3s(MPO const& H,
     std::vector<MPS> const& psis,
     MPS const& psi0,
     Sweeps const& sweeps,
     Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg3s(psi,H,psis,sweeps,args);
    return std::tuple<Cplx,MPS>(energy,psi);
    }

//
//DMRG with a single Hamiltonian MPO, 
//a set of MPS to orthogonalize against, 
//and a custom DMRGObserver.
//(psis vector is 0-indexed)
//Named Args recognized:
// Weight - real number w > 0; calling dmrg(psi,H,psis,sweeps,Args("Weight",w))
//          sets the effective Hamiltonian to be
//          H + w * (|0><0| + |1><1| + ...) where |0> = psis[0], |1> = psis[1]
//          etc.
//
Cplx inline
dmrg3s(MPS & psi, 
     MPO const& H, 
     std::vector<MPS> const& psis, 
     Sweeps const& sweeps, 
     DMRGObserver& obs, 
     Args const& args = Args::global())
    {
    LocalMPO_MPS PH(H,psis,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,obs,args);
    return energy;
    }

std::tuple<Cplx,MPS> inline
dmrg3s(MPO const& H,
     std::vector<MPS> const& psis,
     MPS const& psi0,
     Sweeps const& sweeps,
     DMRGObserver& obs, 
     Args const& args = Args::global())
    {
    auto psi = psi0;
    auto energy = dmrg3s(psi,H,psis,sweeps,obs,args);
    return std::tuple<Cplx,MPS>(energy,psi);
    }


//
// DMRGWorker
//

template<class LocalOpT>
Cplx
DMRGWorker3s(MPS & psi,
           LocalOpT & PH,
           Sweeps const& sweeps,
           Args const& args)
    {
    DMRGObserver obs(psi,{args,"NumCenter",1});
    Cplx energy = DMRGWorker3s(psi,PH,sweeps,obs,args);
    return energy;
    }

template<class LocalOpT>
Cplx
DMRGWorker3s(MPS& psi,
           LocalOpT& PH,
           const Sweeps& sweeps,
           DMRGObserver& obs,
           Args args)
    {
    if( args.defined("WriteM") )
      {
      if( args.defined("WriteDim") )
        {
        Global::warnDeprecated("Args WirteM and WriteDim are both defined. WriteM is deprecated in favor of WriteDim, WriteDim will be used.");
        }
      else
        {
        Global::warnDeprecated("Arg WriteM is deprecated in favor of WriteDim.");
        args.add("WriteDim",args.getInt("WriteM"));
        }
      }
  
    // Truncate blocks of degenerate singular values (or not)
    args.add("RespectDegenerate",args.getBool("RespectDegenerate",true));

    const bool silent = args.getBool("Silent",false);
    if(silent)
        {
        args.add("Quiet",true);
        args.add("PrintEigs",false);
        args.add("NoMeasure",true);
        args.add("DebugLevel",0);
        }
    const bool quiet = args.getBool("Quiet",false);
    const int debug_level = args.getInt("DebugLevel",(quiet ? 0 : 1));

    const int N = length(psi);
    Cplx energy = NAN;

    psi.position(1);
    Spectrum spec;

    args.add("DebugLevel",debug_level);
    args.add("DoNormalize",true);

    for(int sw = 1; sw <= sweeps.nsweep(); ++sw)
        {
        cpu_time sw_time;
        args.add("Sweep",sw);
        args.add("NSweep",sweeps.nsweep());
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("MinDim",sweeps.mindim(sw));
        args.add("MaxDim",sweeps.maxdim(sw));
        args.add("Noise",sweeps.noise(sw));
        args.add("MaxKrylov",sweeps.niter(sw));

        if(!PH.doWrite()
           && args.defined("WriteDim")
           && sweeps.maxdim(sw) >= args.getInt("WriteDim"))
            {
            if(!quiet)
                {
                println("\nTurning on write to disk, write_dir = ",
                        args.getString("WriteDir","./"));
                }

            //psi.doWrite(true);
            PH.doWrite(true,args);
            }

        for(int b = 1, ha = 1; ha <= 2; sweepnext1(b,ha,N))
            {
            if(!quiet)
                {
                printfln("Sweep=%d, HS=%d, Bond=%d/%d",sw,ha,b,N);
                }

            PH.position(b,psi);

            auto phi = psi(b);

            args.add("DMRGb",b);
            args.add("DMRGh",ha);

            arnoldiR(PH,phi,energy,args);

            if(ha == 1 && b != N)
            {
              auto rb = rightLinkIndex(psi,b);
              ITensor Ue, De, Ve(rb);
              spec = svd(phi,Ue,De,Ve,args);
              psi.ref(b) = Ue;
              psi.ref(b+1) *= (De * Ve);
            }
            else if(ha == 2 && b != 1)
            {
              auto la = leftLinkIndex(psi,b);
              ITensor Ue(la), De, Ve;
              spec = svd(phi,Ue,De,Ve,args);
              psi.ref(b) = Ve;
              psi.ref(b-1) *= (Ue * De);
            }
            else
            {
              psi.ref(b) = phi;
            }

            obs.lastSpectrum(spec);

            args.add("AtBond",b);
            args.add("HalfSweep",ha);
            args.add("Energy",energy.real()); 
            args.add("Truncerr",spec.truncerr()); 

            obs.measure(args);

            } //for loop over b

        if(!silent)
            {
            auto sm = sw_time.sincemark();
            printfln("    Sweep %d/%d CPU time = %s (Wall time = %s)",
                          sw,sweeps.nsweep(),showtime(sm.time),showtime(sm.wall));
            }

        if(obs.checkDone(args)) break;

        } //for loop over sw

    psi.rightLim(2);// restore orthoCenter
    psi.normalize();

    return energy;
    }

} //namespace itensor


#endif
