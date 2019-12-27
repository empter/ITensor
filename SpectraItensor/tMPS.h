#ifndef __ITENSOR_TMPS_H
#define __ITENSOR_TMPS_H

#include "itensor/mps/localmpo.h"
#include "itensor/mps/sweeps.h"
#include "itensor/util/cputime.h"
#include "expApplyH.h"

// used LocalMPO class methods: product, productnext, localh, localhnext, size

namespace itensor {

template<class LocalOpT>
Real
tMPSWorker(MPS & psi,
           LocalOpT & PH,
           Sweeps const& sweeps,
           Cplx tau,
           int sw,
           Args args = Args::global());

template<class LocalOpT>
Real
tMPSWorkers(MPS & psi,
           LocalOpT & PH,
           Sweeps const& sweeps,
           Cplx tau,
           int sw,
           Args args = Args::global());

//
// Available tMPS method (PRB 94, 165116):
//

// 
// tMPS with an MPO, single-site algorithm.
// 
Real inline
tMPSs(MPS & psi,
      MPO const& H,
      Sweeps const& sweeps,
      Cplx tau,
      int sw,
      Args const& args = Args::global())
    {
    LocalMPO PH(H,{args,"NumCenter",1});
    Real finaltime = tMPSWorkers(psi,PH,sweeps,tau,sw,args);
    return finaltime;
    }

//
//tMPS with an MPO, two-site algorithm.
//
Real inline
tMPS(MPS & psi,
      MPO const& H,
      Sweeps const& sweeps,
      Cplx tau,
      int sw,
      Args const& args = Args::global())
    {
    // LocalMPO PH(H,args);
    // Real finaltime = tMPSWorker(psi,PH,sweeps,tau,sw,args);
    // return finaltime;
    auto bondm = dim(linkIndex(psi,length(psi)/2));
    if(bondm < sweeps.maxdim(sw))
    {
      LocalMPO PH(H,args);
      Real finaltime = tMPSWorker(psi,PH,sweeps,tau,sw,args);
      return finaltime;
    }
    else
    {
      Real finaltime = tMPSs(psi,H,sweeps,tau,sw,args);
      return finaltime;
    }
    }

//
// tMPSWorker
//

template <class LocalOpT>
Real
tMPSWorker(MPS & psi,
           LocalOpT & PH,
           Sweeps const& sweeps,
           Cplx tau,
           int sw,
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
        const bool silent = args.getBool("Silent",false);
        if(silent)
            {
            args.add("Quiet",true);
            args.add("NoMeasure",true);
            args.add("DebugLevel",0);
            }
        const bool quiet = args.getBool("Quiet",false);
        const int debug_level = args.getInt("DebugLevel",(quiet ? 0 : 1));
        args.add("DebugLevel",debug_level);
        args.add("DoNormalize",true);

        const int N = length(psi);
        Real finaltime = NAN;
        Spectrum spec;
        cpu_time sw_time;
        cpu_time exp_time;

        args.add("Sweep",sw);
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("Mindim",sweeps.mindim(sw));
        args.add("Maxdim",sweeps.maxdim(sw));
        args.add("ErrGoal",sweeps.noise(sw));
        args.add("MaxKrylov",sweeps.niter(sw));

        psi.position(1);
        PH.position(1,psi);

        for(int b = 1, ha = 1; ha <= 2; sweepnext(b,ha,N))
            {
            if(!quiet)
                {
                printfln("Sweep=%d, HS=%d, Bond=(%d,%d)",sw,ha,b,(b+1));
                }

            auto phi = psi(b)*psi(b+1);

            //different from DMRG
            //for loop from b = 1 to N-1
            if(ha == 1)
            {
              expApplyH(PH,phi,-tau/2,NoDir,args);//evol forward tau/2 @ site (b:b+1)
              spec = psi.svdBond(b,phi,Fromleft,args);
              if(b < N-1)
              {
                PH.position(b+1,psi);
                phi = psi(b+1);
                expApplyH(PH,phi,tau/2,Fromleft,args);//evol backward tau/2 @ site (b+1) for next evol @ (b+1:b+2)
                psi.ref(b+1) = phi;
              }
            }
            //for loop from b = N-1 to 1
            else
            {
              if(b == N/2) exp_time.mark();
              expApplyH(PH,phi,-tau/2,NoDir,args);
              // int dm=1;for(auto& I : phi.inds()) {dm *= dim(I);} Print(dm);
              if(b == N/2)
              {
                // Print(args);
                auto expt = exp_time.sincemark();
                printfln("    expH@N/2 Sweep %d CPU time = %s PH.size = %d",
                          sw,showtime(expt.time),PH.size());
              }
              spec = psi.svdBond(b,phi,Fromright,args);
              if(b > 1)
              {
                PH.position(b-1,psi);
                phi = psi(b);
                expApplyH(PH,phi,tau/2,Fromright,args);
                psi.ref(b) = phi;
              }
            }


            if(!quiet)
                { 
                printfln("    Truncated to Cutoff=%.1E, Min_dim=%d, Max_dim=%d",
                          sweeps.cutoff(sw),
                          sweeps.mindim(sw), 
                          sweeps.maxdim(sw) );
                printfln("    Trunc. err=%.1E, States kept: %s",
                         spec.truncerr(),
                         showDim(linkIndex(psi,b)) );
                }

            finaltime = sw*imagRef(tau);
            args.add("AtBond",b);
            args.add("HalfSweep",ha);
            //args.add("LastTime",finaltime);

            } //for loop over b

        if(args.getBool("KrylovNoCover",false))
          printfln("    no covergence in Arnoldi_expm at Sweep %d",sw);
        auto sm = sw_time.sincemark();
        printfln("    Sweep %d CPU time = %s (Wall time = %s)",
                  sw,showtime(sm.time),showtime(sm.wall));

    psi.normalize();

    return finaltime;
    }

//single site algorithm
template <class LocalOpT>
Real
tMPSWorkers(MPS & psi,
             LocalOpT & PH,
             Sweeps const& sweeps,
             Cplx tau,
             int sw,
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
        const bool silent = args.getBool("Silent",false);
        if(silent)
            {
            args.add("Quiet",true);
            args.add("NoMeasure",true);
            args.add("DebugLevel",0);
            }
        const bool quiet = args.getBool("Quiet",false);
        const int debug_level = args.getInt("DebugLevel",(quiet ? 0 : 1));
        args.add("DebugLevel",debug_level);
        // args.add("UseOrigDim",true);

        const int N = length(psi);
        Real finaltime = NAN;
        Spectrum spec;
        cpu_time sw_time;
        cpu_time exp_time;

        args.add("Sweep",sw);
        args.add("Cutoff",sweeps.cutoff(sw));
        args.add("Mindim",sweeps.mindim(sw));
        args.add("Maxdim",sweeps.maxdim(sw));
        args.add("ErrGoal",sweeps.noise(sw));
        args.add("MaxKrylov",sweeps.niter(sw));

        psi.position(1);
        PH.position(1,psi);

        for(int b = 1, ha = 1; ha <= 2; sweepnext1(b,ha,N))
            {
            if(!quiet)
                {
                printfln("Sweep=%d, HS=%d, Bond=(%d,%d)",sw,ha,b,(b+1));
                }

            auto phi = psi(b);
            //different from DMRG
            //for loop from b = 1 to N-1
            if(ha == 1)
            {
              expApplyH(PH,phi,-tau/2,NoDir,args);//evol forward tau/2 @ site (b)
              if(b < N)
              {
                ITensor U,D,V;
                auto indl= leftLinkIndex(psi,b);
                if(indl)
                {
                  U = ITensor(indl,findIndex(phi,"Site"));
                }
                else
                {
                  U = ITensor(findIndex(phi,"Site"));
                }
                spec = svd(phi,U,D,V,args);
                psi.ref(b) = U;
                D *= 1./norm(D);
                phi = D*V;
                PH.position(b+1,psi);
                expApplyH(PH,phi,tau/2,Fromleft,args);//evol backward tau/2 for next evol @ (b+1)
                psi.ref(b+1) *= phi;
              }
              else
              {
                psi.ref(b) = phi;
              }
            }
            //for loop from b = N to 1
            else
            {
              if(b == N/2) exp_time.mark();
              expApplyH(PH,phi,-tau/2,NoDir,args);
              if(b == N/2)
              {
                auto expt = exp_time.sincemark();
                printfln("    expH@N/2 Sweep %d CPU time = %s PH.size = %d",
                          sw,showtime(expt.time),PH.size());
              }
              if(b >1)
              {
                ITensor U,D,V;
                auto indr = rightLinkIndex(psi,b);
                if(indr)
                {
                  V = ITensor(findIndex(phi,"Site"),indr);
                }
                else
                {
                  V = ITensor(findIndex(phi,"Site"));
                }
                spec = svd(phi,U,D,V,args);
                psi.ref(b) = V;
                D *= 1./norm(D);
                phi = U*D;
                PH.position(b-1,psi);
                expApplyH(PH,phi,tau/2,Fromright,args);
                psi.ref(b-1) *= phi;
              }
              else
              {
                psi.ref(b) = phi;
              }
            }


            if(!quiet)
                {
                printfln("    Truncated to Cutoff=%.1E, Min_m=%d, Max_m=%d",
                          sweeps.cutoff(sw),
                          sweeps.mindim(sw),
                          sweeps.maxdim(sw) );
                printfln("    Trunc. err=%.1E, States kept: %s",
                         spec.truncerr(),
                         showDim(linkIndex(psi,b)) );
                }

            finaltime = sw*imagRef(tau);
            args.add("AtBond",b);
            args.add("HalfSweep",ha);

            } //for loop over b

        if(args.getBool("KrylovNoCover",false))
          printfln("    no covergence in Arnoldi_expm at Sweep %d",sw);
        auto sm = sw_time.sincemark();
        printfln("    Sweep %d CPU time = %s (Wall time = %s)",
                  sw,showtime(sm.time),showtime(sm.wall));

    psi.rightLim(2);// restore orthoCenter
    psi.normalize();

    return finaltime;
    }

} //namespace itensor


#endif
