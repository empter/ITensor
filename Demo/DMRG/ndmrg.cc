#include "SpectraItensor/cplxdmrg.h"
#include "itensor/all.h"
#include <stdio.h>
#define Pi 3.14159265358979323846

using namespace itensor;

int
main(int argc, char* argv[] )
    {
      if(argc != 2)
        {
          printfln("Usage: %s ArgvInputFile",argv[0]);
          return 0;
        }
        auto input = InputGroup(argv[1],"input");
        auto N = input.getInt("N"); //number of sites
        auto delt = input.getReal("delta",0);
        delt = delt * Pi / 3.0; // delta = 2*Pi/3
        auto gamm = input.getReal("gamma",0.0);
        auto V0 = input.getReal("V0",1.5);
        auto beta = input.getReal("beta",3.0);
        auto halfU = input.getReal("halfU",0.5);
        auto dB = input.getInt("dB",4); //number of photon/site before cutoff
        auto nsweeps = input.getInt("nsweeps"); //number of sweeps
        auto table = InputGroup(input,"sweeps");
        auto quiet = input.getYesNo("quiet",true);
        auto iter = input.getInt("Iter",2);
        auto errg = input.getReal("ErrG",1E-6);
        auto errg2 = input.getReal("EngErrG",1E-6);
        auto qn = input.getYesNo("QN",true);

    //--------------------------------------//get input parameters
        auto sweeps = Sweeps(1,table);

        auto sites = Boson(N,{"ConserveQNs=",qn,"MaxOcc=",dB});
        auto state = InitState(sites);
        for(auto j : range1(N))
            {
              state.set(j,(j%3==2 ? "Emp" : "1"));
            }
        auto psi = MPS(state);
        Print(totalQN(psi));

        auto ampo = AutoMPO(sites);
        for(int j = 1; j < N; ++j)
            {
              ampo += -1,"Adag",j+1,"A",j;
              ampo += -(1-gamm),"Adag",j,"A",j+1;
            }
        for(int j = 1; j <= N; ++j)
            {
              ampo += V0*std::cos(2*Pi*j/beta+delt)-halfU,"N",j;
              ampo += halfU,"N",j,"N",j;
            }
        auto H = toMPO(ampo);

        std::vector<Cplx> eng{};
        for(auto lp : range(nsweeps))
        {
          auto energy = dmrg(psi,H,sweeps,{"Quiet",quiet,"MaxIter",iter,"ErrGoal",errg,"EnergyErrgoal",1E-12});
          eng.push_back(energy);
        }

        printfln("--------------Re(E2-E1)----------------");
        for(auto i : range(nsweeps-1)) {printf("%.12f ",abs(eng[i].real()-eng[i+1].real()));}
        printfln("\n--------------Im(E2-E1)----------------");
        for(auto i : range(nsweeps-1)) {printf("%.12f ",abs(eng[i].imag()-eng[i+1].imag()));}
        printfln("\n-------------ending----------------");

        return 0;
    }
