#include "itensor/all.h"
#include "SpectraItensor/tMPS.h"
#include <stdio.h>
#include <complex>
#define Pi 3.14159265358979323846

using namespace itensor;

long maxbond(MPS const& psi)
{
  long maxm = 1;
  for(auto i : range1(2,length(psi)))
  {
    auto bondm = dim(leftLinkIndex(psi,i));
    if(bondm > maxm) maxm = bondm;
  }
  return maxm;
}

int main(int argc, char* argv[])
{
    if(argc != 2) 
  	{ 
    	printfln("Usage: %s ArgvInputFile",argv[0]); 
    	return 0; 
  	}

    auto input = InputGroup(argv[1],"input");
    auto N = input.getInt("N"); //number of sites
    auto tau = input.getReal("tau",0.1);
    auto gamm = input.getReal("gamma",0.0);
    auto delt = input.getReal("delta",0);
    delt = delt * Pi / 3.0; // delta = 2*Pi/3
    auto V0 = input.getReal("V0",1.5);
    auto beta = input.getReal("beta",3.0);
    auto halfU = input.getReal("halfU",0.5);
    auto dB = input.getInt("dB",4); //number of photon/site before cutoff
    auto nsweeps = input.getInt("nsweeps"); //number of sweeps
    auto table = InputGroup(input,"sweeps");
    auto quiet = input.getYesNo("quiet",true);
    auto qn = input.getYesNo("QN",true);

    //--------------------------------------//get input parameters
    auto sweeps = Sweeps(nsweeps,table);
    // println(sweeps);
    
    auto sites = Boson(N,{"ConserveQNs=",qn,"MaxOcc=",dB});
    auto state = InitState(sites);
    for(auto j : range1(N))
      {
        state.set(j,(j==N/2+N%1 ? "3" : "Emp"));
      }
    MPS psi = MPS(state);
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

    //Apply MPO to MPS
        Args::global().add("Quiet",quiet);
        std::vector<Real> val{};
        auto N1 = sites.op("N",N/2+N%1);
        cpu_time sw_time;
        printfln("Start applying expH to psi by TDVP:\n");
        
        // two-site TDVP, efficient method.
        int sw = 1;
        LocalMPO PH(H);
        for(; sw <= nsweeps; ++sw)
        {
          auto finaltime = tMPSWorker(psi,PH,sweeps,tau*Cplx_i,sw);
          auto phi = psi;
          phi.position(N/2+N%1);
          auto expval = (phi(N/2+N%1)*N1*dag(prime(phi(N/2+N%1),"Site"))).eltC().real();
          val.push_back(expval);
          printfln("    2Calculated t=%.4f of total T=%.4f, ev = %.6f\n",sw*tau,nsweeps*tau,expval);
        }

        auto sm = sw_time.sincemark();
        printfln("    Sweep finished CPU time = %s (Wall time = %s)",showtime(sm.time),showtime(sm.wall));

// save result: density dist
        // char filename[64];
        // printfln("output: benchmark-%.1f-%.1f.dat",gamm,halfU*2);
        // sprintf(filename, "benchmark-%.1f-%.1f.dat",gamm,halfU*2);
        // std::ofstream file(filename, std::ofstream::binary);
        // file.write((char *)&sxval[0], sizeof(double)*sxval.size());
printfln("--------------<n_c>----------------");
for(auto& n:val) {printf("%.12f ",n);}
printfln("\n-------------ending----------------");
return 0;
}
