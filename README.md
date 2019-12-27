Homepage: http://itensor.org/

An efficient and flexible C++ library for performing tensor network calculations.

The foundation of the library is the Intelligent Tensor or ITensor.
Contracting ITensors is no harder than multiplying scalars: matching indices
automatically find each other and contract. This makes it easy to transcribe
tensor network diagrams into correct, efficient code.

Installation instructions can be found in the [INSTALL](INSTALL.md) file.

Benchmark expamles of the non-Hermitian DMRG and TDVP methods can be found under DEMO folder.

Under Demo/DMRG, run `make && ./ndmrg argv_dmrg.txt`

Under Demo/TDVP, run `make && ./ntdvp argv_tdvp.txt`
