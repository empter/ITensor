// Copyright (C) 2016-2019 Yixuan Qiu <yixuan.qiu@cos.name>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

#ifndef CPLX_GEN_EIGS_SOLVER_H
#define CPLX_GEN_EIGS_SOLVER_H

#include <Eigen/Core>
#include "../itensor/all_mps.h"

#include "CplxGenEigsBase.h"
#include "Util/SelectionRule.h"

namespace Spectra {


template < typename Scalar,
           int SelectionRule,
           typename OpType>
class CplxGenEigsSolver: public CplxGenEigsBase<Scalar, SelectionRule, OpType>
{
private:
    typedef Eigen::Index Index;

public:
    CplxGenEigsSolver(OpType const* op, Index nev, Index ncv) :
        CplxGenEigsBase<Scalar, SelectionRule, OpType>(op, nev, ncv)
    {}
};


} // namespace Spectra

#endif // CPLX_GEN_EIGS_SOLVER_H
