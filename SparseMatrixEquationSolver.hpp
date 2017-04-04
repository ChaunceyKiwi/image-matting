//
//  SparseMatrixEquationSolver.hpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#ifndef SparseMatrixEquationSolver_hpp
#define SparseMatrixEquationSolver_hpp

#include <stdio.h>
#include "umfpack.h"

class SparseMatrixEquationSolver {
private:
  // Input
  int *Ap;
  int *Ai;
  double* Ax;
  double *b;
  int n;
public:
  SparseMatrixEquationSolver(int *Ap, int *Ai, double* Ax, double *b, int n);
  double* solveEquation();
};

#endif /* SparseMatrixEquationSolver_hpp */
