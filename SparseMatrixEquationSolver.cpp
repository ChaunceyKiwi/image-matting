//
//  SparseMatrixEquationSolver.cpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#include "SparseMatrixEquationSolver.hpp"

SparseMatrixEquationSolver::SparseMatrixEquationSolver(int *Ap, int *Ai, double* Ax, double *b, int n) {
  this->Ap = Ap;
  this->Ai = Ai;
  this->Ax = Ax;
  this->b = b;
  this->n = n;
}

double* SparseMatrixEquationSolver::solveEquation() {
  double* alphaArray = new double [n];
  void *Symbolic, *Numeric;
  
  /* symbolic analysis */
  umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, NULL, NULL);
  
  /* LU factorization */
  umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, NULL, NULL);
  umfpack_di_free_symbolic(&Symbolic);
  
  /* solve system */
  umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, alphaArray, b, Numeric, NULL, NULL);
  umfpack_di_free_numeric(&Numeric);
  
  return alphaArray;
}
