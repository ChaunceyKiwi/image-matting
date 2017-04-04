//
//  AlphaCalculator.cpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#include "AlphaCalculator.hpp"

AlphaCalculator::AlphaCalculator(int lambda, int win_size, double epsilon, cv::Mat input, cv::Mat consts_map, cv::Mat consts_vals) {
  this->lambda = lambda;
  this->win_size = win_size;
  this->epsilon = epsilon;
  this->input = input;
  this->consts_map = consts_map;
  this->consts_vals = consts_vals;
}

cv::Mat AlphaCalculator::getAlpha() {
  int height = this->input.size().height;
  int width = this->input.size().width;
  int img_size = height * width;
  
  // Solve the equation x = (A + lambda * D) \ (lambda * consts_vals(:));
  // To make it clear, let left * x = right
  // left = A + lambda * D and right = lambda * consts_vals(:)
  
  // Calculation of left side(A + lambda * D)
  LaplacianCalculator laplacianCalculator(win_size, epsilon, input, consts_map);
  SpMat A = laplacianCalculator.getLaplacianMatrix();
  
  cv::Mat consts_map_trans = consts_map.t();
  SpMat D(img_size, img_size);
  for (int i = 0; i < img_size; i++) {
    D.coeffRef(i,i) = (int)consts_map_trans.at<char>(0, i);
  }
  SpMat left = A + lambda * D;
  
  // Calculation of right side((lambda * consts_vals(:))
  cv::Mat consts_vals_in_a_col;
  cv::Mat transpo = consts_vals.t();
  consts_vals_in_a_col = transpo.reshape(1, img_size);
  Eigen::VectorXd right(img_size);
  for (int i = 0;i < img_size;i++) {
    right(i) = lambda * consts_vals_in_a_col.at<char>(i,0);
  }
  
  int* innerPointer = left.innerIndexPtr();
  int* outerPointer = left.outerIndexPtr();
  double* valuePointer = left.valuePtr();
  double* rightPointer = right.data();
  
  int size = input.size().width * input.size().height;
  
  SparseMatrixEquationSolver sparseMatrixEquationSolver(outerPointer, innerPointer, valuePointer, rightPointer, size);
  double* alphaArray = sparseMatrixEquationSolver.solveEquation();
  
  cv::Mat alpha = cv::Mat::ones(input.size().height, input.size().width, CV_64F);
  
  int count = 0;
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      alpha.at<double>(j, i) = alphaArray[count++];
      if(alpha.at<double>(j, i) > 1)
        alpha.at<double>(j, i) = 1;
      else if (alpha.at<double>(j, i) < 0)
        alpha.at<double>(j, i) = 0;
    }
  }
  
  return alpha;
}
