//
//  AlphaCalculator.hpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#ifndef AlphaCalculator_hpp
#define AlphaCalculator_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include "LaplacianCalculator.hpp"
#include "SparseMatrixEquationSolver.hpp"

class AlphaCalculator {
private:
  // Input
  int lambda;
  int win_size;
  double epsilon;
  cv::Mat input;
  cv::Mat consts_map;
  cv::Mat consts_vals;
public:
  AlphaCalculator(int lambda, int win_size, double epsilon, cv::Mat input, cv::Mat consts_map, cv::Mat consts_vals);
  cv::Mat getAlpha();
};

#endif /* AlphaCalculator_hpp */
