//
//  LaplacianCalculator.hpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#ifndef LaplacianCalculator_hpp
#define LaplacianCalculator_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>

#include "./Eigen/SparseCore"
#include "./Eigen/Core"
#include "./Eigen/Dense"

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

class LaplacianCalculator {
private:
  // Input
  int win_size;
  double epsilon;
  cv::Mat input;
  cv::Mat consts_map;
  
  // Output
  Eigen::SparseMatrix<double> res;
public:
  LaplacianCalculator(int win_size, double epsilon, cv::Mat input, cv::Mat consts_map);
  Eigen::SparseMatrix<double> getLaplacianMatrix();
};

#endif /* LaplacianCalculator_hpp */
