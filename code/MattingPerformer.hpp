//
//  MattingPerformer.hpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#ifndef MattingPerformer_hpp
#define MattingPerformer_hpp

#include <stdio.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include "AlphaCalculator.hpp"

class MattingPerformer {
private:
  // Input
  double lambda;
  int win_size;
  double epsilon;
  double threshold;
  cv::Mat input;
  cv::Mat input_m;
  
  // Output
  cv::Mat mattingResultF;
  cv::Mat mattingResultB;
public:
  MattingPerformer(double lambda, int win_size, double epsilon, double threshold, cv::Mat input, cv::Mat input_m);
  void performMatting();
  cv::Mat getMattingResultF();
  cv::Mat getMattingResultB();
};

#endif /* MattingPerformer_hpp */
