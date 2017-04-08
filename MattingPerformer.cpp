#include "MattingPerformer.hpp"
#include <iostream>

MattingPerformer::MattingPerformer(double lambda, int win_size, double epsilon, double threshold, cv::Mat input, cv::Mat input_m)
{
  this->lambda = lambda;
  this->win_size = win_size;
  this->epsilon = epsilon;
  this->threshold = threshold;
  this->input = input;
  this->input_m = input_m;
  this->mattingResultF = NULL;
  this->mattingResultB = NULL;
}

cv::Mat MattingPerformer::getMattingResultF() {
  return this->mattingResultF;
}

cv::Mat MattingPerformer::getMattingResultB() {
  return this->mattingResultB;
}

void MattingPerformer::performMatting() {
  cv::Mat temp; // The difference between origin image and scribbled image
  cv::Mat consts_map; // 0-1 values where 1 means pixel scribbled
  cv::Mat consts_vals; // The original value of scribbled pixel
  int height = this->input.size().height;
  int width = this->input.size().width;
  
  // Find the scribbled pixels
  temp = abs(input - input_m);
  
  cv::Mat ch1, ch2, ch3;
  
  cv::Mat ch1_final, ch2_final, ch3_final;
  std::vector<cv::Mat> channels(3), channelsFinal(3);
    
  // Calculate consts_maps
  split(temp, channels);
  split(input_m, channelsFinal);
  ch1 = channels[0];
  ch2 = channels[1];
  ch3 = channels[2];
  ch1_final = channelsFinal[0];
  ch2_final = channelsFinal[1];
  ch3_final = channelsFinal[2];
  consts_map = (ch1 + ch2 +ch3) > this->threshold; //get scribbled pixels
  consts_map = consts_map / 255;
  
  // Calculate consts_vals
  ch1_final = ch1_final.mul(consts_map);
  ch2_final = ch2_final.mul(consts_map);
  ch3_final = ch3_final.mul(consts_map);
  consts_vals = ch1_final / 255;
  
  std::vector<cv::Mat> channels_F(3);
  cv::Mat ch1_F, ch2_F, ch3_F;
  split(input, channels_F);
  ch1_F = channels_F[0];
  ch2_F = channels_F[1];
  ch3_F = channels_F[2];
  
  std::vector<cv::Mat> channels_B(3);
  cv::Mat ch1_B, ch2_B, ch3_B;
  split(input, channels_B);
  ch1_B = channels_B[0];
  ch2_B = channels_B[1];
  ch3_B = channels_B[2];

  // Function to get Alpha by natural matting
  AlphaCalculator alphaCalculator(lambda, win_size, epsilon, input, consts_map, consts_vals);
  cv::Mat alpha_F = alphaCalculator.getAlpha();
  cv::Mat alpha_B = 1 - alphaCalculator.getAlpha();
  
  // Apply alpha to image to get image
  for(int i = 0;i < height; i++) {
    for(int j = 0; j < width; j++) {
      ch1_F.at<uchar>(i, j) = (uchar)((int)ch1_F.at<uchar>(i, j) * alpha_F.at<double>(i, j));
      ch2_F.at<uchar>(i, j) = (uchar)((int)ch2_F.at<uchar>(i, j) * alpha_F.at<double>(i, j));
      ch3_F.at<uchar>(i, j) = (uchar)((int)ch3_F.at<uchar>(i, j) * alpha_F.at<double>(i, j));
      ch1_B.at<uchar>(i, j) = (uchar)((int)ch1_B.at<uchar>(i, j) * alpha_B.at<double>(i, j));
      ch2_B.at<uchar>(i, j) = (uchar)((int)ch2_B.at<uchar>(i, j) * alpha_B.at<double>(i, j));
      ch3_B.at<uchar>(i, j) = (uchar)((int)ch3_B.at<uchar>(i, j) * alpha_B.at<double>(i, j));
    }
  }
  
  //combine 3 channels to generate image
  cv::Mat finalImageF, finalImageB;
  merge(channels_F, finalImageF);
  merge(channels_B, finalImageB);
  
  this->mattingResultF = finalImageF;
  this->mattingResultB = finalImageB;
}
