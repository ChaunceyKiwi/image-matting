//
//  LaplacianCalculator.cpp
//  temp
//
//  Created by Chauncey on 2017-04-04.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#include "LaplacianCalculator.hpp"

LaplacianCalculator::LaplacianCalculator(int win_size, double epsilon, cv::Mat input, cv::Mat consts_map) {
  this->win_size = win_size;
  this->epsilon = epsilon;
  this->input = input;
  this->consts_map = consts_map;
}

Eigen::SparseMatrix<double> LaplacianCalculator::getLaplacianMatrix() {
  int len = 0;
  cv::Mat repe_col;
  cv::Mat repe_row;
  
  int neb_size = pow(win_size * 2 + 1, 2); // Size of window
  int neb_size_square = pow(neb_size, 2);
  int height = this->input.size().height;
  int width = this->input.size().width;
  int img_size = height * width;
  
  // Store the index of M
  cv::Mat indsM = cv::Mat::zeros(height, width, CV_32S);
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      indsM.at<int>(j, i) = height * i + j + 1;
    }
  }
  
  // consts_map_sub = consts_map[win_size: height-(win_size+1), width-(winsize+1)]
  cv::Mat consts_map_sub; // consts_map with margin excluded
  consts_map_sub = consts_map.rowRange(win_size, height - (win_size + 1));
  consts_map_sub = consts_map_sub.colRange(win_size, width - (win_size + 1));
  
  int tlen = (height - 2 * win_size) * (width - 2 * win_size);
  tlen -= sum(consts_map_sub)[0];
  tlen *= neb_size_square;
  
  cv::Mat row_inds = cv::Mat::zeros(tlen, 1, CV_32S);
  cv::Mat col_inds = cv::Mat::zeros(tlen, 1, CV_32S);
  cv::Mat vals     = cv::Mat::zeros(tlen, 1, CV_64F);
  
  // Iterate on all window center
  for (int j = win_size; j < width - win_size; j++) {
    for (int i = win_size; i < height - win_size; i++) {
      
      // Skip if the current pixel is scribbled
      if ((int)consts_map.at<char>(i, j) == 1) {
        continue;
      }
      
      // Calculate win_inds, which is a 1 by 9 matrix
      // The value is the index of all element in the window whose center is (i, j)
      cv::Mat win_inds  = cv::Mat::zeros(1, 9, CV_64F);
      cv::Mat temp = indsM.rowRange(i - win_size, i + win_size + 1);
      temp = temp.colRange(j - win_size, j + win_size + 1);
      win_inds.at<double>(0,0) = double(temp.at<int>(0,0));
      win_inds.at<double>(0,1) = double(temp.at<int>(1,0));
      win_inds.at<double>(0,2) = double(temp.at<int>(2,0));
      win_inds.at<double>(0,3) = double(temp.at<int>(0,1));
      win_inds.at<double>(0,4) = double(temp.at<int>(1,1));
      win_inds.at<double>(0,5) = double(temp.at<int>(2,1));
      win_inds.at<double>(0,6) = double(temp.at<int>(0,2));
      win_inds.at<double>(0,7) = double(temp.at<int>(1,2));
      win_inds.at<double>(0,8) = double(temp.at<int>(2,2));
      
      // Calculate winI, which is a 9 by 3 matrix
      // The values on each row are values of a pixel on 3 channels
      // in the window whose center is (i, j)
      // Each colomn as one kind of color depth of winI
      cv::Mat winI = input.rowRange(i - win_size, i + win_size + 1);
      winI = winI.colRange(j - win_size, j + win_size + 1);
      cv::Mat winI_temp  = cv::Mat::zeros(9, 3, CV_64F);
      std::vector<cv::Mat> channels(3);
      split(winI, channels);
      cv::Mat ch1 = channels[0];
      cv::Mat ch2 = channels[1];
      cv::Mat ch3 = channels[2];
      
      winI_temp.at<double>(0,0) = ch1.at<uchar>(0,0);
      winI_temp.at<double>(1,0) = ch1.at<uchar>(1,0);
      winI_temp.at<double>(2,0) = ch1.at<uchar>(2,0);
      winI_temp.at<double>(3,0) = ch1.at<uchar>(0,1);
      winI_temp.at<double>(4,0) = ch1.at<uchar>(1,1);
      winI_temp.at<double>(5,0) = ch1.at<uchar>(2,1);
      winI_temp.at<double>(6,0) = ch1.at<uchar>(0,2);
      winI_temp.at<double>(7,0) = ch1.at<uchar>(1,2);
      winI_temp.at<double>(8,0) = ch1.at<uchar>(2,2);
      
      winI_temp.at<double>(0,1) = ch2.at<uchar>(0,0);
      winI_temp.at<double>(1,1) = ch2.at<uchar>(1,0);
      winI_temp.at<double>(2,1) = ch2.at<uchar>(2,0);
      winI_temp.at<double>(3,1) = ch2.at<uchar>(0,1);
      winI_temp.at<double>(4,1) = ch2.at<uchar>(1,1);
      winI_temp.at<double>(5,1) = ch2.at<uchar>(2,1);
      winI_temp.at<double>(6,1) = ch2.at<uchar>(0,2);
      winI_temp.at<double>(7,1) = ch2.at<uchar>(1,2);
      winI_temp.at<double>(8,1) = ch2.at<uchar>(2,2);
      
      winI_temp.at<double>(0,2) = ch3.at<uchar>(0,0);
      winI_temp.at<double>(1,2) = ch3.at<uchar>(1,0);
      winI_temp.at<double>(2,2) = ch3.at<uchar>(2,0);
      winI_temp.at<double>(3,2) = ch3.at<uchar>(0,1);
      winI_temp.at<double>(4,2) = ch3.at<uchar>(1,1);
      winI_temp.at<double>(5,2) = ch3.at<uchar>(2,1);
      winI_temp.at<double>(6,2) = ch3.at<uchar>(0,2);
      winI_temp.at<double>(7,2) = ch3.at<uchar>(1,2);
      winI_temp.at<double>(8,2) = ch3.at<uchar>(2,2);
      
      winI = winI_temp / 255;
      
      // Calculate mean value of matrix, which is 3 by 1
      cv::Mat win_mu = cv::Mat::zeros(1, 3, CV_64F);
      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;
      
      for(int i = 0; i < neb_size;i++) {
        sum1 += winI.at<double>(i, 0);
        sum2 += winI.at<double>(i, 1);
        sum3 += winI.at<double>(i, 2);
      }
      
      win_mu.at<double>(0, 0) = sum1 / neb_size;
      win_mu.at<double>(0, 1) = sum2 / neb_size;
      win_mu.at<double>(0, 2) = sum3 / neb_size;
      
      // Calculate the covariance matrix
      // Cov = E(X^2) - E(X)^2
      cv::Mat expection_xx = winI.t() * winI / neb_size;
      cv::Mat expection_x = win_mu.t() * win_mu;
      cv::Mat covariance = expection_xx - expection_x;
      
      // Calculate (Cov + epsilon / |Wk| * I_3)^(-1)
      cv::Mat eye_c = cv::Mat::eye(input.channels(), input.channels(), CV_64F);
      cv::Mat before_inv = covariance + epsilon / neb_size * eye_c;
      cv::Mat win_var = before_inv.inv();
      
      // Calculate Ii - Mu_k and Ij - Mu_k, which are 9 by 3 matrix
      cv::Mat IiMinusMuk = winI - repeat(win_mu, neb_size, 1);
      cv::Mat IjMinusMuk = IiMinusMuk.t();
      
      // Calcualte the part on the right hand side of Kronecker delta
      // which is the matrix with size of 9 by 9, and then put them in one column
      cv::Mat eyeMatrix = cv::Mat::eye(neb_size, neb_size, CV_64F);
      cv::Mat tvals = eyeMatrix - (1 + IiMinusMuk * win_var * IjMinusMuk) / neb_size;
      tvals = tvals.reshape(0, neb_size_square);
      
      repe_col = repeat(win_inds.t(), 1, neb_size).reshape(0, neb_size_square);
      repe_row = repeat(win_inds, neb_size, 1).reshape(0, neb_size_square);
      cv::Mat putInRow = tvals.reshape(0, neb_size_square);
      
      for(int i = len; i < neb_size_square + len; i++) {
        row_inds.at<int>(i, 0) = repe_row.at<double>(i - len, 0);
        col_inds.at<int>(i, 0) = repe_col.at<double>(i - len, 0);
        vals.at<double>(i, 0) = tvals.at<double>(i - len, 0);
      }
      
      len = len + neb_size_square;
    }
  }
  
  // Convert row_inds, col_inds, vals to a tripletList
  // Then convert tripletList into a sparse matrix
  SpMat A(img_size, img_size);
  std::vector<T> tripletList;
  tripletList.reserve(len);
  for (int i = 0;i < len; i++) {
    tripletList.push_back(
                          T(row_inds.at<int>(i, 0) - 1,
                            col_inds.at<int>(i, 0) - 1,
                            vals.at<double>(i, 0)));
  }
  A.setFromTriplets(tripletList.begin(), tripletList.end());
  
  return A;
}
