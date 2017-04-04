#include <iostream>
#include <vector>
#include <fstream>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./Eigen/SparseCore"
#include "./Eigen/Core"
#include "./Eigen/Dense"
#include "umfpack.h"
#include "ImageReader.hpp"
#include "Image.hpp"

using namespace cv;
using namespace std;
using namespace Eigen;

#define epsilon 0.0000001
#define thresholdForScribble 0.001

typedef SparseMatrix<double> SpMat;
typedef Triplet<double> T;

// file path
string path_prefix = "/Users/Chauncey/Workspace/imageMatting";
string img_path = path_prefix + "/bmp/kid/kid.bmp";
string img_m_path = path_prefix + "/bmp/kid/kid_m.bmp";

// global variable
Mat alpha;
int lambda = 100; // Weight of scribbled piexel obedience
int win_size = 1; // The distance between center and border
int neb_size = pow(win_size * 2 + 1, 2); // Size of window
int neb_size_square = pow(neb_size, 2);
int height, width, img_size;

// function declaration
Mat Matting(Mat input, Mat input_m, int ForB);
Mat GetAlpha(Mat input, Mat consts_map, Mat consts_vals);
void getAlphaFromTxt(double* alpha);
SpMat GetLaplacian(Mat input, Mat consts_map);
void solveEquation(int *Ap, int *Ai, double* Ax, double *b, int n, double *alphaArray);

int main(void)
{
  ImageReader imageReader;
  Image img(imageReader.readImage(img_path));
  Image img_m(imageReader.readImage(img_m_path));

  height = img.getHeight();
  width = img.getWidth();
  img_size = height * width;
  
  Mat imgOutputF = Matting(img.getMatrix(), img_m.getMatrix(), 0);
  Mat imgOutputB = Matting(img.getMatrix(), img_m.getMatrix(), 1);
  
  namedWindow("FrontObject", WINDOW_AUTOSIZE); // Create a window for display.
  imshow("FrontObject", imgOutputF); // Show our image inside it.
  namedWindow("Background", WINDOW_AUTOSIZE); // Create a window for display.
  imshow("Background", imgOutputB); // Show our image inside it.

  waitKey(0); // Wait for a keystroke in the window
  return 0;
}

Mat Matting(Mat input, Mat input_m ,int ForB){
  Mat temp; // The difference between origin image and scribbled image
  Mat consts_map; // 0-1 values where 1 means pixel scribbled
  Mat consts_vals; // The original value of scribbled pixel
  Mat finalImage; // return image after matting

  // Find the scribbled pixels
  temp = abs(input - input_m);
  
  Mat ch1, ch2, ch3;
  Mat ch1_f, ch2_f, ch3_f;
  vector<Mat> channels(3), channelsFinal(3);
  
  // Calculate consts_maps
  split(temp, channels);
  split(input_m, channelsFinal);
  ch1 = channels[0];
  ch2 = channels[1];
  ch3 = channels[2];
  ch1_f = channelsFinal[0];
  ch2_f = channelsFinal[1];
  ch3_f = channelsFinal[2];
  consts_map = (ch1 + ch2 +ch3) > thresholdForScribble; //get scribbled pixels
  consts_map = consts_map / 255;

  // Calculate consts_vals
  split(input, channels);
  ch1 = channels[0];
  ch2 = channels[1];
  ch3 = channels[2];
  ch1_f = ch1_f.mul(consts_map);
  ch2_f = ch2_f.mul(consts_map);
  ch3_f = ch3_f.mul(consts_map);
  consts_vals = ch1_f / 255;

  // Function to get Alpha by natural matting
  if (ForB == 0) {
    alpha = GetAlpha(input, consts_map, consts_vals);
  }else if(ForB == 1) {
    alpha = 1 - alpha;
  }

  // Apply alpha to image to get image
  for(int i = 0;i < height; i++) {
    for(int j = 0; j < width; j++) {
      ch1.at<uchar>(i,j) = (uchar)((int)ch1.at<uchar>(i,j) * alpha.at<double>(i,j));
      ch2.at<uchar>(i,j) = (uchar)((int)ch2.at<uchar>(i,j) * alpha.at<double>(i,j));
      ch3.at<uchar>(i,j) = (uchar)((int)ch3.at<uchar>(i,j) * alpha.at<double>(i,j));
    }
  }

  //combine 3 channels to generate image
  merge(channels, finalImage);
  return finalImage;
}

// Function to get Alpha by natural matting
Mat GetAlpha(Mat input, Mat consts_map, Mat consts_vals) {
  Mat alpha;

  // Solve the equation x = (A + lambda * D) \ (lambda * consts_vals(:));
  // To make it clear, let left * x = right
  // left = A + lambda * D and right = lambda * consts_vals(:)

  // Calculation of left side(A + lambda * D)
  SpMat A = GetLaplacian(input, consts_map);
  Mat consts_map_trans = consts_map.t();
  SpMat D(img_size, img_size);
  for (int i = 0; i < img_size; i++) {
    D.coeffRef(i,i) = (int)consts_map_trans.at<char>(0, i);
  }
  SpMat left = A + lambda * D;

  // Calculation of right side((lambda * consts_vals(:))
  Mat consts_vals_in_a_col;
  Mat transpo = consts_vals.t();
  consts_vals_in_a_col = transpo.reshape(1, img_size);
  VectorXd right(img_size);
  for (int i = 0;i < img_size;i++) {
    right(i) = lambda * consts_vals_in_a_col.at<char>(i,0);
  }

  int* innerPointer = left.innerIndexPtr();
  int* outerPointer = left.outerIndexPtr();
  double* valuePointer = left.valuePtr();
  double* rightPointer = right.data();
  double alphaArray[img_size];

  int size = input.size().width * input.size().height;
  solveEquation(outerPointer, innerPointer, valuePointer, rightPointer, size, alphaArray);
  alpha = Mat::ones(input.size().height, input.size().width, CV_64F);

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

// Funtion used to get the value of matting laplacian
SpMat GetLaplacian(Mat input, Mat consts_map){
  int len = 0;
  Mat repe_col;
  Mat repe_row;
  
  // Store the index of M
  Mat indsM = Mat::zeros(height, width, CV_32S);
  for(int i = 0; i < width; i++) {
    for(int j = 0; j < height; j++) {
      indsM.at<int>(j, i) = height * i + j + 1;
    }
  }
  
  // consts_map_sub = consts_map[win_size: height-(win_size+1), width-(winsize+1)]
  Mat consts_map_sub; // consts_map with margin excluded
  consts_map_sub = consts_map.rowRange(win_size, height - (win_size + 1));
  consts_map_sub = consts_map_sub.colRange(win_size, width - (win_size + 1));
  
  int tlen = (height - 2 * win_size) * (width - 2 * win_size);
  tlen -= sum(consts_map_sub)[0];
  tlen *= neb_size_square;
  
  Mat row_inds = Mat::zeros(tlen, 1, CV_32S);
  Mat col_inds = Mat::zeros(tlen, 1, CV_32S);
  Mat vals     = Mat::zeros(tlen, 1, CV_64F);
  
  // Iterate on all window center
  for (int j = win_size; j < width - win_size; j++) {
    for (int i = win_size; i < height - win_size; i++) {
      
      // Skip if the current pixel is scribbled
      if ((int)consts_map.at<char>(i, j) == 1) {
        continue;
      }
      
      // Calculate win_inds, which is a 1 by 9 matrix
      // The value is the index of all element in the window whose center is (i, j)
      Mat win_inds  = Mat::zeros(1, 9, CV_64F);
      Mat temp = indsM.rowRange(i - win_size, i + win_size + 1);
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
      Mat winI = input.rowRange(i - win_size, i + win_size + 1);
      winI = winI.colRange(j - win_size, j + win_size + 1);
      Mat winI_temp  = Mat::zeros(9, 3, CV_64F);
      vector<Mat> channels(3);
      split(winI, channels);
      Mat ch1 = channels[0];
      Mat ch2 = channels[1];
      Mat ch3 = channels[2];

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
      Mat win_mu = Mat::zeros(1, 3, CV_64F);
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
      Mat expection_xx = winI.t() * winI / neb_size;
      Mat expection_x = win_mu.t() * win_mu;
      Mat covariance = expection_xx - expection_x;
      
      // Calculate (Cov + epsilon / |Wk| * I_3)^(-1)
      Mat eye_c = Mat::eye(input.channels(), input.channels(), CV_64F);
      Mat before_inv = covariance + epsilon / neb_size * eye_c;
      Mat win_var = before_inv.inv();
      
      // Calculate Ii - Mu_k and Ij - Mu_k, which are 9 by 3 matrix
      Mat IiMinusMuk = winI - repeat(win_mu, neb_size, 1);
      Mat IjMinusMuk = IiMinusMuk.t();
      
      // Calcualte the part on the right hand side of Kronecker delta
      // which is the matrix with size of 9 by 9, and then put them in one column
      Mat eyeMatrix = Mat::eye(neb_size, neb_size, CV_64F);
      Mat tvals = eyeMatrix - (1 + IiMinusMuk * win_var * IjMinusMuk) / neb_size;
      tvals = tvals.reshape(0, neb_size_square);
      
      repe_col = repeat(win_inds.t(), 1, neb_size).reshape(0, neb_size_square);
      repe_row = repeat(win_inds, neb_size, 1).reshape(0, neb_size_square);
      Mat putInRow = tvals.reshape(0, neb_size_square);
      
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
  vector<T> tripletList;
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

void solveEquation(int *Ap, int *Ai, double* Ax, double *b, int n, double *alphaArray)
{
  double x[n];
  void *Symbolic, *Numeric;

  /* symbolic analysis */
  umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, NULL, NULL);

  /* LU factorization */
  umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, NULL, NULL);
  umfpack_di_free_symbolic(&Symbolic);

  /* solve system */
  umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, NULL, NULL);
  umfpack_di_free_numeric(&Numeric);

  for (int i = 0; i < n; i++) {
    // printf("x[%d] = %g\n", i, x[i]);
    alphaArray[i] = x[i];
  }
}
