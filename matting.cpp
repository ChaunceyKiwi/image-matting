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

using namespace cv;
using namespace std;
using namespace Eigen;

#define epsilon 0.0000001
#define thresholdForScribble 0.001

string path_prefix = "/Users/Chauncey/Workspace/imageMatting";

typedef SparseMatrix<double> SpMat;
typedef Triplet<double> T;

int win_size = 1; // Distance between center to border
int neb_size = (win_size * 2 + 1) * (win_size * 2 + 1); // Size of window
int lambda = 100; // Weight of scribbled piexel obedience

Mat Matting(Mat input, Mat input_m, int ForB);
Mat GetAlpha(Mat input, Mat consts_map, Mat consts_vals);
void getAlphaFromTxt(double* alpha);
SpMat GetLaplacian(Mat input, Mat consts_map);
void exportDataToTxtFile(SpMat left, VectorXd right);
void solveEquation(int *Ap, int *Ai, double* Ax, double *b, int n, double *alphaArray);

int main(void) {
  // Read the file
  Mat img, img_m;
  string img_name = path_prefix + "/bmp/pic.bmp";
  string img_m_name = path_prefix + "/bmp/pic_m.bmp";
  img = imread(img_name, CV_LOAD_IMAGE_COLOR);
  img_m = imread(img_m_name, CV_LOAD_IMAGE_COLOR);

  Mat imgOutputF = Matting(img, img_m, 0);
  Mat imgOutputB = Matting(img, img_m, 1);

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

  // Get the height and width of image
  int h = input.size().height;
  int w = input.size().width;

  // Find the scribbled pixels
  temp = abs(input - input_m);
  Mat ch1, ch2, ch3;
  Mat ch1_f, ch2_f, ch3_f;
  vector<Mat> channels(3),channelsFinal(3);
  split(temp, channels);
  split(input_m,channelsFinal);
  ch1 = channels[0];
  ch2 = channels[1];
  ch3 = channels[2];
  ch1_f = channelsFinal[0];
  ch2_f = channelsFinal[1];
  ch3_f = channelsFinal[2];

  consts_map = (ch1 + ch2 +ch3) > thresholdForScribble; //get scribbled pixels
  split(input,channels);
  ch1 = channels[0];
  ch2 = channels[1];
  ch3 = channels[2];
  ch1_f = ch1_f.mul(consts_map);
  ch2_f = ch2_f.mul(consts_map);
  ch3_f = ch3_f.mul(consts_map);
  consts_map = consts_map/255;
  consts_vals = ch1_f/255;

  // Function to get Alpha by natural matting
  Mat alpha = GetAlpha(input, consts_map, consts_vals);
  if(ForB == 1) {
    alpha = 1 - alpha;
  }

  // Apply alpha to image to get image
  for(int i = 0;i < h;i++) {
    for(int j = 0; j < w;j++) {
      ch1.at<uchar>(i,j) = (uchar)((int)ch1.at<uchar>(i,j) * alpha.at<double>(i,j));
      ch2.at<uchar>(i,j) = (uchar)((int)ch2.at<uchar>(i,j) * alpha.at<double>(i,j));
      ch3.at<uchar>(i,j) = (uchar)((int)ch3.at<uchar>(i,j) * alpha.at<double>(i,j));
    }
  }

  //combine 3 channels to 1 matrix
  merge(channels,finalImage);
  return finalImage;
}

// Function to get Alpha by natural matting
Mat GetAlpha(Mat input, Mat consts_map, Mat consts_vals) {
  Mat alpha;
  int img_size = input.size().height * input.size().width;

  // Solve the equation x = (A + lambda*D) \ (lambda * consts_vals(:));
  // To make it clear, let left * x = right
  // left =  A+lambda*D and right = lambda*consts_vals(:)

  // Calculation of left side(A + lambda * D)
  SpMat A = GetLaplacian(input, consts_map);
  Mat consts_map_trans = consts_map.t();
  SpMat D(img_size,img_size);
  for (int i = 0; i < img_size; i++) {
    D.coeffRef(i,i) = (int)consts_map_trans.at<char>(0, i);
  }
  SpMat left = A + lambda * D;

  // Calculation of right side((lambda * consts_vals(:))
  Mat consts_vals_in_a_col;
  Mat transpo = consts_vals.t();
  consts_vals_in_a_col = transpo.reshape(1,img_size);
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
  for (int i = 0; i < input.size().width; i++) {
    for (int j = 0; j < input.size().height; j++) {
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
// Annotation unfinished for this function
SpMat GetLaplacian(Mat input, Mat consts_map){
  int len(0);
  vector<T> tripletList;

  // Annotation later
  Mat consts_map_sub;
  Mat row_inds;
  Mat col_inds;
  Mat vals;
  Mat win_inds;
  Mat col_sum;
  Mat winI;
  Mat repe_col;
  Mat repe_row;

  //neb_size as the windows size (win_size is just the distance between center to border)
  int neb_size = (win_size * 2 + 1) * (win_size * 2 + 1);
  int h = input.size().height;
  int w = input.size().width;
  int img_size = w * h;
  double tlen = ((h - 2 * win_size) * (w - 2 * win_size)
                 - sum(consts_map_sub)[0]) * neb_size * neb_size;

  Mat indsM = Mat::zeros(h, w, CV_32S);
  for(int i = 0; i <= w -1 ;i++)
    for(int j = 0; j <= h - 1; j++){
      indsM.at<int>(j, i) = h * i + j + 1;
    }

  consts_map_sub = consts_map.rowRange(win_size,
                                       h - (win_size + 1)).colRange(win_size, w - (win_size + 1));

  row_inds = Mat::zeros(tlen, 1, CV_32S);
  col_inds = Mat::zeros(tlen, 1, CV_32S);
  vals     = Mat::zeros(tlen, 1, CV_64F);

  for (int j = win_size;j <= w - win_size - 1;j++) {
    for (int i = win_size;i <= h - win_size - 1;i++) {
      if ((int)consts_map.at<char>(i,j) == 1) {
        continue;
      }

      //all elements in the window whose center is (i,j) and add their index up to a line
      win_inds = indsM.rowRange(i - win_size,i + win_size + 1)\
      .colRange(j - win_size, j + win_size + 1);

      Mat col_sum  = Mat::zeros(1, 9, CV_64F);

      col_sum.at<double>(0,0) = double(win_inds.at<int>(0,0));
      col_sum.at<double>(0,1) = double(win_inds.at<int>(1,0));
      col_sum.at<double>(0,2) = double(win_inds.at<int>(2,0));
      col_sum.at<double>(0,3) = double(win_inds.at<int>(0,1));
      col_sum.at<double>(0,4) = double(win_inds.at<int>(1,1));
      col_sum.at<double>(0,5) = double(win_inds.at<int>(2,1));
      col_sum.at<double>(0,6) = double(win_inds.at<int>(0,2));
      col_sum.at<double>(0,7) = double(win_inds.at<int>(1,2));
      col_sum.at<double>(0,8) = double(win_inds.at<int>(2,2));

      win_inds = col_sum;

      //all elements in the window whose center is (i,j) and add their color up to a line
      winI = input.rowRange(i - win_size,i + win_size + 1)\
      .colRange(j - win_size, j + win_size + 1);

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

      //reshape winI. Each colomn as one kind of color depth of winI
      winI = winI_temp / 255;

      //get the mean value of matrix
      Mat win_mu = Mat::zeros(3, 1, CV_64F);

      double sum1 = 0;
      double sum2 = 0;
      double sum3 = 0;

      for(int i = 0; i <= 8;i++){
        sum1 += winI.at<double>(i, 0);
        sum2 += winI.at<double>(i, 1);
        sum3 += winI.at<double>(i, 2);
      }

      win_mu.at<double>(0, 0) = sum1 / 9;
      win_mu.at<double>(1, 0) = sum2 / 9;
      win_mu.at<double>(2, 0) = sum3 / 9;

      //get the variance value of matrix
      Mat win_mu_squ = win_mu.t();
      Mat multi = win_mu * win_mu_squ;
      Mat multi2 = winI.t() * winI / neb_size;
      Mat eye_c = Mat::eye(input.channels(), input.channels(), CV_64F);
      Mat before_inv = multi2 - multi + epsilon/neb_size * eye_c;
      Mat win_var = before_inv.inv();
      winI = winI - repeat(win_mu_squ, neb_size, 1);
      Mat tvals = (1 + winI * win_var * winI.t()) / neb_size;
      tvals = tvals.reshape(0,neb_size * neb_size);
      repe_col = repeat(win_inds.t(),1,neb_size).reshape(0, neb_size * neb_size);
      repe_row = repeat(win_inds,neb_size,1).reshape(0, neb_size * neb_size);
      Mat putInRow = tvals.reshape(0,neb_size * neb_size);

      for(int i = len;i <= neb_size * neb_size + len - 1;i++){
        row_inds.at<int>(i,0) = repe_row.at<double>(i - len,0);
        col_inds.at<int>(i,0) = repe_col.at<double>(i - len,0);
        vals.at<double>(i,0) = tvals.at<double>(i - len,0);
      }

      len = len + neb_size * neb_size;
    }
  }

  SpMat A(img_size,img_size);
  SpMat matrixOfOne(img_size,1);

  for (int i = 0;i < img_size ;i++) {
    matrixOfOne.insert(i, 0) = 1;
  }

  tripletList.reserve(len);
  for (int i = 0;i < len ;i++) {
    tripletList.push_back(T(row_inds.at<int>(i, 0) - 1, \
                            col_inds.at<int>(i, 0) - 1, vals.at<double>(i, 0)));
  }

  A.setFromTriplets(tripletList.begin(), tripletList.end());
  SpMat sumA = A * matrixOfOne;
  SpMat sparse_mat(img_size,img_size);

  for(int i = 0; i < img_size; i++) {
    sparse_mat.coeffRef(i,i) = sumA.coeffRef(i, 0);
  }

  A = sparse_mat - A;

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
