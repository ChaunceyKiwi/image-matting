## Natural Image Matting

Given an image, the code in this project can separate its foreground and background. The algorithm is derived from  Levin's research[1] and I have implemented this algorithm in C++. This applicaiton requires openCV, Eigen and Suitesparse. Please make sure you have installed these libraries before building and running my application.

A pdf report is also included in `report` folder to help you understand how this algorithm works.

## Before Running

### 1. Install OpenCV

It can be much easier to install OpenCV on macOS with Homebrew.
1. If you do not have Homebrew, please [install it](https://brew.sh).
2. Run the following commands to install OpenCV.
~~~~
brew update
brew tap homebrew/homebrew-science
brew install opencv
~~~~

### 2. Install Eigen

This repo contains the required source files of Eigen. You do not need to install it explicitly.

### 3. Install Suiteparse
1. Download `SuiteSparse` from http://faculty.cse.tamu.edu/davis/suitesparse.html and unzip it to `/imageMatting/code`.

2. Build the library using the following commands.
~~~
cd imageMatting/code/SuiteSparse;
make library;
~~~

## Build and Run
1. Set the image path in `src/main.cpp`. Put the original image and the scribbled image in the corresponding path.
2. Build the binaries.
~~~~
cd code
cd make
cmake .
make
~~~~

3. The application can now run on the image provided in `main.cpp`.
~~~~
./imageMatting
~~~~

## Overview of My Implementation

This section explains the function of each file in my implementation. These files are in `code/src` folder. For the details and the workflow of this algorithm, please refer to the report in `report` folder. 

### 1. main.cpp
The main function to run the ImageMatting App. The path to the source image is also assigned in this file.

### 2. Image.cpp
This is the abstract class of the images. It provides some methods of retrieving the properties of images.

### 3. ImageReader.cpp
Given the file path of an image, ImageReader will read the image and return the matrix representation of the image.

### 4. LaplacianCalculator.cpp
Given the matrices of the image and the scribbled image, LaplacianCalculator calculates the Laplacian of the image.

### 5. SparseMatrixEquationSolver.cpp
SparseMatrixEquationSolver is specifically used to solve the equation Ax = B, where A is a sparse matrix of N by N and N is the number of pixels in the image.

### 6. AlphaCalculator.cpp
AlphaCalculator will firstly use LaplacianCalculator to get a Laplacian matrix. Then it constructs the sparse matrix equation Ax = B.

### 7. MattingPerformer.cpp
MattingPerformer uses AlphaCalculator to calculate the `matte alpha`. Then it applies `matte alpha` to the original image to get matting image. The matting image is the dot product of the original image and `matte alpha`.

### 8. ImagePrinter.cpp
ImagePrinter receives the result from MattingPerformer and then prints the matting result. 

## Reference

[1] A. Levin D. Lischinski and Y. Weiss. A Closed Form Solution to Natural Image Matting.
[(http://webee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)](http://webee.technion.ac.il/people/anat.levin/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)

[2] Alpha Matting Evaluation [http://www.alphamatting.com/code.php](http://www.alphamatting.com/code.php)
