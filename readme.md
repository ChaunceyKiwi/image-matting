## Natural Image Matting

This algorithm separates the background and the foreground of an image.

The code is written by myself but the train of thought is derived from http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf. 

The functions acting the calculation are from the libraries of openCV, Eigen and Suitesparse. 

A .pdf report is also provided in directory ./report for the purpose of illustrating the idea of this paper.

## How to install SuiteSpare
1. Install SuiteSparse.tar.gz from http://faculty.cse.tamu.edu/davis/suitesparse.html, unzip it to the directory /imageMatting/code.

2. Run following instructions to install library:
~~~
cd imageMatting/code/SuiteSparse;
make library;
~~~

## How to run
1. Before running the code, make sure you have already installed openCV library on your local machine.
2. Enter ./code and type:

~~~~
cd make
cmake .
make
./imageMatting
~~~~

## Files description in /code

### main.cpp
The main function to run the image matting.

### ImageReader.cpp
Given the file path of an image, ImageReader will read the image and return a matrix which represents the image.

### LaplacianCalculator.cpp
Given the matrices of image and scribbled image. The task of LaplacianCalculator is to get such a matrix L.

### SparseMatrixEquationSolver.cpp
SparseMatrixEquationSolver is specially used to solve the equation Ax = B, where A is a sparse matrix of N by N where N is the pixel number of an image.

### AlphaCalculator.cpp
Alpha calculator will firstly use LaplacianCalculator to get Laplacian matrix. Then it constructs the sparse matrix equation

### MattingPerformer.cpp
Matting performer uses AlphaCalculator to calculate the matte alpha. Then apply alpha to the original image and get matting image

### ImagePrinter.cpp
ImagePrinter receives the result from MattingPerformer and then prints the matting result. 

## Reference

[http://www.alphamatting.com/code.php](http://www.alphamatting.com/code.php)

[http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf](http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)
