## Natural Image Matting

This algorithm separates the background and foreground with a few scribbles as constriction.

The part of this code tries to realize this algorithm in iOS so we can learn whether Image Matting can be performed well or not in mobile devices.

The code is written by myself but the train of thought is derived from http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf. 

The functions acting the calculation are from the libraries of openCV, Eigen and Suitesparse. 

You can read .pdf paper in the /report to get a better and more stright-forward comprehension.

## Configuration

To run the code, you need to install openCV library and suiteSparse toolbox on your local machine.

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

### ImagePrinter.cppImagePrinter receieves the result from MattingPerformer and then prints the matting result. 

## Reference

[http://www.alphamatting.com/code.php](http://www.alphamatting.com/code.php)

[http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf](http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf)