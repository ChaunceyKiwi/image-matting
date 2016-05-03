## Natural Image Matting

It's an algorithm used to separate the background and foreground with a few scribbles as constriction.

The part of this code tries to realize this algorithm in iOS so we can learn whether Image Matting can be performed well or not in mobile devices.

The code is written by myself but the train of thought is derived from http://www.wisdom.weizmann.ac.il/~levina/papers/Matting-Levin-Lischinski-Weiss-CVPR06.pdf. 

The functions which act the calculation is from the library of openCV, Eigen and suitesparse. 

## Note 
1. Skip erode.

## Update Log
02/05/2016: First version code. Still hard coded.

02/05/2016: Code polished and not hard coded any more.

## Demo
input:

![input] (https://raw.githubusercontent.com/ChaunceyKiwi/imageMatting/master/bmp/result/input.png)

output:

![output] (https://raw.githubusercontent.com/ChaunceyKiwi/imageMatting/master/bmp/result/output.png)

