#include "ImagePrinter.hpp"
#include <iostream>
using namespace std;

void Image_Printer::printImage(cv::Mat imageMatrix, std::string title) {  
  imwrite("../" + title + ".jpg", imageMatrix);
}
