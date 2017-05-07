#ifndef ImagePrinter_hpp
#define ImagePrinter_hpp

#include <stdio.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class Image_Printer
{
public:
  void printImage(cv::Mat imageMatrix, std::string title);
};

#endif /* ImagePrinter_hpp */
