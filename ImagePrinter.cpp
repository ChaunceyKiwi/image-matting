#include "ImagePrinter.hpp"

void Image_Printer::printImage(cv::Mat imageMatrix, std::string title) {
  // Create a window for display
  namedWindow(title, cv::WINDOW_AUTOSIZE);
  
  // Show image in the window created
  imshow(title, imageMatrix);
}
