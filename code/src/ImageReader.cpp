#include "ImageReader.hpp"

cv::Mat ImageReader::readImage(std::string path) {
  return cv::imread(path, cv::IMREAD_COLOR);
}
