#include "Image.hpp"

Image::Image(cv::Mat matrix) {
  this->height = matrix.size().height;
  this->width = matrix.size().width;
  this->matrix = matrix;
}

int Image::getWidth() {
  return this->width;
}

int Image::getHeight() {
  return this->height;
}

int Image::getImageSize() {
  return this->height * this->width;
}

cv::Mat Image::getMatrix() {
  return this->matrix;
}
