//
//  Image.cpp
//  temp
//
//  Created by Chauncey on 2017-04-03.
//  Copyright © 2017 Chauncey. All rights reserved.
//

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

cv::Mat Image::getMatrix() {
  return this->matrix;
}
