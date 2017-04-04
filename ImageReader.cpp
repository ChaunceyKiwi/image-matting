//
//  ImageReader.cpp
//  temp
//
//  Created by Chauncey on 2017-04-03.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#include "ImageReader.hpp"

cv::Mat ImageReader::readImage(std::string path) {
  return cv::imread(path, CV_LOAD_IMAGE_COLOR);
}
