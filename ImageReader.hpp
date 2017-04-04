//
//  ImageReader.hpp
//  temp
//
//  Created by Chauncey on 2017-04-03.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#ifndef ImageReader_hpp
#define ImageReader_hpp

#include <stdio.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageReader {
public:
  cv::Mat readImage(std::string path);
};

#endif /* ImageReader_hpp */
