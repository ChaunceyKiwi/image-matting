//
//  Image.hpp
//  temp
//
//  Created by Chauncey on 2017-04-03.
//  Copyright © 2017 Chauncey. All rights reserved.
//

#ifndef Image_hpp
#define Image_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>

class Image {
private:
  int width;
  int height;
  int img_size;
  cv::Mat matrix;
public:
  Image(cv::Mat matrix);
  int getWidth();
  int getHeight();
  int getImageSize();
  cv::Mat getMatrix();
};

#endif /* Image_hpp */
