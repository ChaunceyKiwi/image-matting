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
