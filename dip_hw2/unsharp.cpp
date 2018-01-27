#include <cstdio>
#include <cassert>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    assert(argc == 3);
    
    Mat srcImg, smoothImg;
    Mat maskImg;
    
    // load image
    srcImg = imread(argv[1], 0);
    assert(srcImg.data);
    
    // 6.1: smoothing with 5x5 box filter
    boxFilter(srcImg, smoothImg, -1, Size(5,5));
    imwrite("smooth.jpg", smoothImg);
    
    // 6.2: create unsharp masking image
    assert(srcImg.size == smoothImg.size);
    subtract(srcImg, smoothImg, maskImg);
    imwrite("mask.jpg", maskImg);
    
    // 6.3: unsharp masking
    float k = atof(argv[2]);
    Mat dstImg = srcImg + k*maskImg;
    
    ostringstream buff;
    buff << "unsharp_" << k << ".jpg";
    imwrite(buff.str(), dstImg);
    
    return 0;
}
