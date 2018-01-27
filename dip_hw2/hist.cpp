#include <cstdio>
#include <cassert>
#include <string>

#include <opencv2/opencv.hpp>
#define cvQueryHistValue_1D( hist, idx0 ) \
cvGetReal1D( (hist)->bins, (idx0) )


using namespace std;
using namespace cv;

// parameters for calcHist()
int nBins = 256;
float range[2] = {0,256};
const float* histRange = {range};

int hist_w=nBins;
int hist_h=256;

void genHistImg(Mat &hist_o, Mat &histImg) {
    Mat hist = hist_o.clone();    
    normalize(hist, hist, 0, 1, NORM_MINMAX);
    
    // plot histogram
    histImg.setTo(Scalar(0));
    for(int i=1; i<nBins; i++)
    {
        line(histImg, Point(i-1, hist_h-hist.at<float>(i-1)*hist_h), Point(i, hist_h-hist.at<float>(i)*hist_h), Scalar(255));
    }
} 

int main(int argc, char** argv)
{
    assert(argc==2);    
    
    Mat srcImg;     // source image
    Mat hist;       // histogram
    Mat histImg(hist_h, hist_w, CV_8U, Scalar(0));    // visualize histogram

    // load image
    srcImg = imread(argv[1], 0);
    assert(srcImg.data);
    
    // Step1: draw histogram
    calcHist(&srcImg, 1, 0, Mat(), hist, 1, &nBins, &histRange);
    genHistImg(hist, histImg);
    imwrite("src_hist.jpg", histImg);
    
    /// Step2-1: gamma transformation
    Mat gammaImg;
    
    // build look-up table (lut)
    Mat lut(256,1,CV_8U,Scalar(0));
    float gamma = 2.5f;
    for(int i=0; i<256; i++)
        lut.at<uchar>(i) = pow(((float)i/256.0f), gamma)*256.0f;
    
    gammaImg = srcImg.clone();
    LUT(srcImg, lut, gammaImg);
    
    calcHist(&gammaImg, 1, 0, Mat(), hist, 1, &nBins, &histRange);
    genHistImg(hist, histImg);
    
    imwrite("gamma.jpg", gammaImg);
    imwrite("gamma_hist.jpg", histImg);
    
    // Step2-2: degration
    Mat degImg;
    
    // build look-up table
    lut.setTo(Scalar(0));
    float dfactor = 0.6f;
    for(int i=0; i<256; i++)
        lut.at<uchar>(i) = (float)i*dfactor;
    
    degImg = gammaImg.clone();
    LUT(gammaImg, lut, degImg);
    calcHist(&degImg, 1, 0, Mat(), hist, 1, &nBins, &histRange);
    genHistImg(hist, histImg);

    imwrite("degrad.jpg", degImg);
    imwrite("degrad_hist.jpg", histImg);
    
    
    // Step3: histogram stretch
    Mat strImg;
    
    // build look-up table
    lut.setTo(Scalar(0));
    
    // find r_min and r_max
    int r_min = 0;
    int r_max = 255;
    for(int i=0; i<256; i++) {
        if(hist.at<float>(i) > 0.0f)
            break;
        r_min = i;
    }

    for(int i=255; i>=0; i--) {
        if(hist.at<float>(i) > 0.0f)
            break;
        r_max = i;
    }
    assert(r_max > r_min);
    
    for(int i=0; i<256; i++) {
        if(i<=r_min)
            lut.at<uchar>(i) = 0;
        else if(i>r_max)
            lut.at<uchar>(i) = 255;
        else
            lut.at<uchar>(i) = 255.0f*((float)(i-r_min))/((float)(r_max-r_min));
    }
    
    strImg = degImg.clone();
    LUT(degImg, lut, strImg);   
    
    calcHist(&strImg, 1, 0, Mat(), hist, 1, &nBins, &histRange);
    genHistImg(hist, histImg);
    
    imwrite("stretch.jpg", strImg);
    imwrite("stretch_hist.jpg", histImg);
    
    // Step4: Histogram Equalization
    Mat equImg;
    
    // take degImg as input
    calcHist(&degImg, 1, 0, Mat(), hist, 1, &nBins, &histRange);
    
    // obtain the CDF function (lut)
    // CDF function is the transformation function
    int npixels = degImg.rows*degImg.cols;
    float factor = 255.0f/(float)npixels;
    
    lut.setTo(Scalar(0));
    lut.convertTo(lut, CV_32F);
    
    int sum = 0;
    for(int i=0; i<nBins; i++) {
        sum += hist.at<float>(i);
        lut.at<float>(i)=roundf(factor*sum);
    }
    
    lut.convertTo(lut, CV_8U);
    
    
    equImg = degImg.clone();
    LUT(degImg, lut, equImg);
    calcHist(&equImg, 1, 0, Mat(), hist, 1, &nBins, &histRange);
    
    genHistImg(hist, histImg);
    
    imwrite("equal.jpg", equImg);
    imwrite("equal_hist.jpg", histImg);
    
    return 0;
    
}
