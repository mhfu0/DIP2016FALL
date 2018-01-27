#include <iostream>
#include <string>
#include <cstdio>
#include <cassert>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat doDFT(Mat img);
Mat doIDFT(Mat imgComplex);
Mat makeSpecImg(Mat imgComplex);
void shift(Mat &specImg);

void createGHPF(Mat &mask, float d0);
void createGLPF(Mat &mask, float d0);

string type2str(int type);

int main(int argc, char ** argv)
{
    const char* filename = argc >=2 ? argv[1] : "selfie.jpg";
    Mat img = imread(filename, 0);
    if(img.empty())
        return -1;
   
    int opt;
    printf("Choose options from below:\n  [0] GHPF\n  [1] GLPF\n");
    scanf("%d", &opt);

switch(opt) {
case 0:
{
    Mat imgComplex = doDFT(img);
    Mat specImg = makeSpecImg(imgComplex);

    // create Gaussian highpass filter
    int d0;
    printf("GHPF: D0 = ");
    scanf("%d", &d0);
    Mat GHPF = Mat::zeros(img.size(), CV_32F);
    createGHPF(GHPF, d0);
    shift(GHPF);
    
    // multiply with Gaussian filter in freq. domain
    Mat planes[] = {Mat::zeros(GHPF.size(), CV_32F), Mat::zeros(GHPF.size(), CV_32F)};
    Mat kernel_spec;
    planes[0] = GHPF; // real
    planes[1] = GHPF; // imaginary
    merge(planes, 2, kernel_spec);
 
    mulSpectrums(imgComplex, kernel_spec, imgComplex, DFT_ROWS); // only DFT_ROWS accepted
    specImg = makeSpecImg(GHPF);
    imwrite("filter.jpg", specImg);
    specImg = makeSpecImg(imgComplex);
    imwrite("spectrum.jpg", specImg);
    
    // perform inverse fourier transform
    Mat dstImg = doIDFT(imgComplex);
    imwrite("output.jpg", dstImg);
    break;
}
case 1:
{
    Mat imgComplex = doDFT(img);
    Mat specImg = makeSpecImg(imgComplex);

    // create Gaussian highpass filter
    int d0;
    printf("GLPF: D0 = ");
    scanf("%d", &d0);
    Mat GLPF = Mat::zeros(img.size(), CV_32F);
    createGLPF(GLPF, d0);
    shift(GLPF);
    
    // multiply with Gaussian filter in freq. domain
    Mat planes[] = {Mat::zeros(GLPF.size(), CV_32F), Mat::zeros(GLPF.size(), CV_32F)};
    Mat kernel_spec;
    planes[0] = GLPF; // real
    planes[1] = GLPF; // imaginary
    merge(planes, 2, kernel_spec);
 
    mulSpectrums(imgComplex, kernel_spec, imgComplex, DFT_ROWS); // only DFT_ROWS accepted
    specImg = makeSpecImg(GLPF);
    imwrite("filter.jpg", specImg);
    specImg = makeSpecImg(imgComplex);
    imwrite("spectrum.jpg", specImg);
    
    // perform inverse fourier transform
    Mat dstImg = doIDFT(imgComplex);
    imwrite("output.jpg", dstImg);
    break;
}
default:
{
    fprintf(stderr, "bad parameter\n");
    return -1;
}

}
    
    return 0;
}

Mat doDFT(Mat img) {
    // expand input image to optimal size
    Mat padded;                            
    int m = getOptimalDFTSize(img.rows);
    int n = getOptimalDFTSize(img.cols);
    
    // expand border with zeros
    copyMakeBorder(img, padded, 0, m-img.rows, 0, n-img.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    
    Mat complex;
    merge(planes, 2, complex);
    dft(complex, complex);
    
    return complex;
}

Mat doIDFT(Mat imgComplex) {
    Mat dstImg;
    
    idft(imgComplex, dstImg, DFT_REAL_OUTPUT);
    
    normalize(dstImg, dstImg, 0, 1, CV_MINMAX);
    dstImg.convertTo(dstImg, CV_8U, 255.0);
    
    return dstImg;
}

Mat makeSpecImg(Mat imgComplex) {
    // compute the magnitude in logarithmic scale
    Mat planes[] = {Mat::zeros(imgComplex.size(), CV_32F), Mat::zeros(imgComplex.size(), CV_32F)};
    split(imgComplex, planes);  // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    
    magnitude(planes[0], planes[1], planes[0]);
    Mat specImg = planes[0];
    specImg += Scalar::all(1);  // avoid log(0)
    log(specImg, specImg);
    
    shift(specImg); // shift to center
    normalize(specImg, specImg, 0, 1, CV_MINMAX);
    specImg.convertTo(specImg, CV_8U, 255.0);
    
    return specImg;
}

void shift(Mat &specImg) {
    // crop when odd number of rows or columns
    specImg = specImg(Rect(0, 0, specImg.cols & -2, specImg.rows & -2));

    int cx = specImg.cols/2;
    int cy = specImg.rows/2;

    Mat q0(specImg, Rect(0, 0, cx, cy));
    Mat q1(specImg, Rect(cx, 0, cx, cy));
    Mat q2(specImg, Rect(0, cy, cx, cy)); 
    Mat q3(specImg, Rect(cx, cy, cx, cy));

    Mat tmp;
    // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    // swap quadrant (Top-Right with Bottom-Left)
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
}

void createGHPF(Mat &mask, float d0) {
    // center position
    int cx = mask.cols/2;
    int cy = mask.rows/2;
    for(int i=0; i<mask.cols; i++)
        for(int j=0; j<mask.rows; j++) {
            float px = i-cx+1;
            float py = j-cy+1;
            
            float d = sqrt(px*px+py*py);
            
            float fxy0 = exp(-pow(d,2)/(2*pow(d0,2)));
            float fxy = 1-fxy0;
            
            mask.at<float>(Point(i,j)) = fxy;
        }

}
void createGLPF(Mat &mask, float d0) {
    // center position
    int cx = mask.cols/2;
    int cy = mask.rows/2;
    for(int i=0; i<mask.cols; i++)
        for(int j=0; j<mask.rows; j++) {
            float px = i-cx+1;
            float py = j-cy+1;
            
            float d = sqrt(px*px+py*py);
            
            float fxy = exp(-pow(d,2)/(2*pow(d0,2)));
            mask.at<float>(Point(i,j)) = fxy;
        }

}

string type2str(int type) {
    // check the type of Mat
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}
