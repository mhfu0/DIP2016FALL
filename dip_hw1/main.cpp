#include <cstdio>
#include <string>
#include <cassert>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void printMat(Mat &mat);
int bilinear(uchar p[2][2], double x, double y);
int bicubic(uchar p[4][4], double x, double y);
int myresize_l(Mat &srcMat, Mat &dstMat, double s);
int myresize_c(Mat &srcMat, Mat &dstMat, double s);

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("usage: main.out <image_path> <scaling_factor> <option>\n");
        printf("<option>:\n\t0: bilinear inter.\n\t1: bicubic inter.\n");
        return -1;
    }

    int i, j, k;
    Mat srcMat;
    Mat dstMat;
    srcMat = imread(argv[1], 1);

    if (!srcMat.data) {
        printf("Image data does not exist.\n");
        return -1;
    }
    
    string fullname = argv[1];
    if(!atoi(argv[3])) {
        // bilinear interpolation
        myresize_l(srcMat, dstMat, atof(argv[2]));
        //resize(srcMat, dstMat, cv::Size(srcMat.cols*atof(argv[2]),srcMat.rows*atof(argv[2])));
        
        string newname = fullname.substr(0, fullname.find_last_of(".")) + "_" +\
            argv[2] + "_l" + ".jpg";
        imwrite(newname, dstMat);
    }
    
    if(atoi(argv[3])) {
        // bicubic interpolation
        myresize_c(srcMat, dstMat, atof(argv[2]));
        //resize(srcMat, dstMat, cv::Size(srcMat.cols*atof(argv[2]),srcMat.rows*atof(argv[2])), INTER_CUBIC);
        
        string newname = fullname.substr(0, fullname.find_last_of(".")) + "_" +\
            argv[2] + "_c" + ".jpg";
        imwrite(newname, dstMat);
    }
    
    return 0;
}

int myresize_l(Mat &srcMat, Mat &dstMat, double s) {
    // create dstMat by scaling factor s 
    dstMat.create(cvFloor(srcMat.rows*s), cvFloor(srcMat.cols*s), srcMat.type());
        
    int i, j, k;
    // where 'i' for rows, 'j' for cols (in dstMat),
    // 'k' for channels
    for(i = 0; i < dstMat.rows; i++) {
        // find the nearest point sx in srcMat
        double delx = (double)i/s;
        int sx = cvFloor(delx);
        delx -= sx; // where delx is the dist. from sx
        
        for(j = 0; j < dstMat.cols; j++) {  
            // for each (i, j) in dstMat
            // find corresponding 2x2 grid in scrMat [sx,sx+1]x[sy,sy+1]
            double dely = (double)j/s;
            int sy = cvFloor(dely);
            dely -= sy;
            
            // shift points by sx, sy
            // we have (delx, dely) in [0,1]x[0,1]
            for(k = 0; k < dstMat.channels(); k++) {
                uchar p[2][2];
                p[0][0] = srcMat.at<Vec3b>(sx, sy)[k];
                p[1][0] = srcMat.at<Vec3b>(sx+(sx<srcMat.rows-1), sy)[k];
                p[0][1] = srcMat.at<Vec3b>(sx, sy+(sy<srcMat.cols-1))[k];
                p[1][1] = srcMat.at<Vec3b>(sx+(sx<srcMat.rows-1), sy+(sy<srcMat.cols-1))[k];
                dstMat.at<Vec3b>(i, j)[k] = bilinear(p, delx, dely); 
            
            }
        }
    }
    return 1;
}

int myresize_c(Mat &srcMat_o, Mat &dstMat, double s) {
    // create dstMat by scaling factor s 
    dstMat.create(cvFloor(srcMat_o.rows*s), cvFloor(srcMat_o.cols*s), \
        srcMat_o.type());
    
    // expand the boarder for convolution
    Mat srcMat;
    copyMakeBorder(srcMat_o, srcMat, 1, 2, 1, 2, BORDER_REPLICATE);
    
    int i, j, k;
    for(i = 0; i < dstMat.rows; i++) {
        // find the nearest point sx in srcMat
        double delx = (double)i/s;
        int sx = cvFloor(delx);
        delx -= sx; // where delx is the dist. from sx
        
        for(j = 0; j < dstMat.cols; j++) {
            // for each (i, j) in dstMat
            // find corresponding 4x4 grid in scrMat [sx-1,sx+2]x[sy-1,sy+2]
            double dely = (double)j/s;
            int sy = cvFloor(dely);
            dely -= sy;
            
            for(k = 0; k < dstMat.channels(); k++) {
                uchar p[4][4];                
                
                for(int l = 0; l < 4; l++)
                    for(int m = 0; m < 4; m++) 
                        //p[l][m] = srcMat.at<Vec3b>(sx+l-1, sy+m-1)[k];
                        p[l][m] = srcMat.at<Vec3b>(sx+l, sy+m)[k];
                dstMat.at<Vec3b>(i, j)[k] = bicubic(p, delx, dely);
            }
        }
    }
    
    // cut duplicate part
    //dstMat.rows -= 2*s;
    //dstMat.cols -= 2*s;
    
    return 1;
}

int bilinear(uchar p[2][2], double x, double y) {
    // interpolate (x,y) on grid [0,1]x[0,1]
    assert(x>=0 && x<1 && y>=0 && y<1);
    return (1-x)*(1-y)*p[0][0]+x*(1-y)*p[1][0]+\
        (1-x)*y*p[0][1]+x*y*p[1][1];
}

int bicubic(uchar p[4][4], double x, double y){
    // interpolate (x,y) on grid [-1,2]x[-1,2]
    assert(x>=0 && x<1 && y>=0 && y<1);
    
    // use kernel convolution to perform cubic interpolation
    double wx[4], wy[4];
    double a = -0.75f;
    
    wx[0] = ((a*(x+1)-5*a)*(x+1)+8*a)*(x+1)-4*a;
    wx[1] = ((a+2)*x-(a+3))*x*x+1;
    wx[2] = ((a+2)*(1-x)-(a+3))*(1-x)*(1-x)+1;
    wx[3] = ((a*(2-x)-5*a)*(2-x)+8*a)*(2-x)-4*a;
    
    wy[0] = ((a*(y+1)-5*a)*(y+1)+8*a)*(y+1)-4*a;
    wy[1] = ((a+2)*y-(a+3))*y*y+1;
    wy[2] = ((a+2)*(1-y)-(a+3))*(1-y)*(1-y)+1;
    wy[3] = ((a*(2-y)-5*a)*(2-y)+8*a)*(2-y)-4*a;
    
    int conv = cvFloor(((double)p[0][0]*wx[0]+(double)p[1][0]*wx[1]+ \
        (double)p[2][0]*wx[2]+(double)p[3][0]*wx[3])*wy[0]+ \
        ((double)p[0][1]*wx[0]+(double)p[1][1]*wx[1]+ \
        (double)p[2][1]*wx[2]+(double)p[3][1]*wx[3])*wy[1]+ \
        ((double)p[0][2]*wx[0]+(double)p[1][2]*wx[1]+ \
        (double)p[2][2]*wx[2]+(double)p[3][2]*wx[3])*wy[2]+ \
        ((double)p[0][3]*wx[0]+(double)p[1][3]*wx[1]+ \
        (double)p[2][3]*wx[2]+(double)p[3][3]*wx[3])*wy[3]);
    
    // some results could be out of boundary
    if(conv > 255) conv = 255;
    return abs(conv);
}

void printMat(Mat &mat){
    // for check
    for(int k = 0; k < mat.channels(); k++)
    {
        int i = 0;
        for(Mat_<Vec3b>::iterator it = mat.begin<Vec3b>(); \
            it != mat.end<Vec3b>(); \
            it++) {
            printf("%3d ", (*it)[k]);
            
            if(++i % mat.cols == 0)
                printf("\n");
         
        }
        printf("\n");
    }

}
