//** Face Morphing implementation with opencv2 and dlib
//** Source: www.learnopencv.com/face-morph-using-opencv-cpp-python
//** Credit: Satya Mallick 
//** modified and merged

#include <dlib/opencv.h>  
#include <opencv2/opencv.hpp>  

#include <dlib/image_processing/frontal_face_detector.h>  
#include <dlib/image_processing/render_face_detections.h>  
#include <dlib/image_processing.h>  
#include <dlib/gui_widgets.h>

#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

using namespace std;
using namespace dlib;
using namespace cv;

void help() {
    fprintf(stderr, "Face Landmark Detection and Morphing implementation with opencv2 and dlib.\n");
    fprintf(stderr, "Source: www.learnopencv.com/face-morph-using-opencv-cpp-python\n");
    fprintf(stderr, "Credit: Satya Mallick.\n\n");
}

static void drawpoint(Mat& img, Point2f fp, Scalar color)
{
    circle(img, fp, 1, color, CV_FILLED, CV_AA, 0);
}

// Apply affine transform calculated using srcTri and dstTri to src
void applyAffineTransform(Mat &warpImage, Mat &src, std::vector<Point2f> &srcTri, std::vector<Point2f> &dstTri)
{
    // Given a pair of triangles, find the affine transform.
    Mat warpMat = getAffineTransform( srcTri, dstTri );
    // Apply the Affine Transform just found to the src image
    warpAffine( src, warpImage, warpMat, warpImage.size(), INTER_LINEAR, BORDER_REFLECT_101);
}

// Warps and alpha blends triangular regions from img1 and img2 to img
void morphTriangle(Mat &img1, Mat &img2, Mat &img, std::vector<Point2f> &t1, std::vector<Point2f> &t2, std::vector<Point2f> &t, double alpha)
{
    // Find bounding rectangle for each triangle
    Rect r = boundingRect(t);
    Rect r1 = boundingRect(t1);
    Rect r2 = boundingRect(t2);
    
    // Offset points by left top corner of the respective rectangles
    std::vector<Point2f> t1Rect, t2Rect, tRect;
    std::vector<Point> tRectInt;
    for(int i = 0; i < 3; i++)
    {
        tRect.push_back( Point2f( t[i].x - r.x, t[i].y -  r.y) );
        tRectInt.push_back( Point(t[i].x - r.x, t[i].y - r.y) ); // for fillConvexPoly
        
        t1Rect.push_back( Point2f( t1[i].x - r1.x, t1[i].y -  r1.y) );
        t2Rect.push_back( Point2f( t2[i].x - r2.x, t2[i].y - r2.y) );
    }

    // Get mask by filling triangle
    Mat mask = Mat::zeros(r.height, r.width, CV_32FC3);
    fillConvexPoly(mask, tRectInt, Scalar(1.0, 1.0, 1.0), 16, 0);
    // Apply warpImage to small rectangular patches
    Mat img1Rect, img2Rect;
    
    img1(r1).copyTo(img1Rect);
    img2(r2).copyTo(img2Rect);
    
    Mat warpImage1 = Mat::zeros(r.height, r.width, img1Rect.type());
    Mat warpImage2 = Mat::zeros(r.height, r.width, img2Rect.type());
    
    applyAffineTransform(warpImage1, img1Rect, t1Rect, tRect);
    applyAffineTransform(warpImage2, img2Rect, t2Rect, tRect);
    
    // Alpha blend rectangular patches
    Mat imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2;
    // Copy triangular region of the rectangular patch to the output image
    
    multiply(imgRect, mask, imgRect);
    multiply(img(r), Scalar(1.0,1.0,1.0) - mask, img(r));
    img(r) = img(r) + imgRect;
    
    
}

int main(int argc, char **argv) {
    help();
    
    // Usage
    if(argc != 4) {
        fprintf(stderr, "Invalid argument.\n");
        fprintf(stderr, "Usage: ./face_landmark_detection <img1_path> <img2_path> <alpha>\n");
        return -1;
    }
    
    //----- Face Landmark Detection -----//
    // Load face detection and pose estimation models 
    // TODO: bottleneck -- some configuration need
    frontal_face_detector detector = get_frontal_face_detector();  
    shape_predictor pose_model;  
    deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;  

    Mat img1, img2;
    img1 = imread(argv[1], 1);        
    img2 = imread(argv[2], 1);
    resize(img1, img1, img2.size());
    
    // copies of original images
    Mat img1_orig = img1.clone();
    Mat img2_orig = img2.clone();
            
    cv_image<bgr_pixel> cimg1(img1);
    cv_image<bgr_pixel> cimg2(img2);
    
    // Detect faces in images
    // Actually only one face is to be used
    std::vector<dlib::rectangle> faces1 = detector(cimg1);  
    std::vector<dlib::rectangle> faces2 = detector(cimg2); 
    
    // Find the pose of each face
    std::vector<full_object_detection> shapes1; 
    std::vector<full_object_detection> shapes2; 
    for(unsigned long i = 0; i < faces1.size(); ++i)  
        shapes1.push_back(pose_model(cimg1, faces1[i]));  
    for(unsigned long i = 0; i < faces2.size(); ++i)  
        shapes2.push_back(pose_model(cimg2, faces2[i])); 
    
    // Draw feature points on images
    if(!shapes1.empty()) for(int i = 0; i < 68; i++) {
        drawpoint(img1, Point(shapes1[0].part(i).x(), 
            shapes1[0].part(i).y()), cv::Scalar(0, 0, 255)); 
    }  

    if(!shapes2.empty()) for(int i = 0; i < 68; i++) {
        drawpoint(img2, Point(shapes2[0].part(i).x(), 
            shapes2[0].part(i).y()), cv::Scalar(0, 0, 255)); 
    }  

    namedWindow("Face1", WINDOW_AUTOSIZE);
    namedWindow("Face2", WINDOW_AUTOSIZE);
    imshow("Face1", img1);
    imshow("Face2", img2);
    waitKey(0);

    //----- Delaunay Triangulation -----//
    // Reset images
    img1 = img1_orig.clone();
    img2 = img2_orig.clone();
    
    // Set region of calculation of img1
    Size size1 = img1.size();
    Rect rect1(0, 0, size1.width, size1.height);
    Subdiv2D subdiv1(rect1);
    std::vector<Point2f> points1;
    
    if(!shapes1.empty()) {
        // Add 68 facial feature points
        for(int i = 0; i < 68; i++) 
            points1.push_back(Point(shapes1[0].part(i).x(), shapes1[0].part(i).y()));
        
        // Add 8 border points
        points1.push_back(Point(0,0));
        points1.push_back(Point(size1.width/2,0));
        points1.push_back(Point(size1.width-1,0));
        points1.push_back(Point(0,size1.height/2));
        points1.push_back(Point(size1.width/2,size1.height/2));
        points1.push_back(Point(0,size1.height-1));
        points1.push_back(Point(size1.width/2,size1.height-1));
        points1.push_back(Point(size1.width-1,size1.height-1));
    }
    for(std::vector<Point2f>::iterator it = points1.begin(); it != points1.end(); it++) {
        subdiv1.insert(*it);
    }
    
    // Perform Delaunay triangulation and get the list of point indice
    std::vector<Point2f> pt(3);
    std::vector<std::array<int, 3>> triindexlist;
    // 
    
    std::vector<Vec6f> triangleList1;
    subdiv1.getTriangleList(triangleList1);
    for( size_t i = 0; i < triangleList1.size(); i++ )
    {
        Vec6f t = triangleList1[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
        // Consider the points in the region only
        if ( rect1.contains(pt[0]) && rect1.contains(pt[1]) && rect1.contains(pt[2]))
        {
            line(img1, pt[0], pt[1], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
            line(img1, pt[1], pt[2], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
            line(img1, pt[2], pt[0], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
            
            // Find the corresponding indice of the points
            // TODO: Kind of brute force search here -- need refinement
            std::array<int, 3> idx;
            idx[0]=-1;idx[1]=-1;idx[2]=-1;
            for(std::vector<Point2f>::iterator it = points1.begin(); it != points1.end(); it++)
                if((*it) == pt[0])
                    idx[0] = it - points1.begin();
            for(std::vector<Point2f>::iterator it = points1.begin(); it != points1.end(); it++)
                if((*it) == pt[1]) 
                    idx[1] = it - points1.begin();           
            for(std::vector<Point2f>::iterator it = points1.begin(); it != points1.end(); it++)
                if((*it) == pt[2]) 
                    idx[2] = it - points1.begin();
            assert(idx[0]!=-1&&idx[1]!=-1&&idx[2]!=-1);
            triindexlist.push_back(idx);
            
        }
    }
/*
    // Print the list of triangle point indices
    for(std::vector<std::array<int, 3>>::iterator it = triindexlist.begin(); \
        it != triindexlist.end(); it++) {
            cout << (*it)[0] << " " << (*it)[1] << " " << (*it)[2] << endl;
    }
*/
    // Set region of calculation of img2
    Size size2 = img2.size();
    Rect rect2(0, 0, size2.width, size2.height);
    Subdiv2D subdiv2(rect2);
    std::vector<Point2f> points2;
    
    if(!shapes2.empty()) {
        // Add 68 facial feature points
        for(int i = 0; i < 68; i++)
            points2.push_back(Point(shapes2[0].part(i).x(), shapes2[0].part(i).y()));
        
        // Add 8 border points
        points2.push_back(Point(0,0));
        points2.push_back(Point(size2.width/2,0));
        points2.push_back(Point(size2.width-1,0));
        points2.push_back(Point(0,size2.height/2));
        points2.push_back(Point(size2.width/2,size2.height/2));
        points2.push_back(Point(0,size2.height-1));
        points2.push_back(Point(size2.width/2,size2.height-1));
        points2.push_back(Point(size2.width-1,size2.height-1));
    }
    for(std::vector<Point2f>::iterator it = points2.begin(); it != points2.end(); it++) {
        subdiv2.insert(*it);
    }
    
    std::vector<Vec6f> triangleList2;
    subdiv2.getTriangleList(triangleList2);
    
    for( size_t i = 0; i < triangleList2.size(); i++ )
    {
        Vec6f t = triangleList2[i];
        pt[0] = Point(cvRound(t[0]), cvRound(t[1]));
        pt[1] = Point(cvRound(t[2]), cvRound(t[3]));
        pt[2] = Point(cvRound(t[4]), cvRound(t[5]));
         
        if ( rect2.contains(pt[0]) && rect2.contains(pt[1]) && rect2.contains(pt[2]))
        {
            line(img2, pt[0], pt[1], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
            line(img2, pt[1], pt[2], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
            line(img2, pt[2], pt[0], cv::Scalar(0, 255, 0), 1, CV_AA, 0);
        }
    }


    // Display it all on the screen
    imshow("Face1", img1);
    imshow("Face2", img2);
    
    waitKey(0);

    //----- Morphing -----//
    // alpha: linear combination factor
    double alpha = atof(argv[3]);
    
    // Convert to floating-point:
    // We do floating-point calculation during linear combination
    img1_orig.convertTo(img1, CV_32F);
    img2_orig.convertTo(img2, CV_32F);
    
    // imgMorph: the morphed image
    Mat imgMorph = Mat::zeros(img1.size(), CV_32FC3);
    std::vector<Point2f> points;



    // Find the corresponding position in imgMorph
    for(int i = 0; i < points1.size(); i++)
    {
        float x, y;
        x = (1-alpha) * points1[i].x + alpha * points2[i].x;
        y = (1-alpha) * points1[i].y + alpha * points2[i].y;
        
        points.push_back(Point2f(x,y));
    }  
    
    // Note that we only have one copy of the list of the point indice
    // since the feature points found by get_frontal_face_detector()
    // are always in the same order
    for(std::vector<std::array<int, 3>>::iterator it = triindexlist.begin(); \
        it != triindexlist.end(); it++) {
        
        // Triangles (t1:img1|t2:img2|t:imgMorph)
        std::vector<Point2f> t1, t2, t;
        int x, y, z;
        x=(*it)[0]; y=(*it)[1]; z=(*it)[2];
        
        // Triangle corners for image 1.
        t1.push_back( points1[x] );
        t1.push_back( points1[y] );
        t1.push_back( points1[z] );
        
        // Triangle corners for image 2.
        t2.push_back( points2[x] );
        t2.push_back( points2[y] );
        t2.push_back( points2[z] );
        
        // Triangle corners for morphed image.
        t.push_back( points[x] );
        t.push_back( points[y] );
        t.push_back( points[z] );
        
        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha);
    }
    
    namedWindow("Morphed Face", WINDOW_AUTOSIZE);
    imshow("Morphed Face", imgMorph / 255.0);
    waitKey(0);

    return 0;
}  
