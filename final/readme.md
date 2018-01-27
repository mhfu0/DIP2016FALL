## Offspring Prediction
The repository contains source codes for offspring prediction.

### Description
Offspring prediction using image morphing. Process includes facial landmark detection, triangulation and affine transformation. More details can be found in the report.

We use dlib's facial landmark detection, which is implemented with ensemble of regression trees, to extract 68 facial landmarks for further purposes.

&ast; dlib predictor can be found in http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

### Requirements
`cmake 2.8.12`
`opencv 2.4.13` 
`dlib 19.2` 

### Usage
`$ ./face_landmark_detection <img1_path> <img2_path> <alpha>`
<alpha>: ratio between two source images

### Reference
* http://www.learnopencv.com/face-morph-using-opencv-cpp-python
* http://dlib.net/face_landmark_detection_ex.cpp.html
* https://pdfs.semanticscholar.org/d78b/6a5b0dcaa81b1faea5fb0000045a62513567.pdf
