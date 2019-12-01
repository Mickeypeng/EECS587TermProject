#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace cv::xfeatures2d;

int main(int argc, char** argv ){
    if ( argc != 3 ){
        printf("usage: serial <Image_Path_Left> <Image_Path_Right>\n");
        return -1;
    }
    Mat left_img;
    left_img = imread( argv[1], 1 );
    Mat right_img;
    right_img = imread( argv[2], 1 );
    if ( !left_img.data || !right_img.data){
        printf("No image data \n");
        return -1;
    }

    Ptr<SIFT> detector = SIFT::create( 400 );
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( left_img, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( right_img, noArray(), keypoints2, descriptors2 );

    Mat nlimg;
    drawKeypoints(left_img, keypoints1, nlimg);
    
    imwrite("./keypoints_left.jpg", nlimg);
    
    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", nlimg);
    // waitKey(0);
    return 0;
}