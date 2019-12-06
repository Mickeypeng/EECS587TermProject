#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <limits>
#include <fstream>
#include <stdlib.h> 
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#pragma comment ( lib,"opencv_core320.lib")
#pragma comment ( lib,"opencv_highgui320.lib")
#pragma comment ( lib,"opencv_calib3d320.lib")
#pragma comment ( lib,"opencv_imgcodecs320.lib")
#pragma comment ( lib,"opencv_imgproc320.lib")
#pragma comment ( lib,"opencv_cudaimgproc320.lib")
#pragma comment ( lib,"opencv_cudaarithm320.lib")
#pragma comment ( lib,"cudart.lib")
#pragma comment ( lib,"mulMatrix.lib")

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

extern "C" int mulMatrix();

int main(){
    Mat left_img;
    left_img = imread("bbb_left.jpg", 0);
    Mat right_img;
    right_img = imread("bbb_right.jpg", 0);
    if ( !left_img.data || !right_img.data){
        cout << "No image data" << endl;
        return -1;
    }
    cout << "into cuda" << endl;
    mulMatrix();
    cout << "out cuda" << endl;
    return 0;
}
