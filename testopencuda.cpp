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

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

extern "C" int mulMatrix();

int main(){
    Mat left_img;
    left_img = imread(argv[1], 0);
    Mat right_img;
    right_img = imread(argv[2], 0);
    if ( !left_img.data || !right_img.data){
        cout << "No image data" << endl;
        return -1;
    }
    cout << "into cuda" << endl;
    mulMatrix();
    cout << "out cuda" << endl;
    return 0;
}
