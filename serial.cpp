#include <iostream>
#include <vector>
#include <math.h>
#include <numeric>
#include <limits>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

void writeMatFile(vector<vector<double> >& v, string fileName){
    string output;
    for(int i = 0; i < v.size(); i++){
        for(int j = 0; j < v[0].size(); j++){
            output += to_string(v[i][j]) + " ";
        }
        output += "\n";
    }
    ofstream myfile;
    myfile.open (fileName);
    myfile << output;
    myfile.close();
    return;
}

double calDis(vector<double>& d1, vector<double>& d2){
    double result = 0;
    for(int i = 0; i < d1.size(); i++){
        result += (d1[i] - d2[i])*(d1[i] - d2[i]);
    }
    result = sqrt(result);
    // cout << result << endl;
    return result;
}

int findMatch(vector<double>& dis, double thres){
    double firstmin = numeric_limits<double>::max();
    double secondmin = numeric_limits<double>::max();
    int firstindex = -1;
    for(int i = 0; i < dis.size(); i++){
        if(dis[i] < firstmin){
            secondmin = firstmin;
            firstmin = dis[i];
            firstindex = i;
        }else if(dis[i] < secondmin && dis[i] != firstmin){
            secondmin = dis[i];
        }else{
            continue;
        }
    }
    // cout << "first: " << firstmin << endl;
    // cout << "second: " << secondmin << endl;
    if(firstmin/secondmin< thres){
        // cout << firstindex << endl;
        return firstindex;
    }
    return -1;
}

void findPair(vector<vector<double> > & distance, vector<DMatch>& matches){
    double threshold = 0.7;
    for(int i = 0; i < distance.size(); i++){
        int pind = findMatch(distance[i], threshold);
        if(pind != -1){
            DMatch nmatch = DMatch(i, pind, distance[i][pind]);
            matches.push_back(nmatch);
        }
    }
}

void calMeanVar(vector<double>& v, double& mean, double& var){
    double sum = accumulate(v.begin(), v.end(), 0.0);
    mean =  sum / v.size();
    double sq_sum = inner_product(v.begin(), v.end(), v.begin(), 0.0);
    var = sqrt(sq_sum / v.size() - mean * mean);
    return; 
}

void normalize(vector<double>& v){
    double mean = 0;
    double var = 0;
    calMeanVar(v, mean, var);
    for(int i = 0; i < v.size(); i++){
        v[i] = (v[i] - mean) / var;
    }
    return;
}

int main(int argc, char** argv ){
    if ( argc != 3 ){
        cout << "usage: serial <Image_Path_Left> <Image_Path_Right>" << endl;
        return -1;
    }
    Mat left_img;
    left_img = imread(argv[1], 0);
    Mat right_img;
    right_img = imread(argv[2], 0);
    if ( !left_img.data || !right_img.data){
        cout << "No image data" << endl;
        return -1;
    }

    int nums_des = 400;

    Ptr<SIFT> detector = SIFT::create(nums_des);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( left_img, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( right_img, noArray(), keypoints2, descriptors2 );

    // cout << descriptors1.type() << endl;

    vector<vector<double> > des1;
    vector<vector<double> > des2;

    if(descriptors1.rows < nums_des || descriptors2.rows < nums_des ){
        cout << "Not " << nums_des << " descriptors, have total " << descriptors1.rows << endl;
        return -1;
    }

    for (int i = 0; i < nums_des; i++) {
        vector<double> tmp1;
        vector<double> tmp2;
        for(int j = 0; j < descriptors1.cols; j++){
            tmp1.push_back(double(*(descriptors1.ptr<float>(i) + j)));
            tmp2.push_back(double(*(descriptors2.ptr<float>(i) + j)));
        }
        des1.push_back(tmp1);
        des2.push_back(tmp2);
    }

    // cout << des1.size() << endl;
    // for(int i = 0; i < 128; i++){
    //     cout << des1[0][i] << endl;
    // }
    // normalize(des1[0]);
    // for(int i = 0; i < 128; i++){
    //     cout << des1[0][i] << endl;
    // }

    writeMatFile(des1, "img1_400.txt");
    writeMatFile(des2, "img2_400.txt");

    vector<vector<double> > distance(nums_des, vector<double>(nums_des, 0));
    for(int i = 0; i < nums_des; i++){
        for(int j = 0; j < nums_des; j++){
            normalize(des1[i]);
            normalize(des2[j]);
            distance[i][j] = calDis(des1[i], des2[j]);
            // distance[i][j] = norm(des1[i], des2[j], NORM_L2, noArray() );
        }
    }

    writeMatFile(distance, "400_distance.txt");

    vector<DMatch> matches;
    findPair(distance, matches);
    cout << matches.size() << endl;
    Mat outImg;
    drawMatches(left_img, keypoints1, right_img, keypoints2, matches, outImg);



    // Mat nlimg;
    // drawKeypoints(left_img, keypoints1, nlimg);
    
    // imwrite("./keypoints_left.jpg", nlimg);
    
    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", outImg);
    waitKey(0);
    return 0;
}