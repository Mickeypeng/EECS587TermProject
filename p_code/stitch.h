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
#pragma comment ( lib,"MulWithCuda.lib")


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
    if(firstmin/secondmin< thres){
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

void findPos(vector<DMatch>& matches, vector<KeyPoint>& k1, vector<KeyPoint>& k2, vector<vector<float> >& coords){
    for(int i = 0; i < matches.size(); i++){
        vector<float> c;
        c.push_back(k1[matches[i].queryIdx].pt.x);
        c.push_back(k1[matches[i].queryIdx].pt.y);
        c.push_back(k2[matches[i].trainIdx].pt.x);
        c.push_back(k2[matches[i].trainIdx].pt.y);
        coords.push_back(c);
    }
    return;
}

void findRandomIdx(vector<int>& idx, int m, int K){
    for(int i = 0; i < K; i++){
        idx.push_back(rand() % m);
    }
    return;
}

void findSubset(vector<vector<float> >& coords, int K, vector<vector<float> >& subset){
    vector<int> idx;
    findRandomIdx(idx, coords.size() - 1, K);
    for(int i = 0; i < idx.size(); i++){
        subset.push_back(coords[idx[i]]);
    }
    return;
}

void fitModel(vector<vector<float> >& coords, vector<vector<double> >& h){
    Mat w, u, vt;
    Mat A = Mat::zeros(2*coords.size(), 9, CV_32F);
    for(int i = 0; i < coords.size(); i++){
        A.at<float>((2*i), 3) = -coords[i][0];
        A.at<float>((2*i), 4) = -coords[i][1];
        A.at<float>((2*i), 5) = -1;
        A.at<float>((2*i), 6) = coords[i][3] * coords[i][0];
        A.at<float>((2*i), 7) = coords[i][3] * coords[i][1];
        A.at<float>((2*i), 8) = coords[i][3];
        A.at<float>((2*i+1), 0) = coords[i][0];
        A.at<float>((2*i+1), 1)= coords[i][1];
        A.at<float>((2*i+1), 2) = 1;
        A.at<float>((2*i+1), 6)= -coords[i][2] * coords[i][0];
        A.at<float>((2*i+1), 7) = -coords[i][2] * coords[i][1];
        A.at<float>((2*i+1), 8) = -coords[i][2];
    }
    SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);
    double norm = 0;
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            h[i][j] = vt.at<float>(8, 3*i+j);
            norm += h[i][j]*h[i][j];
        }
    }
    norm = sqrt(norm);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            h[i][j] = h[i][j] / norm;
        }
    }
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            h[i][j] = h[i][j] / h[2][2];
        }
    }
    return;
}

double pointDis(vector<float>& p, vector<vector<double> >& h){
    vector<float> ep1(3, 0);
    ep1[0] = h[0][0]*p[0] + h[0][1]*p[1] + h[0][2];
    ep1[1] = h[1][0]*p[0] + h[1][1]*p[1] + h[1][2];
    ep1[2] = h[2][0]*p[0] + h[2][1]*p[1] + h[2][2];
    ep1[0] /= ep1[2];
    ep1[1] /= ep1[2];
    return sqrt((ep1[0] - p[2])*(ep1[0] - p[2])+(ep1[1] - p[3])*(ep1[1] - p[3]));
}

vector<vector<double> > ransac(vector<vector<float> >& coords, int& bestcount){
    int numTrials = 100;
    float thres = 5.0;
    int K = 4;
    vector<vector<float> > bestinliers;
    double besterror;
    for(int i = 0; i < numTrials; i++){
        vector<vector<float> > subset;
        findSubset(coords, K, subset);
        vector<vector<double> > h(3, vector<double>(3, 0));
        fitModel(subset, h);
        vector<vector<float> > inliers;
        double allerror = 0;
        for(int j = 0; j < coords.size(); j++){
            double dis = pointDis(coords[j], h);
            if(dis < thres){
                inliers.push_back(coords[j]);
                allerror += dis;
            }
        }
        if(inliers.size() > bestinliers.size()){
            bestinliers = inliers;
            besterror = allerror;
        }
    }
    vector<vector<double> > besth(3, vector<double>(3, 0));
    fitModel(bestinliers, besth);
    bestcount = bestinliers.size();
    // cout << bestinliers.size() << endl;
    // cout << "error: " << besterror / bestinliers.size() << endl;
    return besth;
}

Mat warpImages(Mat left_img, Mat right_img, vector<vector<double> >& h){
    Mat trans(3, 3, CV_32F);
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < 3; j++){
            trans.at<float>(i, j) = h[i][j];
        }
    }
    double h1 = left_img.rows;
    double w1 = left_img.cols;
    double h2 = right_img.rows;
    double w2 = right_img.cols;
    Mat corner1 = (Mat_<float>(3,4) << 0, 0, w1, w1, 0, h1, h1, 0, 1, 1, 1, 1);
    Mat corner2 = (Mat_<float>(2,4) << 0, 0, w2, w2, 0, h2, h2, 0);
    Mat sc1 = trans*corner1;
    for(int i = 0; i < 2; i++){
        for(int j = 0; j < 4; j++){
            sc1.at<float>(i, j) = sc1.at<float>(i, j) / sc1.at<float>(2, j);
        }
    }
    // cout << "1" << endl;
    float xmin, ymin = numeric_limits<float>::max();
    float xmax, ymax = numeric_limits<float>::min();
    for(int i = 0; i < 4; i++){
        xmin = std::min(xmin, sc1.at<float>(0, i));
        xmin = std::min(xmin, corner2.at<float>(0, i));
        ymin = std::min(ymin, sc1.at<float>(1, i));
        ymin = std::min(ymin, corner2.at<float>(1, i));
        xmax = std::max(xmax, sc1.at<float>(0, i));
        xmax = std::max(xmax, corner2.at<float>(0, i));
        ymax = std::max(ymax, sc1.at<float>(1, i));
        ymax = std::max(ymax, corner2.at<float>(1, i));
    }
    // cout << "2" << endl;
    xmin -= 0.5;
    ymin -= 0.5;
    xmax += 0.5;
    ymax += 0.5;
    xmin = int(xmin);
    ymin = int(ymin);
    xmax = int(xmax);
    ymax = int(ymax);
    Mat ht = (Mat_<float>(3,3) << 1,0,-xmin,0,1,-ymin,0,0,1);
    Mat outimg;
    warpPerspective(left_img, outimg, ht*trans, Size_<int>(xmax-xmin, ymax-ymin));
    // cout << "3"<< endl;
    // cout << outimg.rows << "\t" << outimg.cols << endl;
    // cout << right_img.rows << "\t" << right_img.cols << endl;

    for (int i = -ymin; i < -ymin + h2; i++){
        for (int j = -xmin; j < -xmin + w2; j++){
            Vec3b outPix= outimg.at<Vec3b>(i, j);
            Vec3b rightPix = right_img.at<Vec3b>(i + ymin, j + xmin);
            if (outPix.val[0] == 0 && outPix.val[1] == 0 && outPix.val[2] == 0){
                outimg.at<Vec3b>(i, j)[0] = rightPix.val[0]; 
                outimg.at<Vec3b>(i, j)[1] = rightPix.val[1]; 
                outimg.at<Vec3b>(i, j)[2] = rightPix.val[2];
            }else if(rightPix.val[0] == 0 && rightPix.val[1] == 0 && rightPix.val[2] == 0){
                continue;
            }else{
                outimg.at<Vec3b>(i, j)[0] = (int(outPix.val[0]) + int(rightPix.val[0]))/2;
                outimg.at<Vec3b>(i, j)[1] = (int(outPix.val[1]) + int(rightPix.val[1]))/2;
                outimg.at<Vec3b>(i, j)[2] = (int(outPix.val[2]) + int(rightPix.val[2]))/2;
            }
        }   
    }
       
    return outimg;
}

extern "C" void MulWithCuda(double* A, double* B, int* indC, int featureNum, double thres);

int runStitch(Mat left_img, Mat right_img, vector<vector<double> >& h, int& matchNum){
    if ( !left_img.data || !right_img.data){
        cout << "No image data" << endl;
    }

    int nums_des = 400;
    // cout << "1" << endl;
    Ptr<SIFT> detector = SIFT::create(nums_des);
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    detector->detectAndCompute( left_img, noArray(), keypoints1, descriptors1 );
    detector->detectAndCompute( right_img, noArray(), keypoints2, descriptors2 );
    // cout << "2" << endl;
    // vector<vector<double> > des1;
    // vector<vector<double> > des2;

    if(descriptors1.rows < nums_des || descriptors2.rows < nums_des ){
        cout << "Not " << nums_des << " descriptors, have total " << descriptors1.rows << endl;
    }
    if(descriptors1.rows != descriptors2.rows){
        cout << "Dimension not equal." << endl;
    }
    // cout << "3" << endl;
    // for (int i = 0; i < nums_des; i++) {
    //     vector<double> tmp1;
    //     vector<double> tmp2;
    //     for(int j = 0; j < descriptors1.cols; j++){
    //         tmp1.push_back(double(*(descriptors1.ptr<float>(i) + j)));
    //         tmp2.push_back(double(*(descriptors2.ptr<float>(i) + j)));
    //     }
    //     des1.push_back(tmp1);
    //     des2.push_back(tmp2);
    // }
    // cout << "4" << endl;
    // vector<vector<double> > distance(nums_des, vector<double>(nums_des, 0));
    // for(int i = 0; i < nums_des; i++){
    //     for(int j = 0; j < nums_des; j++){
    //         normalize(des1[i]);
    //         normalize(des2[j]);
    //         distance[i][j] = calDis(des1[i], des2[j]);
    //     }
    // }
    // cout << "5" << endl;
    // vector<DMatch> matches;
    // findPair(distance, matches);
    double* des1 = new double[nums_des * 128];
    double* des2 = new double[nums_des * 128];
    for(int i = 0; i < nums_des; i++){
        for(int j = 0; j < 128; j++){
            des1[i*nums_des + j] = double(*(descriptors1.ptr<float>(i) + j));
            des2[i*nums_des + j] = double(*(descriptors2.ptr<float>(i) + j));
        }
    }
    double thres = 0.7;
    int* indC = new int[nums_des];
    MulWithCuda(des1, des2, indC, nums_des, thres);
    vector<DMatch> matches;
    for(int i = 0; i < nums_des; i++){
        if(indC[i] != -1){
            DMatch nmatch = DMatch(i, indC[i], 0);
            matches.push_back(nmatch);
        }
    }
    matchNum = matches.size();
    // cout << "6" << endl;
    vector<vector<float> > coords;
    findPos(matches, keypoints1, keypoints2, coords);
    int bestcount = 0;
    // cout << "7" << endl;
    // cout << coords.size() << endl;
    if(coords.size() < 4){
        return -1;
    }
    h = ransac(coords, bestcount);
    // cout << "8" << endl;
    return bestcount;
}