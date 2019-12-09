#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <chrono>
#include "stitch.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv){
    if ( argc != 3 ){
        cout << "usage: pano <path> <image_list.txt>" << endl;
        return -1;
    }
    string path = argv[1];
    string filename = argv[2];

    ifstream myFile;
    myFile.open(path+filename);
    if(!myFile.is_open()){
        cout << "cannot open list file" << endl;
        return -1; 
    }
    string line;
    vector<Mat> imageData;
    while(!myFile.eof()){
        getline(myFile, line);
        imageData.push_back(imread(path + line, 1));
        // cout << line << '\n';
    }
    myFile.close();

    // vector<int> imageIndex(imageData.size());
    // iota(imageIndex.begin(), imageIndex.end(), 0);

    Mat outimg;
    int count = imageData.size();
    while(count > 1){
        cout << count << endl;
        int max_inliers = -1;
        vector<vector<double> > besth;
        int left = -1;
        int right = -1;

        auto start = high_resolution_clock::now();
        for(int i = 0; i < imageData.size() - 1; i++){
            for(int j = i + 1; j < imageData.size(); j++){
                vector<vector<double> > h;
                int matchNum = 0;
                // cout << "run stitch" << endl;
                int inliersNum = runStitch(imageData[i], imageData[j], h, matchNum);
                // cout << "finish stitch" << endl;
                if(inliersNum == -1){
                    continue;
                }
                // if (inliersNum > 5.9 + 0.22*double(matchNum)){
                //     cout << i << j <<endl;

                //     besth = h;
                //     left = i;
                //     right = j;
                //     break;
                // }
                if(inliersNum > max_inliers){
                    max_inliers = inliersNum;
                    besth = h;
                    left = i;
                    right = j;
                }
            }
            // if(left != -1){
            //     break;
            // }
        }
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop - start); 
        cout << "total stitch time: " << duration.count() << endl; 
        // cout << "run warp" << endl;
        // cout << left << right <<endl;
        outimg = warpImages(imageData[left], imageData[right], besth);
        // cout << "finish warp" << endl;
        // namedWindow("Display Image", WINDOW_AUTOSIZE );
        // imshow("Display Image", outimg);
        // waitKey(0);
        imageData.erase(imageData.begin() + right);
        imageData.erase(imageData.begin() + left);
        imageData.push_back(outimg);
        count--;
    }
    cout << "finish" << endl;
    // namedWindow("Display Image", WINDOW_AUTOSIZE );
    // imshow("Display Image", outimg);
    // waitKey(0);
    imwrite("./pano.jpg", outimg);
    return 0;
}