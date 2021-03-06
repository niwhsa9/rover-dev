// The "Square Detector" program.
// It loads several images sequentially and tries to find squares in
// each image
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ximgproc.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <chrono>

using namespace cv;
using namespace std;
using namespace cv::ximgproc;
using namespace std::chrono;

int thresh = 50;
const char* wndname = "Square Detection Demo";

// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
static double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}
// returns sequence of squares detected on the image.
static void findSquares(const Mat& image, vector<vector<Point> >& squares) {   
    
    squares.clear();
    Mat pyr, timg, gray0(image.size(), CV_8U), gray, dst;
    // down-scale and upscale the image to filter out the noise
    auto start = high_resolution_clock::now();
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;
    // find squares in every color plane of the image
    for(int c = 0; c < 3; c++) {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);
        // Apply Canny. Take the upper threshold from slider and set the lower to 0 (which forces edges merging)
        
        Canny(gray0, gray, 500, 500*2, 5);
        imshow("Canny", gray);      
        // dilate canny output to remove potential between edge segments
        dilate(gray, gray, Mat(), Point(-1,-1));
        imshow("Dialated", gray);      

        // find contours and store them all as a list
        findContours(gray, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);
        vector<Point> approx;
        drawContours(image, contours, -1, Scalar(255, 0, 0));
        // test each contour
        for(size_t i = 0; i < contours.size(); i++) {
            // approximate contour with accuracy proportional
            // to the contour perimeter
            approxPolyDP(contours[i], approx, arcLength(contours[i], true)*0.08, true);
            // square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours) and be convex.
            // Note: absolute value of an area is used because area may be positive or negative - in accordance with the contour orientation
            if(approx.size() == 4 && fabs(contourArea(approx)) > 400 && isContourConvex(approx)) {
                double maxCosine = 0;
                for( int j = 2; j < 5; j++ ) {
                    // find the maximum cosine of the angle between joint edges
                    double cosine = fabs(angle(approx[j%4], approx[j-2], approx[j-1]));
                    maxCosine = MAX(maxCosine, cosine);
                }
                // if cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence
                if(maxCosine < 0.25)
                    squares.push_back(approx);
            }
        }
        
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop-start);
    cout << duration.count() << endl;
}

int main(int argc, char** argv) {
    string filename, filenameTemp;
    for (int i = 120; i < 212; ++i) {
        if (i < 10) {
            string iString = to_string(i);
            filenameTemp = "000" + iString + ".jpg";
            //filename = samples::findFile(filenameTemp);
            Mat image = imread(filenameTemp, IMREAD_COLOR);
            if(image.empty()) {
                cout << "Couldn't load " << filename << endl;
                continue;
            }
            vector<vector<Point>> squares;
            findSquares(image, squares);
            polylines(image, squares, true, Scalar(0, 255, 0), 3, LINE_AA);
            imshow(wndname, image); 
            int c = waitKey(0);
        }
        if (i >= 10 && i < 100) {
            string iString = to_string(i);
            filenameTemp = "00" + iString + ".jpg";
            //filename = cv::samples::findFile(filenameTemp);
            Mat image = imread(filenameTemp, IMREAD_COLOR);
            if(image.empty()) {
                cout << "Couldn't load " << filename << endl;
                continue;
            }
            vector<vector<Point>> squares;
            findSquares(image, squares);
            polylines(image, squares, true, Scalar(0, 255, 0), 3, LINE_AA);
            imshow(wndname, image);
            int c = waitKey(0);
        }
        if (i >= 100 && i < 212) {
            
            string iString = to_string(i);
            filenameTemp = "0" + iString + ".jpg";
            //filename = cv::samples::findFile(filenameTemp);  
            Mat image = imread(filenameTemp, IMREAD_COLOR);
            vector<vector<Point>> squares;
            
            findSquares(image, squares);
            polylines(image, squares, true, Scalar(0, 255, 0), 3, LINE_AA);
            auto stop = high_resolution_clock::now();
            imshow(wndname, image);
            int c = waitKey(0);
        }
    }
    return 0;

    //g++ $(pkg-config --cflags --libs opencv4) -std=c++11 -o squareDetection squareDetection.cpp
}