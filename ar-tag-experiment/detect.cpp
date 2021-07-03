#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ximgproc.hpp"

using namespace std;
using namespace cv;

const string imdir = "data/";

Mat adjustContrast(Mat &image, double alpha) {
    Mat new_image = Mat::zeros( image.size(), image.type() );
    int beta = 0;       /*< Simple brightness control */
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*image.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
    return new_image;
}

int main() {
    for(int i = 111; i <= 211; i++) {
        // Image load
        cout << "Showing image " << i << endl;
        stringstream num;
        num << setfill('0') << setw(4) << i;
        string imagePath = imdir + num.str() + ".jpg";
        Mat image = imread(imagePath);
        imshow("orig", image);
        
        // Image proc
        Mat hsvImage;
        cvtColor(image, hsvImage, COLOR_BGR2HSV);
        
        // Compute union image of two color masks
        Mat whiteMask(hsvImage.size(), CV_8U);
        Mat blackMask(hsvImage.size(), CV_8U);
        //int bw cutoff = 40
        inRange(hsvImage, Scalar(0, 0, 40), Scalar(179, 160, 255), whiteMask);
        inRange(hsvImage, Scalar(0, 0, 0), Scalar(179, 255, 40), blackMask);
        //imshow("white mask", whiteMask);
        //imshow("black mask", blackMask);

       // Mat bl;
        //blur(image, image, Size(4,4)); 

        Mat edge;
        
        Canny(image, edge, 50, 50*2);
        imshow("canny", edge);
        /*
        vector<Vec4i> lines;
        HoughLinesP( edge, lines, 1, CV_PI/180, 80, 30, 10 );
        for( size_t i = 0; i < lines.size(); i++ )
        {
         //   line( image, Point(lines[i][0], lines[i][1]),
          //  Point( lines[i][2], lines[i][3]), Scalar(0,0,255), 3, 8 );
        }*/
        //imshow("done", image);

        // Window poll
        waitKey(0);
    }
}
