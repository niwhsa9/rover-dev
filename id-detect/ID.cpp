#include <iostream>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/aruco.hpp>

using namespace cv;
using namespace std;

#define COLOR_WHITE Scalar(255, 255, 255)	//bgr color space...
#define COLOR_RED Scalar(0, 0, 255)
#define COLOR_CYAN Scalar(255, 255, 0)
#define COLOR_ORANGE Scalar(0, 128, 255)

cv::Ptr<cv::aruco::DetectorParameters> alvarParams;
cv::Ptr<cv::aruco::Dictionary> alvarDict;
Mat intrinsics, distCoeffs;

// rover tag setup
void tagSetup() {
    cv::FileStorage fsr("alvar_dict.yml", cv::FileStorage::READ);
    if (!fsr.isOpened()) {  //throw error if dictionary file does not exist
        std::cerr << "ERR: \"alvar_dict.yml\" does not exist! Create it before running main\n";
        throw Exception();
    }

    // read dictionary from file
    int mSize, mCBits;
    cv::Mat bits;
    fsr["MarkerSize"] >> mSize;
    fsr["MaxCorrectionBits"] >> mCBits;
    fsr["ByteList"] >> bits;
    fsr.release();
    alvarDict = new cv::aruco::Dictionary(bits, mSize, mCBits);

    // initialize other special parameters that we need to properly detect the URC (Alvar) tags
    alvarParams = new cv::aruco::DetectorParameters();
    alvarParams->markerBorderBits = 2;
    alvarParams->doCornerRefinement = false;
    alvarParams->polygonalApproxAccuracyRate = 0.08;
}

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


int main()
{
    // One time AR Dict setup
    tagSetup();


    // Open the ZED camera
    //VideoCapture cap(2);
    //if(!cap.isOpened())
     //   return -1;

    // Set the video resolution to HD720 (2560*720)
    //cap.set(CV_CAP_PROP_FRAME_WIDTH, 2560);
    //cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

   
    for(int i = 128; i <= 211; i++) {
        // Image load
        cout << "Showing image " << i << endl;
        stringstream num;
        num << setfill('0') << setw(4) << i;
        string imagePath = "data/" + num.str() + ".jpg";
        Mat image = imread(imagePath);
        //image = adjustContrast(image, 3.0);
        //imshow("orig", image);


        // AR Detect
        
        Mat arMat;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cvtColor(image, arMat, COLOR_RGBA2RGB);
        cv::aruco::detectMarkers(arMat, alvarDict, corners, ids, alvarParams);

        vector< vector< Point2f > > candidates;
        vector< vector< Point > > contours;
        cv::aruco::_detectCandidates(arMat, candidates, contours, alvarParams);
        cout << "num candid: " << candidates.size() << endl;
        //vector< vector< Point2f > > candidates = {{cv::Point2f(1025, 434), cv::Point2f(1091, 433), cv::Point2f(1091, 488), cv::Point2f(1027, 486)}};
        //vector< vector< Point > > contours = {{cv::Point(1025, 434), cv::Point(1091, 433), cv::Point(1091, 488), cv::Point(1027, 486)}};
        //vector< vector< Point > > contours = {{cv::Point(1024, 432), cv::Point(1091, 432), cv::Point(1092, 487), cv::Point(1026, 487)}};
        //vector< int > ids;
        /*
        for(int i = 0; i < candidates.size(); i++) {
            for(int pidx = 0; pidx < candidates[i].size(); pidx++) {
                cv::Point p = candidates[i][pidx];
                cout << "(" << p.x << ", " << p.y << ") ";
            }
            cout << endl;
        }*/
        for(int i = 0; i < contours.size(); i++) {
            for(int pidx = 0; pidx < contours[i].size(); pidx++) {
                cv::Point p = contours[i][pidx];
                cout << "(" << p.x << ", " << p.y << ") ";
            }
            cout << endl;
        }

        cv::drawContours(image, contours, 0, cv::Scalar(0, 0, 255), 3);
        cv::drawContours(image, contours, 1, cv::Scalar(0, 255, 0), 3);
        cv::drawContours(image, contours, 2, cv::Scalar(255, 0, 0), 3);

        cv::aruco::_identifyCandidates(arMat, candidates, contours, alvarDict, candidates, ids, alvarParams);
        if(ids.size() > 0) cout << "ID: " << ids[0] << endl;
        imshow("orig", image);

        // Undistortion and visualization
        /*
        Mat undistorted = arMat.clone();
        if(ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(undistorted, corners, ids);
            circle(undistorted, corners[0][0], 3, COLOR_CYAN, -1);
            circle(undistorted, corners[0][1], 3, COLOR_RED, -1);
            circle(undistorted, corners[0][2], 3, COLOR_WHITE, -1);
            circle(undistorted, corners[0][3], 3, COLOR_ORANGE, -1);
        }
        
        imshow("undistort", undistorted);
        */
        cv::waitKey(0);
    }
    

    // Deinitialize camera in the VideoCapture destructor
    return 0;
}
