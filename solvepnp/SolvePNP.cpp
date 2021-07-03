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

int main()
{
    // One time AR Dict setup
    tagSetup();

    // Camera mats
    intrinsics = Mat::eye(3, 3, CV_64F);
    intrinsics.at<double>(0, 0) = 1399.58; // fx
    intrinsics.at<double>(1, 1) = 1399.44;
    intrinsics.at<double>(0, 2) = 974.21;
    intrinsics.at<double>(1, 2) = 533.858;

    distCoeffs = Mat::zeros(8, 1, CV_64F);
    distCoeffs.at<double>(0, 0) = -0.171441;
    distCoeffs.at<double>(1, 0) = 0.0257079;
    distCoeffs.at<double>(2, 0) = -3.5109e-05;
    distCoeffs.at<double>(3, 0) = 3.96575e-05;
    distCoeffs.at<double>(4, 0) = 1.84418e-11;

    // SolvePNP AR tag corner world coordinates in clockwise order from top left to match AR tag library
    std:;vector<cv::Point3d> worldPoints;
    worldPoints.push_back(cv::Point3d(0, 0, 0)); //top left
    worldPoints.push_back(cv::Point3d(7.375, 0, 0)); //top right
    worldPoints.push_back(cv::Point3d(7.375, 7.375, 0)); //bot right
    worldPoints.push_back(cv::Point3d(0, 7.375, 0)); //bot left

    // Open the ZED camera
    VideoCapture cap(2);
    if(!cap.isOpened())
        return -1;

    // Set the video resolution to HD720 (2560*720)
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 2560);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);

    while(cv::waitKey(10) != ' ')
    {
        Mat frame, left, right;
        // Get a new frame from camera
        cap >> frame;
        
        // Extract left and right images from side-by-side
        left = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
        right = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

        // AR Detect
        Mat arMat;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f> > corners;
        cvtColor(left, arMat, COLOR_RGBA2RGB);
        cv::aruco::detectMarkers(arMat, alvarDict, corners, ids, alvarParams);

        // Undistortion and visualization
        Mat undistorted = arMat.clone();
        //undistort(arMat, undistorted, intrinsics, distCoeffs);
        if(ids.size() > 0) {
            cv::aruco::drawDetectedMarkers(undistorted, corners, ids);
            circle(undistorted, corners[0][0], 3, COLOR_CYAN, -1);
            circle(undistorted, corners[0][1], 3, COLOR_RED, -1);
            circle(undistorted, corners[0][2], 3, COLOR_WHITE, -1);
            circle(undistorted, corners[0][3], 3, COLOR_ORANGE, -1);
        }

        imshow("undistort", undistorted);

        // Perspective n point to solve for extrinsic parameters
        if(ids.size() <= 0) continue;
        cv::Mat rvec, tvec, R;		
        solvePnP(worldPoints, corners[0], intrinsics, distCoeffs, rvec, tvec);
        
        // Pose estimation
        cv::Rodrigues(rvec, R); //convert axis-angle notation into rotation matrix
        Mat pose = -R.t() * tvec; //Result of linear algebra on transformation from world to camera space
                                  //Rw + t = c, where w is in world space, c in 3D camera space
                                  //solve and plug in c = [0,0,0] for position of camera in world space
                                  //w = -R^(-1) * t, but R^(-1) is R^T since rotation is always an orthogonal matrix
        
        cout << pose << endl;
        // Euclidean norm
        //cout << sqrt(pose.at<double>(0,0)*pose.at<double>(0,0) + pose.at<double>(1,0)*pose.at<double>(1,0)+pose.at<double>(2,0)+pose.at<double>(2,0)) << endl;
    }
    

    // Deinitialize camera in the VideoCapture destructor
    return 0;
}
