#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <iostream>
#include <vector>
using namespace cv;
using namespace std;

void cornerDetection(const Mat &image) {

    Mat new_image = Mat::zeros( image.size(), image.type() );
    double alpha = 5.0; /*< Simple contrast control */
    int beta = 0;       /*< Simple brightness control */
    //cout << " Basic Linear Transforms " << endl;
    //cout << "-------------------------" << endl;
    //cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
    //cout << "* Enter the beta value [0-100]: ";    cin >> beta;
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < image.channels(); c++ ) {
                new_image.at<Vec3b>(y,x)[c] =
                  saturate_cast<uchar>( alpha*image.at<Vec3b>(y,x)[c] + beta );
            }
        }
    }
    
    
    imshow("Original Image", new_image);

    Mat hsvImage;
    cvtColor(new_image, hsvImage, COLOR_BGR2HSV);
    Mat binImage(hsvImage.size(), CV_8U);
    inRange(hsvImage, Scalar(0, 0, 0), Scalar(179, 255, 90), binImage);

    Mat binImage2(hsvImage.size(), CV_8U);
    inRange(hsvImage, Scalar(0, 0, 100), Scalar(179, 50, 255), binImage2);
    
    dilate(binImage, binImage, Mat(), Point(-1,-1));
    dilate(binImage2, binImage2, Mat(), Point(-1,-1));

    imshow("binary", binImage);
    imshow("binary white", binImage2);

    /*
  
    //parameters
    int maxCorners = 600;
    vector<Point2f> corners;
    double qualityLevel = 0.01;
    double minDistance = 10;
    int blockSize = 3, gradientSize = 3;
    bool useHarrisDetector = false;
    double k = 0.04;

    
    Mat imgGray, whiteCopy;
    whiteCopy = image.clone();
    whiteCopy.setTo(Scalar(255, 255, 255));
    cvtColor( image, imgGray, COLOR_BGR2GRAY );

    
    int morph_size = 2;
    Mat original = imgGray;
    Mat dst;
    Mat element = getStructuringElement(MORPH_RECT, Size( 2*morph_size + 1, 2*morph_size + 1 ), 
                                            Point( morph_size, morph_size ));
    morphologyEx(binImage, dst, MORPH_OPEN, element);
    imshow("Filtering", dst);
    
    //Mat dst = binImage;
    
    
    Mat dst;
    cvtColor( new_image, dst, COLOR_BGR2GRAY );

    GaussianBlur(dst, dst, Size(5,5), 0);
    imshow("Blur", dst); 

    goodFeaturesToTrack( dst,
                         corners,
                         maxCorners,
                         qualityLevel,
                         minDistance,
                         Mat(),
                         blockSize,
                         useHarrisDetector,
                         k );


    for (size_t j = 0; j < corners.size(); ++j) {
        circle(dst, corners[j], 4, Scalar(255, 0, 0), FILLED);
    }
    imshow("Contrast adjusted", dst);

    //imshow("Points", whiteCopy);
    Mat labels;
    Mat centers;
    int clusterCount = 11;
    kmeans(corners, clusterCount, labels, TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),
            3, KMEANS_PP_CENTERS, centers);

    vector<Point2f> centerPoints;
    for (size_t k = 0; k < clusterCount; ++k) {
        Point2f pt = centers.at<Point2f>(k);
        centerPoints.push_back(pt);
    } 
    
    Mat copy = image.clone();
    for (size_t i = 0; i < centerPoints.size(); ++i) {
        circle( copy, centerPoints[i], 40, Scalar(0, 255, 0), 1, FILLED );
    }
    imshow("Cluster Centers", copy);  
    */
}

int main() {
    /*
    Mat src = imread("0182.jpg");
    cornerDetection(src);
    waitKey();*/
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
            cornerDetection(image);
           // imshow("window", image); 
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
            cornerDetection(image);
           // imshow("window", image); 
            int c = waitKey(0);
        }
        if (i >= 100 && i < 212) {
            
            string iString = to_string(i);
            filenameTemp = "0" + iString + ".jpg";
            //filename = cv::samples::findFile(filenameTemp);  
            Mat image = imread(filenameTemp, IMREAD_COLOR);
            
            cornerDetection(image);
           // imshow("window", image); 
            int c = waitKey(0);
        }
    }
    return 0;
}
