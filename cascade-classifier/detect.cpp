#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ximgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"

using namespace std;
using namespace cv;

const string imdir = "test/";


int main() {
    // Load   
    CascadeClassifier ar_cascade;
    ar_cascade.load("cascade30.xml");
    vector<cv::String> res;
    glob(imdir, res, false);


    for(cv::String imagePath : res) {
	cout << imagePath << endl;
        // Image load
        /*
        cout << "Showing image " << i << endl;
        stringstream num;
        num << setfill('0') << setw(4) << i;
        string imagePath = imdir + num.str() + ".jpg";
        cout << "path: " << imagePath << endl;*/
        Mat image = imread(imagePath);
        
        
        // Detect
        Mat gray;
        cvtColor( image, gray, COLOR_BGR2GRAY );
        equalizeHist( gray, gray );
        std::vector<Rect> detections;
        ar_cascade.detectMultiScale( gray, detections );

        for(int i = 0; i < detections.size(); i++) {
           rectangle(image, detections[i], Scalar(255, 0, 0), 3);
        }

        imshow("done", image);

        // Window poll
        waitKey(0);
    }
}
