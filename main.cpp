#include <iostream>
#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

RNG rng(12345);
int thresh = 100;

std::tuple<double,double,double> get_line_through_points
(Point_<int> p0,Point_<int> p1){

    std::tuple<double,double,double> result(
            p1.y - p0.y, p0.x - p0.x,
            p1.x*p0.y - p0.x*p1.y);
};

double  distance_point_line_squared
(std::tuple<double,double,double> coef, Point_<int> p0){

    double a = std::get<0>(coef);
    double b = std::get<1>(coef);
    double c = std::get<2>(coef);

    double result = (a*p0.x + b*p0.y + c)*(a*p0.x + b*p0.y + c)/(a*a + b*b);
    return result;
};

double distance_point_line_signed
(std::tuple<double,double,double> coef, Point_<int> p0){
    double a = std::get<0>(coef);
    double b = std::get<1>(coef);
    double c = std::get<2>(coef);

    double result = (a*p0.x + b*p0.y + c)/std::sqrt(a*a + b*b);
    return result;
};

std::pair<int,int> rotate (Mat image, double degrees){
    Point2f center((image.cols - 1) / 2.0, (image.rows - 1) / 2.0);
    Mat rotation_matix = getRotationMatrix2D(center, degrees, 1.0);

    Mat rotated_image;
    warpAffine(image, rotated_image, rotation_matix, image.size());
    imshow("Rotated image", rotated_image);
    waitKey(0);
};

cv::Mat GetSquareImage( const cv::Mat& img, int target_width)
{
    int width = img.cols,
            height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
};

cv::Mat preprocessing(const cv::Mat& img){
    Mat im = img;
    cv::cvtColor(img, im, COLOR_BGR2GRAY);
    cv::threshold(im, im, 128, 255, THRESH_BINARY);

    //для удобства поворота в дальнейшем делаем квадратным изображение
//    int length = img.cols > img.rows ? img.cols : img.rows;
//    Mat result = GetSquareImage(im, length);
//    namedWindow("pre",WINDOW_AUTOSIZE);
//    imshow("pre",result);
//    waitKey(0);
    return img;
};

cv::Mat finding_corners(const cv::Mat& img){
    Mat gray = img;
    cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
    Mat dst,dst_norm,dst_norm_scaled;
    dst=Mat::zeros(gray.size(),CV_32FC1);

    cornerHarris(gray,dst,15,5,0.05,BORDER_DEFAULT);

    normalize(dst,dst_norm,0,255,NORM_MINMAX,CV_32FC1,Mat());
    convertScaleAbs(dst_norm,dst_norm_scaled);

    for(int j = 0;j<dst_norm.rows;j++){
        for(int i = 0;i<dst_norm.cols;i++){
            if((int)dst_norm.at<float>(j,i)>150){
//                std::cout<<(int)dst_norm.at<float>(j,i)<<"  ";
                circle(dst_norm_scaled,Point(i,j),10,Scalar(255,255,255),1,8,0);
            }
        }
    }
//    imwrite("result.png",dst_norm_scaled);
    cv::moveWindow("corners_window", 200, 0);
    imshow("corners_window",dst_norm_scaled);
    waitKey(0);
};

std::vector<cv::Point> find_contours(const cv::Mat& img)
{
    Mat im = img;
    cv::cvtColor(im, im, COLOR_BGR2GRAY);
    cv::threshold(im, im, 128, 255, THRESH_BINARY);

    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = im.clone();
    cv::findContours( contourOutput, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );

//    for(uint i = 0; i < contours.size(); ++i){
//        for(uint j = 0; j < contours[i].size(); ++j){
//            std::cout<<contours[i][j]<<", ";
//        }
//        std::cout<<std::endl;
//    }
    //Draw the contours
    cv::Mat contourImage(im.size(), CV_8UC3, cv::Scalar(0,0,0));
    Mat res = contourImage;

    approxPolyDP(Mat(contours[0]), contours[0], 3, false);
    for (int i = 0; i < contours[0].size(); ++i){
        circle(res, contours[0][i], 3, Scalar(0,255,0));
    }
    for (size_t idx = 0; idx < contours.size(); idx++) {
        cv::drawContours(contourImage, contours, 0, 255);
    }
//    cv::imshow("ys", res);
    cv::imshow("Contours", contourImage);
    cv::moveWindow("Contours", 200, 0);
    cv::waitKey(0);

    return  contours[0];
}

int main(int argc, char* argv[]) {
    Mat img = imread(argv[1]);
    std::vector<Mat> parts;

    //получаем отдельный кусочек
    Mat roi_1(img, Rect(0, 240, 426, 500));
    Mat part_1;
    roi_1.copyTo(part_1);
    parts.push_back(part_1);

    Mat roi_2(img, Rect(426,150,345,500));
    Mat part_2;
    roi_2.copyTo(part_2);
    parts.push_back(part_2);

    Mat roi_3(img, Rect(760,400,500,500));
    Mat part_3;
    roi_3.copyTo(part_3);
    parts.push_back(part_3);

    for(auto im: parts){
        find_contours(im);
//        std::cout<<"\n_________\n";
    }

    return 0;
}