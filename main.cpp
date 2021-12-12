#include <iostream>
#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

RNG rng(12345);
int thresh = 100;

std::tuple<double,double,double> get_line_through_points
(cv::Point p0,cv::Point p1){

    std::tuple<double,double,double> result(
            p1.y - p0.y, p0.x - p0.x,
            p1.x*p0.y - p0.x*p1.y);
    std::cout<<"["<<p1.x<<":"<<p0.x<<"]["<<p1.y<<";"<<p0.y<<"]\n";
    std::cout<<"A: "<<std::get<0>(result)<<"B: "<<std::get<1>(result)<<"C: "<<std::get<2>(result)<<"\n";
    return result;
};

double distance(cv::Point p0,cv::Point p1){
    return sqrt((p1.x-p0.x)*(p1.x-p0.x) + (p1.y-p0.y)*(p1.y-p0.y));
}
double the_angle_between_the_lines(std::tuple<double,double,double> l1,std::tuple<double,double,double> l2){
    double A1 = std::get<0>(l1);
    double B1 = std::get<1>(l1);

    double A2 = std::get<0>(l2);
    double B2 = std::get<1>(l2);

    double cos = (A1*A2+B1*B2)/(sqrt(A1*A1+B1*B1)*sqrt(A2*A2+B2*B2));
    return cos;
}

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

void preprocessing(cv::Mat& im){
    medianBlur(im, im, 3);
    cv::cvtColor(im, im, COLOR_BGR2GRAY);
    cv::threshold(im, im, 128, 255, THRESH_BINARY);

    //для удобства поворота в дальнейшем делаем квадратным изображение
//    int length = img.cols > img.rows ? img.cols : img.rows;
//    Mat result = GetSquareImage(im, length);
//    namedWindow("pre",WINDOW_AUTOSIZE);
//    imshow("pre",im);
//    waitKey(0);
};

std::vector<std::vector<cv::Point>> find_contours(const cv::Mat& img)
{
    Mat im = img;
//    cv::cvtColor(im, im, COLOR_BGR2GRAY);
//    cv::threshold(im, im, 128, 255, THRESH_BINARY);

    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = im.clone();
    cv::findContours( contourOutput, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );

    approxPolyDP(Mat(contours[0]), contours[0], 3, false);
    return  contours;
}

void contour_processing(std::vector<std::vector<cv::Point>> &contours, Mat &im){
    cv::Mat contourImage(im.size(), CV_8UC3, cv::Scalar(0,0,0));
    Mat res = contourImage;
    for (int i = 0; i < contours[0].size(); ++i){
        circle(res, contours[0][i], 3, Scalar(0,255,0));
    }

    for (size_t idx = 0; idx < contours.size(); idx++) {
        cv::drawContours(contourImage, contours, 0, 255);
    }
    cv::imshow("Contours", contourImage);
    cv::moveWindow("Contours", 200, 0);
    cv::waitKey(0);


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
        preprocessing(im);
        std::vector<std::vector<cv::Point>> c = find_contours(im);
        contour_processing(c, im);
    }

    return 0;
}
