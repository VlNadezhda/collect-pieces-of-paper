#include <iostream>
#include <fstream>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/imgproc.hpp"
#include <map>

using namespace cv;

RNG rng(12345);
int thresh = 100;

typedef std::tuple<double,double,double> ABC;

struct spec_line{
    double l1;
    double l2;
    int index;
    int num;
};

std::tuple<double,double,double> get_line_through_points
(cv::Point p0,cv::Point p1){

    std::tuple<double,double,double> result(
            p1.x - p0.x,  p1.y - p0.y,
            p1.x*p0.y - p0.x*p1.y);
//    std::cout<<"["<<p1.x<<":"<<p0.x<<"]["<<p1.y<<";"<<p0.y<<"]\n";
//    std::cout<<"A: "<<std::get<0>(result)<<"B: "<<std::get<1>(result)<<"C: "<<std::get<2>(result)<<"\n";
    return result;
};

double distance(cv::Point p0,cv::Point p1){
    double result = sqrt((p1.x-p0.x)*(p1.x-p0.x) + (p1.y-p0.y)*(p1.y-p0.y));
    return round(result*10)/10;
}
double the_angle_between_the_lines(std::tuple<double,double,double> l1,std::tuple<double,double,double> l2){
    double A1 = std::get<0>(l1);
    double B1 = std::get<1>(l1);

    double A2 = std::get<0>(l2);
    double B2 = std::get<1>(l2);

    double cos = abs(A1*A2+B1*B2)/(sqrt(A1*A1+B1*B1)*sqrt(A2*A2+B2*B2));
//    std::cout<<cos<<"\n";
    cos = round(cos*10000)/10000;
    cos = round(cos*100)/100;
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

//    для удобства поворота в дальнейшем делаем квадратным изображение
    int length = im.cols > im.rows ? im.cols : im.rows;
    im = GetSquareImage(im, length);
//    namedWindow("pre",WINDOW_AUTOSIZE);
//    imshow("pre",im);
//    waitKey(0);
};

std::vector<std::vector<cv::Point>> find_contours(const cv::Mat& img)
{
    Mat im = img;
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = im.clone();
    cv::findContours( contourOutput, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE );

    approxPolyDP(Mat(contours[0]), contours[0], 3, false);
    return  contours;
}


std::vector<std::pair<double,spec_line>> contour_processing(
        std::vector<std::vector<cv::Point>> &contours, Mat &im, int ind){
//    cv::Mat contourImage(im.size(), CV_8UC3, cv::Scalar(0,0,0));
//    Mat res = contourImage;
//    for (int i = 0; i < contours[0].size(); ++i){
//        circle(res, contours[0][i], i+3, Scalar(0,255,0));
//    }
//
//    for (size_t idx = 0; idx < contours.size(); idx++) {
//        cv::drawContours(contourImage, contours, 0, 255);
//    }
//    cv::imshow("Contours", contourImage);
//    cv::moveWindow("Contours", 200, 0);
//    cv::waitKey(0);

    std::vector<std::pair<ABC,double>> coef;
    std::pair<ABC,double> temp;
    for (uint i = 1; i < contours[0].size(); ++i){
        temp.first = get_line_through_points(contours[0][i-1],contours[0][i]);
        temp.second = distance(contours[0][i-1],contours[0][i]);
        coef.push_back(std::move(temp));
    }

    std::vector<std::pair<double,spec_line>> result;
    std::pair<double,spec_line> tmp;
    uint  k;
    for(uint i = 1; i < coef.size(); ++i){
      k = i;
      tmp.first = the_angle_between_the_lines(coef.at(k-1).first,coef.at(i).first);
      tmp.second.l1 = coef.at(i-1).second;
      tmp.second.l2 = coef.at(i).second;
      tmp.second.index = ind;
      tmp.second.num = i;
//        std::cout<<"degree: "<<tmp.first<<"\nl1: "<<tmp.second.l1<<
//                 "l2: "<<tmp.second.l2<<"\nindex: "<<tmp.second.index
//                 <<" num: "<<tmp.second.num<<"\n------\n";
      result.push_back(tmp);
    }
    return result;
}

bool are_same(spec_line line_1,spec_line line_2){
    int lM_1;
    int lm_1;

    int lM_2;
    int lm_2;

    if(line_1.l1 > line_1.l2){
        lM_1 =  line_1.l1;
        lm_1 = line_1.l2;
    }else{
        lM_1 =  line_1.l2;
        lm_1 = line_1.l1;
    }

    if(line_2.l1 > line_2.l2){
        lM_2 =  line_2.l1;
        lm_2 = line_2.l2;
    }else{
        lM_2 =  line_2.l2;
        lm_2 = line_2.l1;
    }

//    std::cout<<lM_1<<" "<<lm_1<<"; "<<lM_2<<" "<<lm_2<<"\n";
    if( (lM_1 - 6 < lM_2 && lM_2 < lM_1 + 6) && (lm_1 - 6 < lm_2 && lm_2 < lm_1 + 6)){
        return  true;
    }
    return false;
}

void search_for_matches(std::vector<std::pair<double,spec_line>> d1,
                        std::vector<std::pair<double,spec_line>>d2){
    std::map<double,spec_line> find_map;
    double eps = 0.01;

    for(auto i: d1){
        find_map[i.first] = i.second;
    }

    std::vector<std::pair<spec_line,spec_line>>coincidences;
    std::pair<spec_line,spec_line>tmp;
//    for(auto i: d2){
//        if(find_map.count(i.first) && are_same(find_map[i.first],i.second)){
//            tmp.first = i.second;
//            tmp.second = find_map[i.first];
//            coincidences.push_back(std::move(tmp));
//        }
//    }

    for(uint i = 0; i < d1.size(); ++i){
        for(uint j = 0; j < d1.size(); ++j){
            if(d1[i].first == d2[j].first || (d1[i].first - 0.02 < d2[j].first && d2[j].first < d1[i].first + 0.02)){
                if(are_same(d1[i].second,d2[j].second)){
                    tmp.first = d1[i].second;
                    tmp.second = d2[j].second;
                    coincidences.push_back(std::move(tmp));
                }
            }
        }
    }

    for(auto i:coincidences){
            std::cout<<"line: "<<i.first.index<<"\nl1: "<<i.first.l1
            <<" l2: "<<i.first.l2<<"\nnum: "<<i.first.num<<"**********\n";;
            std::cout<<"line: "<<i.second.index<<"\nl1: "<<i.second.l1
            <<" l2: "<<i.second.l2<<"\nindex: "<<i.second.num<<"\n------\n";
    }

//    for(auto i: find_map){
//        std::cout<<"degree: "<<i.first<<"\nl1: "<<i.second.l1<<
//        "l2: "<<i.second.l2<<"\nindex: "<<i.second.index<<"\n------\n";
//    }

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

    preprocessing(parts.at(0));
    preprocessing(parts.at(1));

    std::vector<std::vector<cv::Point>> c0 = find_contours(parts.at(0));
    std::vector<std::vector<cv::Point>> c1= find_contours(parts.at(1));

    std::vector<std::pair<double,spec_line>> v0;
    std::vector<std::pair<double,spec_line>> v1;

    v0 = contour_processing(c0,parts.at(0),0);
    v1 = contour_processing(c1,parts.at(2), 1);
    std::reverse(v1.begin(), v1.end());

    search_for_matches(v0,v1);

//    for(auto im: parts){
//
//    }

    return 0;
}
