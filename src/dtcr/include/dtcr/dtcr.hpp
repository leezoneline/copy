// std
#include <cmath>
#include <string>
#include <vector>
// #include <iostream>
#include <numeric>
#include <algorithm>
#include <execution>
#include <array>

// cv
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

// ros2
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"

using namespace cv;
using namespace std;

// Armor type
enum class ArmorType { SMALL, LARGE, INVALID };

// Color type
enum class EnemyColor { RED = 0, BLUE = 1, WHITE = 2, };

///////////////
//           //
// parameter //
//           //
///////////////

// Light parameter
class LightParams {
public:
    LightParams(double min_rati = 0.08, double max_ratio = 0.4, double max_angle = 40.0, int color_diff_thresh = 25) {
        // width / height
        this->min_ratio = min_rati;
        this->max_ratio = max_ratio;
        // vertical angle
        this->max_angle = max_angle;
        // judge color
        this->color_diff_thresh = color_diff_thresh;
    }

    // width / height
    double min_ratio;
    double max_ratio;

    // vertical angle
    double max_angle;

    // judge color
    int color_diff_thresh;
};

// Armor parameter
class ArmorParams {
public:
    ArmorParams(double min_light_ratio = 0.6, double min_small_center_distance = 0.8, double max_small_center_distance = 3.2, double min_large_center_distance = 3.2, double max_large_center_distance = 5.0, double max_angle = 15.0, double max_angle_difference = 15.0) {
        this->min_light_ratio = min_light_ratio;

        this->min_small_center_distance = min_small_center_distance;
        this->max_small_center_distance = max_small_center_distance;
        this->min_large_center_distance = min_large_center_distance;
        this->max_large_center_distance = max_large_center_distance;

        this->max_angle = max_angle;

        this->max_angle_difference = max_angle_difference;
    }
    // width / height
    double min_light_ratio = 0.6;
    // light pairs distance
    double min_small_center_distance = 0.8;
    double max_small_center_distance = 3.2;
    double min_large_center_distance = 3.2;
    double max_large_center_distance = 5.0;
    // horizontal angle
    double max_angle = 35.0;

    // max angle difference between two light
    double max_angle_difference = 15.0;
};

///////////////////
//               //
// light & armor //
//               //
///////////////////

struct Light : public RotatedRect {
    Light() = default;
    explicit Light(const vector<Point>& contour)
        : RotatedRect(cv::minAreaRect(contour)) {

        center = std::accumulate(
            contour.begin(),
            contour.end(),
            cv::Point2f(0, 0),
            [n = static_cast<float>(contour.size())](const cv::Point2f& a, const Point& b) {
                return a + cv::Point2f(b.x, b.y) / n;
            });

        cv::Point2f p[4];

        this->points(p);  // 001  (取外接旋转矩形的4个角)

        corner_p[0] = p[0];
        corner_p[1] = p[1];
        corner_p[2] = p[2];
        corner_p[3] = p[3];

        std::sort(p, p + 4, [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
        top = (p[0] + p[1]) / 2;
        bottom = (p[2] + p[3]) / 2;

        length = cv::norm(top - bottom);
        width = cv::norm(p[0] - p[1]);

        axis = top - bottom;
        axis = axis / cv::norm(axis); // 002 (倾斜方向的单位向量)

        // Calculate the tilt angle
        tilt_angle = std::atan2(-(top.y - bottom.y), (top.x - bottom.x));
        tilt_angle = tilt_angle / CV_PI * 180; // 003 (倾斜方向(单位度))
    }
    EnemyColor color = EnemyColor::WHITE;
    cv::Point2f corner_p[4];
    cv::Point2f top, bottom, center;
    cv::Point2f axis;
    double length;
    double width;
    float tilt_angle;

    void drawLight(Mat img) {
        line(img, corner_p[0], corner_p[1], Scalar(255, 0, 255), 2);
        line(img, corner_p[1], corner_p[2], Scalar(255, 0, 255), 2);
        line(img, corner_p[2], corner_p[3], Scalar(255, 0, 255), 2);
        line(img, corner_p[0], corner_p[3], Scalar(255, 0, 255), 2);
        putText(img, to_string(tilt_angle), Point(corner_p[0].x - 5, corner_p[0].y - 5), FONT_HERSHEY_PLAIN, 1, Scalar(0, 69, 255), 1);
    }
};

// Struct used to store the armor
class Armor {
public:
    static constexpr const int N_LANDMARKS = 6;
    static constexpr const int N_LANDMARKS_2 = N_LANDMARKS * 2;
    Armor() = default;
    Armor(const Light& l1, const Light& l2) {
        if (l1.center.x < l2.center.x) {
            left_light = l1, right_light = l2;
        }
        else {
            left_light = l2, right_light = l1;
        }

        center = (left_light.center + right_light.center) / 2;
    }

    // Build the points in the object coordinate system, start from bottom left in
    // clockwise order
    template <typename PointType>
    static inline std::vector<PointType> buildObjectPoints(const double& w,
        const double& h) noexcept {
        if constexpr (N_LANDMARKS == 4) {
            return { PointType(0, w / 2, -h / 2),
                            PointType(0, w / 2, h / 2),
                            PointType(0, -w / 2, h / 2),
                            PointType(0, -w / 2, -h / 2) };
        }
        else {
            return { PointType(0, w / 2, -h / 2),
                            PointType(0, w / 2, 0),
                            PointType(0, w / 2, h / 2),
                            PointType(0, -w / 2, h / 2),
                            PointType(0, -w / 2, 0),
                            PointType(0, -w / 2, -h / 2) };
        }
    }

    // Landmarks start from bottom left in clockwise order
    std::vector<cv::Point2f> landmarks() const {
        if constexpr (N_LANDMARKS == 4) {
            return { left_light.bottom, left_light.top, right_light.top, right_light.bottom };
        }
        else {
            return { left_light.bottom,
                            left_light.center,
                            left_light.top,
                            right_light.top,
                            right_light.center,
                            right_light.bottom };
        }
    }

    void drawArmor(Mat& result) {
        line(result, lights_vertices[0], lights_vertices[1], Scalar(0, 255, 0), 2);
        line(result, lights_vertices[1], lights_vertices[2], Scalar(0, 255, 0), 2);
        line(result, lights_vertices[2], lights_vertices[3], Scalar(0, 255, 0), 2);
        line(result, lights_vertices[0], lights_vertices[3], Scalar(0, 255, 0), 2);
    }

    void getVertices(cv::Point2f vertices[4]) const {
        float half_width = width / 2.0;
        float half_height = height / 2.0;

        vertices[0] = cv::Point2f(center.x - half_width, center.y - half_height); // 左上角
        vertices[1] = cv::Point2f(center.x - half_width, center.y + half_height); // 左下角
        vertices[2] = cv::Point2f(center.x + half_width, center.y + half_height); // 右下角
        vertices[3] = cv::Point2f(center.x + half_width, center.y - half_height); // 右上角
    }

    // Light pairs part
    Light left_light, right_light;
    Point2f center;
    ArmorType type;

    // Number part
    Mat number_img;
    string number;
    float confidence;
    string classfication_result;

    cv::Point2f lights_vertices[4];
    float width;
    float height;
};

//////////////
//          //
// function //
//          //
//////////////

class Detector {
public:
    Detector(LightParams light_params, ArmorParams armor_params, int binary_thres, EnemyColor ourColor, bool is_drawArmor = true) {
        this->light_params = light_params;
        this->armor_params = armor_params;
        this->binary_thres = binary_thres;
        this->ourColor = ourColor;
        this->is_drawArmor = is_drawArmor;
    }
    std::vector<Armor> detect(const Mat& input, const string picName = "result"); // Modified signature

    Mat resultPic;
    Mat binary_img;

private:
    int binary_thres;

    LightParams light_params;
    ArmorParams armor_params;

    bool is_drawArmor;

    EnemyColor ourColor;

    Mat preprocessImage(const Mat& rgb_img);

    bool isLight(const Light& light);

    vector<Light> findLights(const Mat& rgb_img, const cv::Mat& binary_img);

    bool containLight(const int i, const int j, const vector<Light>& lights);

    ArmorType isArmor(const Light& light_1, const Light& light_2);

    vector<Armor> matchLights(const std::vector<Light>& lights, const Mat& img);

    Mat extractNumber(const Mat& src, Armor& armor);
};