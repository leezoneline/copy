#include "dtcr.hpp"
#include "pnpsvr.hpp" // 包含 PnPSolver 头文件
#include <iostream>
#include <opencv2/opencv.hpp>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/pose.hpp" // 包含 Pose 消息类型

using namespace cv;
using namespace std;

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("dtcr_node");
    auto raw_publisher = node->create_publisher<sensor_msgs::msg::Image>("image_raw", 10);
    auto result_publisher = node->create_publisher<sensor_msgs::msg::Image>("result", 10);
    auto binary_publisher = node->create_publisher<sensor_msgs::msg::Image>("binary_img", 10); // 新的发布者
    auto pose_publisher = node->create_publisher<geometry_msgs::msg::Pose>("armor_pose", 10); // 新的姿态发布者

    // 默认参数（您可以从配置文件加载这些参数）
    LightParams myLightParams;
    ArmorParams myArmorParams;
    int binary_thres = 30;
    EnemyColor ourColor = EnemyColor::BLUE;
    bool is_drawArmor = true;
    string video_path = "/home/lee/Documents/gofkuself/copy/src/dtcr/resource/v.mp4"; // 视频文件路径

    // 创建 Detector 实例
    Detector myDetector(myLightParams, myArmorParams, binary_thres, ourColor, is_drawArmor);

    // 打开视频文件
    VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        RCLCPP_ERROR(node->get_logger(), "无法打开视频文件: %s", video_path.c_str());
        return -1;
    }

    // Create and spin the PnPSolver node
    auto pnp_solver = std::make_shared<dtcr::PnPSolver>();
    std::thread pnp_thread([&pnp_solver]() {
        rclcpp::spin(pnp_solver);
    });

    Mat frame;
    while (rclcpp::ok() && cap.read(frame)) {
        if (frame.empty()) {
            RCLCPP_WARN(node->get_logger(), "抓取到空白帧");
            break;
        }

        // 将原始帧转换为 sensor_msgs::msg::Image 并发布
        cv_bridge::CvImage raw_img_bridge;
        sensor_msgs::msg::Image raw_img_msg;
        raw_img_bridge = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::BGR8, frame);
        raw_img_bridge.toImageMsg(raw_img_msg);
        raw_publisher->publish(raw_img_msg);

        // 处理帧
        std::vector<Armor> armors = myDetector.detect(frame, "frame"); // 您可以传递帧标识符

        // 发布装甲板的姿态
        for (const auto &armor : armors) {
            geometry_msgs::msg::Pose pose_msg;
            pose_msg.position.x = armor.center.x;
            pose_msg.position.y = armor.center.y;
            pose_msg.position.z = 0.0; // 假设 Z 坐标为 0

            // 这里需要根据你的实际情况计算四元数
            pose_msg.orientation.x = 0.0;
            pose_msg.orientation.y = 0.0;
            pose_msg.orientation.z = 0.0;
            pose_msg.orientation.w = 1.0;

            pose_publisher->publish(pose_msg);
        }

        // 将结果 cv::Mat 转换为 sensor_msgs::msg::Image 并发布
        cv_bridge::CvImage result_img_bridge;
        sensor_msgs::msg::Image result_img_msg;
        result_img_bridge = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::BGR8, myDetector.resultPic);
        result_img_bridge.toImageMsg(result_img_msg);
        result_publisher->publish(result_img_msg);

        // 将二值图像转换为 sensor_msgs::msg::Image 并发布
        cv_bridge::CvImage binary_img_bridge;
        sensor_msgs::msg::Image binary_img_msg;
        binary_img_bridge = cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8, myDetector.binary_img); // 假设 binary_img 是灰度图
        binary_img_bridge.toImageMsg(binary_img_msg);
        binary_publisher->publish(binary_img_msg);

        rclcpp::spin_some(node);
    }

    // 释放视频捕获
    cap.release();
    rclcpp::shutdown();
    pnp_thread.join(); // Wait for the PnPSolver thread to finish
    return 0;
}

std::vector<Armor> Detector::detect(const Mat &input, const string picName) { // 修改后的签名
    if (is_drawArmor)
        resultPic = input.clone();  // 如果需要绘制盔甲，复制输入图像到 resultPic

    Mat binary_img;
    binary_img = preprocessImage(input);  // 对输入图像进行预处理（转换为二值图）
    this->binary_img = binary_img.clone(); // 保存二值图像

    auto lights_ = findLights(input, binary_img);  // 查找图像中的灯条

    auto armors_ = matchLights(lights_, input);  // 根据灯条匹配盔甲

    for (long unsigned int i = 0; i < armors_.size(); i++) {
        armors_[i].number_img = extractNumber(input, armors_[i]);
    }

    if (is_drawArmor) {
        for (long unsigned int i = 0; i < armors_.size(); i++) {
            // 绘制装甲板的矩形框
            cv::Point2f vertices[4];
            armors_[i].getVertices(vertices); // 获取装甲板的四个顶点

            for (int j = 0; j < 4; j++) {
                cv::line(resultPic, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 2); // 绿色线条
            }

            armors_[i].drawArmor(resultPic);
            // 在图像上绘制装甲板中心坐标
            cv::putText(resultPic,
                        "(" + std::to_string(armors_[i].center.x) + ", " + std::to_string(armors_[i].center.y) + ")",
                        armors_[i].center, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
            imwrite("./result/num_" + picName + "_" + to_string(i) + ".jpg", armors_[i].number_img);
        }
        imwrite("./result/result_" + picName + "_" + to_string(binary_thres) + ".jpg", resultPic);
    }
    return armors_; // 确保返回一个 std::vector<Armor> 对象
}

Mat Detector::preprocessImage(const Mat &rgb_img) {
    Mat img;
    cvtColor(rgb_img, img, COLOR_RGB2GRAY);

    Mat binary_img;
    threshold(img, binary_img, binary_thres, 255, THRESH_BINARY);

    return binary_img;
}

bool Detector::isLight(const Light &light) {
    // 计算灯条的宽高比
    float ratio = light.width / light.length;
    bool ratio_ok = light_params.min_ratio < ratio && ratio < light_params.max_ratio;

    // 判断灯条的倾斜角度是否符合要求
    bool angle_ok = (light.tilt_angle < (light_params.max_angle + 90.0)) &&
                    (light.tilt_angle > (90.0 - light_params.max_angle));

    bool is_light = ratio_ok && angle_ok;  // 如果宽高比和角度符合条件，则认为是灯条

    return is_light;
}

vector<Light> Detector::findLights(const Mat &rgb_img, const Mat &binary_img) {
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);  // 查找二值图像中的轮廓

    vector<Light> lights;

    for (const auto &contour : contours) {
        if (contour.size() < 6) continue;  // 如果轮廓点数小于6，跳过该轮廓

        auto light = Light(contour);  // 创建一个 Light 对象，表示灯条

        if (isLight(light)) {  // 如果符合灯条特征
            int sum_r = 0, sum_b = 0;
            for (const auto &point : contour) {
                sum_r += rgb_img.at<cv::Vec3b>(point.y, point.x)[0];  // 累加红色通道的值
                sum_b += rgb_img.at<cv::Vec3b>(point.y, point.x)[2];  // 累加蓝色通道的值
            }
            // 判断红色和蓝色通道的差值，来确定灯条的颜色
            if (std::abs(sum_r - sum_b) / static_cast<int>(contour.size()) > light_params.color_diff_thresh) {
                light.color = sum_r > sum_b ? EnemyColor::RED : EnemyColor::BLUE;
            }
            // 如果灯条颜色是我们所关注的颜色，则保存该灯条
            if (light.color == ourColor) {
                lights.emplace_back(light);
                light.drawLight(resultPic);  // 在结果图上绘制灯条
            }
        }
    }

    std::sort(lights.begin(), lights.end(), [](const Light &l1, const Light &l2) {
        return l1.center.x < l2.center.x;
    });  // 根据灯条的中心点 x 坐标排序

    return lights;
}

bool Detector::containLight(const int i, const int j, const vector<Light> &lights) {
    const Light &light_1 = lights.at(i), light_2 = lights.at(j);
    auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
    auto bounding_rect = cv::boundingRect(points);  // 计算包围框
    double avg_length = (light_1.length + light_2.length) / 2.0;
    double avg_width = (light_1.width + light_2.width) / 2.0;

    // 判断是否有其他灯条在这两个灯条之间
    for (int k = i + 1; k < j; k++) {
        const Light &test_light = lights.at(k);

        if (test_light.width > 2 * avg_width || test_light.length < 0.5 * avg_length) {
            continue;  // 防止噪声干扰（例如数字、红点准星等）
        }

        if (bounding_rect.contains(test_light.top) || bounding_rect.contains(test_light.bottom) ||
            bounding_rect.contains(test_light.center)) {
            return true;  // 如果有灯条在这两个灯条之间，返回 true
        }
    }
    return false;  // 否则返回 false
}

ArmorType Detector::isArmor(const Light &light_1, const Light &light_2) {
    // 判断两个灯条是否符合盔甲的特征（包括灯条的长度比、中心距离、倾斜角度等）
    float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length : light_2.length / light_1.length;
    bool light_ratio_ok = light_length_ratio > armor_params.min_light_ratio;

    // 计算两个灯条中心点的距离
    float avg_light_length = (light_1.length + light_2.length) / 2;
    float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
    bool center_distance_ok = (armor_params.min_small_center_distance <= center_distance &&
                               center_distance < armor_params.max_small_center_distance) ||
                              (armor_params.min_large_center_distance <= center_distance &&
                               center_distance < armor_params.max_large_center_distance);

    // 判断灯条之间的角度是否符合
    cv::Point2f diff = light_1.center - light_2.center;
    float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
    float angle_difference = std::abs(light_1.tilt_angle - light_2.tilt_angle);
    bool angle_ok = angle < armor_params.max_angle && angle_difference < armor_params.max_angle_difference;

    bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;  // 如果符合条件，则为有效盔甲

    ArmorType type;
    if (is_armor) {
        type = center_distance > armor_params.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    } else {
        type = ArmorType::INVALID;
    }

    return type;  // 返回盔甲类型（大、小或无效）
}

std::vector<Armor> Detector::matchLights(const std::vector<Light> &lights, const cv::Mat &img) {
    std::vector<Armor> armors;
    // 遍历所有灯条对，尝试配对成盔甲
    for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
        double max_iter_width = light_1->length * armor_params.max_large_center_distance;
        double min_iter_width = light_1->width;

        for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
            double distance_1_2 = light_2->center.x - light_1->center.x;
            if (distance_1_2 < min_iter_width) continue;
            if (distance_1_2 > max_iter_width) break;

            if (containLight(light_1 - lights.begin(), light_2 - lights.begin(), lights)) {
                continue;  // 如果灯条之间有其他灯条，跳过该配对
            }

            auto type = isArmor(*light_1, *light_2);  // 判断是否为盔甲
            if (type != ArmorType::INVALID) {
                auto armor = Armor(*light_1, *light_2);  // 创建盔甲对象
                armor.type = type;
                armor.center = (light_1->center + light_2->center) / 2; // 计算装甲板中心
                armors.emplace_back(armor);  // 将有效盔甲添加到结果中
            }
        }
    }
    return armors;
}

Mat Detector::extractNumber(const Mat &src, Armor &armor){
    // Number ROI size
    static const Size roi_size(20, 28);
    static const Size input_size(28*10, 28*10);

    cv::Point2f target_vertices[4] = {
        Point(0, 11),
        Point(0, 0),
        Point(53, 0),
        Point(53, 11)
    };
    cv::Point2f lights_vertices[4] = {
        armor.left_light.bottom,
        armor.left_light.top,
        armor.right_light.top,
        armor.right_light.bottom
    };

    Mat number_image;
    auto rotation_matrix = getPerspectiveTransform(lights_vertices, target_vertices);
    warpPerspective(src, number_image, rotation_matrix, Size(54, 28));

    // Get ROI
    number_image = number_image(Rect(Point(17, 0), Size(20, 28)));

    // Binarize
    cvtColor(number_image, number_image, COLOR_RGB2GRAY);
    threshold(number_image, number_image, 0, 255, THRESH_BINARY | THRESH_OTSU);
    resize(number_image, number_image, input_size);
    return number_image;
}