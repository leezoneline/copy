#ifndef DTCR_PNPSVR_HPP_
#define DTCR_PNPSVR_HPP_

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>

namespace dtcr {

// 定义 ArmorCoordinates 消息类型
struct ArmorCoordinates {
    std::array<float, 8> image_points;  // 8个float，分别代表四个角点的x, y坐标
    std::string armor_type;             // 装甲板类型，例如 "small" 或 "large"
};

class PnPSolver : public rclcpp::Node {
public:
    PnPSolver();

private:
    void solvePnP(const ArmorCoordinates& msg);

    rclcpp::Subscription<ArmorCoordinates>::SharedPtr subscription_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr publisher_;
    cv::Mat cameraMatrix, distCoeffs;
};

} // namespace dtcr

#endif // DTCR_PNPSVR_HPP_