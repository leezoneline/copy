#include "pnpsvr.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp> // Add this line

using namespace std;
using namespace dtcr;

PnPSolver::PnPSolver() : Node("pnp_solver") {
    publisher_ = this->create_publisher<geometry_msgs::msg::Pose>("armor_pose", 10);

    // 相机内参矩阵
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    cameraMatrix.at<double>(0, 0) = 640; // fx
    cameraMatrix.at<double>(1, 1) = 480; // fy
    cameraMatrix.at<double>(0, 2) = 320; // cx
    cameraMatrix.at<double>(1, 2) = 240; // cy

    // 畸变系数
    distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
}

void PnPSolver::solvePnP(const ArmorCoordinates& msg) {
    // 获取图像坐标
    std::vector<cv::Point2f> image_points;
    for (size_t i = 0; i < 4; ++i) {
        image_points.push_back(cv::Point2f(msg.image_points[i * 2], msg.image_points[i * 2 + 1]));
    }

    // 定义世界坐标
    std::vector<cv::Point3f> object_points;
    if (msg.armor_type == "small") {
        // 小型装甲板的世界坐标
        object_points.push_back(cv::Point3f(-0.1, 0.1, 0));   // 左上角
        object_points.push_back(cv::Point3f(-0.1, -0.1, 0));  // 左下角
        object_points.push_back(cv::Point3f(0.1, -0.1, 0));   // 右下角
        object_points.push_back(cv::Point3f(0.1, 0.1, 0));    // 右上角
    } else {
        // 大型装甲板的世界坐标
        object_points.push_back(cv::Point3f(-0.2, 0.1, 0));   // 左上角
        object_points.push_back(cv::Point3f(-0.2, -0.1, 0));  // 左下角
        object_points.push_back(cv::Point3f(0.2, -0.1, 0));   // 右下角
        object_points.push_back(cv::Point3f(0.2, 0.1, 0));    // 右上角
    }

    // 解算 PnP
    cv::Mat rvec, tvec;
    cv::solvePnP(object_points, image_points, cameraMatrix, distCoeffs, rvec, tvec, cv::SOLVEPNP_ITERATIVE);

    // 将结果转换为 ROS 2 消息并发布
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = tvec.at<double>(0);
    pose_msg.position.y = tvec.at<double>(1);
    pose_msg.position.z = tvec.at<double>(2);

    // 将旋转向量转换为四元数
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);

    // Convert cv::Mat to Eigen::Matrix3d
    Eigen::Matrix3d rotation_matrix_eigen;
    cv::cv2eigen(rotation_matrix, rotation_matrix_eigen);

    Eigen::Quaterniond q(rotation_matrix_eigen);

    pose_msg.orientation.x = q.x();
    pose_msg.orientation.y = q.y();
    pose_msg.orientation.z = q.z();
    pose_msg.orientation.w = q.w();

    publisher_->publish(pose_msg);
}