#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tf/transform_datatypes.h>

namespace tvss_nav {

inline std::pair<int, int> projectToImage(float x, float y, float z, const cv::Mat& K)
{
    float fx = K.at<double>(0, 0);
    float fy = K.at<double>(1, 1);
    float cx = K.at<double>(0, 2);
    float cy = K.at<double>(1, 2);
    int u = static_cast<int>((x / z) * fx + cx);
    int v = static_cast<int>((y / z) * fy + cy);
    return {u, v};
}

inline Eigen::Matrix4f convertTfToMatrix(const tf::StampedTransform& tf)
{
    Eigen::Matrix4f mat = Eigen::Matrix4f::Identity();
    tf::Matrix3x3 rot = tf.getBasis();
    tf::Vector3 trans = tf.getOrigin();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) mat(i, j) = rot[i][j];
        mat(i, 3) = trans[i];
    }
    return mat;
}

} // namespace tvss_nav
