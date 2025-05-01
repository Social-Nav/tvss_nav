#pragma once
#include <opencv2/core.hpp>

namespace tvss_nav {

inline std::pair<int, int> projectToImage(float x, float y, float z,
                                          const cv::Mat& K, const cv::Mat& D)
{
    std::vector<cv::Point3f> object_points = { cv::Point3f(x, y, z) };
    std::vector<cv::Point2f> image_points;

    cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);

    cv::projectPoints(object_points, rvec, tvec, K, D, image_points);

    int u = static_cast<int>(image_points[0].x);
    int v = static_cast<int>(image_points[0].y);
    return {u, v};
}


} // namespace tvss_nav
