/**
 * Class to adapt the sensory input and detections to be used by the
 * sfm_local_controller
 *
 * Author: Noé Pérez-Higueras
 * Service Robotics Lab, Pablo de Olavide University 2021
 *
 * Software License Agreement (BSD License)
 *
 */

#ifndef SFMCONTROLLER_SENSOR_H_
#define SFMCONTROLLER_SENSOR_H_

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Vector3.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Range.h>
#include <tf/tf.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
// sensor input for obstacles
#include <sensor_msgs/LaserScan.h>
// Detection input for social navigation
#include <dynamic_obstacle_detector/DynamicObstacles.h>
#include <people_msgs/People.h>
// Social Force Model
#include <lightsfm/angle.hpp>
#include <lightsfm/sfm.hpp>
#include <lightsfm/vector2d.hpp>

#include <math.h>
#include <mutex>

#include <tvss_nav/StringStamped.h>   
#include <unordered_map>
#include <string>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/CompressedImage.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
namespace Upo {
namespace Navigation {
namespace sfm_controller {

class SFMSensorInterface {
public:
  /**
   * @brief  Default constructor
   * @param n pointer to a ros node handle to publish to topics
   * @param tfBuffer Pointer to tf2 buffer
   * @param robot_max_lin_speed the maximum robot linear speed [m/s]
   * @param robot_radius the radius of the robot circunference [m]
   * @param person_radius the approximated radius of the people body
   * @param robot_frame the coordinate frame of the robot base
   * @param odom_frame the coordinate frame of the robot odometry
   **/
  SFMSensorInterface(ros::NodeHandle *n, tf2_ros::Buffer *tfBuffer,
                     float robot_max_lin_speed, float robot_radius,
                     float person_radius, std::string robot_frame,
                     std::string odom_frame);

  /**
   * @brief  Destructor class
   */
  ~SFMSensorInterface();

  /**
   * @brief  Callback to process the laser scan sensory input.
   * @param laser laserScan message to be processed
   */
  void laserCb(const sensor_msgs::LaserScan::ConstPtr &laser);

  /**
   * @brief  Callback to process the people detected in the robot vecinity.
   * @param people message with the people to be processed
   */
  void peopleCb(const people_msgs::People::ConstPtr &people);

  /**
   * @brief  Callback to process the moving obstacles detected in the robot
   * vecinity.
   * @param obs messages with the obstacles to be processed.
   */
  void dynamicObsCb(
      const dynamic_obstacle_detector::DynamicObstacles::ConstPtr &obs);

  /**
   * @brief  Callback to process the odometry messages with the robot movement.
   * @param odom messages with the obstacles to be processed.
   */
  void odomCb(const nav_msgs::Odometry::ConstPtr &odom);

  /**
   * @brief  Tranform a coordinate vector from one frame to another
   * @param vector coordinate vector in the origin frame
   * @param from string with the name of the origin frame
   * @param to string with the name of the target frame
   * @return coordinate vector in the target frame
   */
  geometry_msgs::Vector3 transformVector(geometry_msgs::Vector3 &vector,
                                         std::string from, std::string to);

  /**
   * @brief  returns the vector of sfm agents
   * @return agents vector
   */
  std::vector<sfm::Agent> getAgents();

  void setMaxLinVel(float max_lin_vel) {
    agents_mutex_.lock();
    agents_[0].desiredVelocity = max_lin_vel;
    agents_mutex_.unlock();
  }

private:
  /**
   * @brief  publish the transformed points in RViz
   * @param points vector with the coordinates of the points
   * @return none
   */
  void publish_obstacle_points(const std::vector<utils::Vector2d> &points);

  // void updateAgents();

  ros::NodeHandle *nh_; // Pointer to the node node handle
  ros::NodeHandle n_;
  tf2_ros::Buffer *tf_buffer_; // Pointer to the tfBuffer created in the node

  ros::Subscriber odom_sub_;
  std::mutex odom_mutex_;
  ros::Subscriber laser_sub_, people_sub_, dyn_obs_sub_, sonar_sub_;
  ros::Publisher points_pub_;

    // —— 类别订阅与缓存 —— 
  ros::Subscriber class_sub_;
  std::unordered_map<int, std::string> class_map_;
  std::mutex class_map_mtx_;

  ros::Subscriber cam_info_sub_;     // 新增：相机内参
ros::Subscriber label_mask_sub_;   // 新增：实例ID图（压缩PNG）

image_geometry::PinholeCameraModel cam_model_;
bool cam_ready_ = false;
std::string camera_frame_;   // 如 "camera_color_optical_frame"

cv::Mat last_label_mask_;    // 单通道 uint8，像素值=instance_id
std::mutex mask_mtx_;
ros::Time last_label_mask_stamp_;  // ★ 记录 mask 时间戳

// 订阅回调
void classDictCb(const tvss_nav::StringStamped::ConstPtr& msg);
void camInfoCb(const sensor_msgs::CameraInfo::ConstPtr& msg);
void labelMaskCb(const sensor_msgs::CompressedImage::ConstPtr& msg);

// 工具函数：投影 & 读像素
bool projectToImage(const geometry_msgs::Point& p_odom, int& u, int& v);
int  instanceIdAt(int u, int v);
// ---------- GSAM2: 深度 + 分割 生成 agents ----------
  ros::Subscriber depth_sub_;
 std::string depth_topic_;
  cv::Mat last_depth_;                   // 深度图（float米）
  std::mutex depth_mtx_;
  ros::Time last_depth_stamp_;
  bool depth_ready_ = false;

  struct TrackState {
    geometry_msgs::Point world_pt;
    ros::Time stamp;
  };
  std::unordered_map<int, TrackState> track_map_; // key: semantic_instance_id

  // 类别 → 半径（可参数化）
  std::unordered_map<std::string, double> class_radius_map_;

  void depthCb(const sensor_msgs::ImageConstPtr& msg);
  void buildAgentsFromGSAM(const ros::Time& stamp); // = gsamCb
  static inline float medianInVector(std::vector<float>& v);

  std::vector<sfm::Agent> agents_; // 0: robot, 1..: Others
  // sfm::Agent robot_agent_;
  std::mutex agents_mutex_;
  std::vector<utils::Vector2d> obstacles_;

  bool laser_received_;
  ros::Time last_laser_;

  people_msgs::People people_;
  std::mutex people_mutex_;
  dynamic_obstacle_detector::DynamicObstacles dyn_obs_;
  std::mutex obs_mutex_;

  float person_radius_;
  float naive_goal_time_;
  float people_velocity_;

  float max_obstacle_dist_;

  std::string robot_frame_;
  std::string odom_frame_;

  // bool use_static_map_;
  // sfm::RosMap *static_map_;
};

} // namespace sfm_controller
} // namespace Navigation
} // namespace Upo

#endif