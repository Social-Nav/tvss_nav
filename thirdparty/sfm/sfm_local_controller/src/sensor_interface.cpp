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

#include <sfm_local_controller/sensor_interface.h>
#include <tvss_nav/StringStamped.h>       
#include <jsoncpp/json/json.h>            
#include <unordered_map>
#include <string> 
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/imgcodecs.hpp>   // cv::imdecode
#include <opencv2/core.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc.hpp>
#include <limits>
#include <cmath>
namespace Upo {
namespace Navigation {
namespace sfm_controller {

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
SFMSensorInterface::SFMSensorInterface(ros::NodeHandle *n,
                                       tf2_ros::Buffer *tfBuffer,
                                       float robot_max_lin_speed,
                                       float robot_radius, float person_radius,
                                       std::string robot_frame,
                                       std::string odom_frame)
    : nh_(n), n_(), tf_buffer_(tfBuffer), person_radius_(person_radius),
      robot_frame_(robot_frame), odom_frame_(odom_frame) {

  nh_->param("max_obstacle_dist", max_obstacle_dist_, float(3.0));
  nh_->param("naive_goal_time", naive_goal_time_, float(2.0));
  nh_->param("people_velocity", people_velocity_, float(1.0));
nh_->param("camera_optical_frame", camera_frame_, std::string("camera_color_optical_frame"));

cam_info_sub_ = n_.subscribe<sensor_msgs::CameraInfo>(
    "/camera/color/camera_info", 1, &SFMSensorInterface::camInfoCb, this);

label_mask_sub_ = n_.subscribe<sensor_msgs::CompressedImage>(
    "/segmented_image/label_mask/compressed", 1, &SFMSensorInterface::labelMaskCb, this);
class_sub_ = n_.subscribe<tvss_nav::StringStamped>(
      "/instance_class_dict", 1, &SFMSensorInterface::classDictCb, this);

  laser_received_ = false;
  last_laser_ = ros::Time::now();

  std::string laser_topic;
  nh_->param("laser_topic", laser_topic, std::string("scan"));
  std::string people_topic;
  nh_->param("people_topic", people_topic, std::string("people"));
  std::string obs_topic;
  nh_->param("dyn_obs_topic", obs_topic, std::string("obstacles"));
  std::string odom_topic;
  nh_->param("odom_topic", odom_topic, std::string("odom"));
  nh_->param("depth_topic", depth_topic_, std::string("/camera/aligned_depth_to_color/image_raw"));
  depth_sub_ = n_.subscribe<sensor_msgs::Image>(depth_topic_.c_str(), 1, &SFMSensorInterface::depthCb, this);

  // Initialize SFM. Just one agent (the robot)
  agents_.resize(1);
  agents_[0].desiredVelocity = robot_max_lin_speed;
  agents_[0].radius = robot_radius;
  agents_[0].cyclicGoals = false;
  agents_[0].teleoperated = true;
  agents_[0].groupId = -1;

  // ros::NodeHandle n;
  laser_sub_ = n_.subscribe<sensor_msgs::LaserScan>(
      laser_topic.c_str(), 1, &SFMSensorInterface::laserCb, this);

  people_sub_ = n_.subscribe<people_msgs::People>(
      people_topic.c_str(), 1, &SFMSensorInterface::peopleCb, this);

  dyn_obs_sub_ = n_.subscribe<dynamic_obstacle_detector::DynamicObstacles>(
      obs_topic.c_str(), 1, &SFMSensorInterface::dynamicObsCb, this);

  odom_sub_ = n_.subscribe<nav_msgs::Odometry>(
      odom_topic.c_str(), 1, &SFMSensorInterface::odomCb, this);

  points_pub_ = n_.advertise<visualization_msgs::Marker>(
      "/sfm/markers/obstacle_points", 0);
  // sonars ?
  // sonar_sub_= n_.subscribe<sensor_msgs::Range>(
  //    rt.c_str(), 1, boost::bind(&SFMSensorInterface::sonarCb, this, _1));
}
// ========== instance_class_dict 回调 ==========
static bool parseClassDictSimple(const std::string& s,
                                 std::unordered_map<int, std::string>& out) {
  auto is_ws = [](unsigned char c){ return c==' '||c=='\t'||c=='\r'||c=='\n'; };

  out.clear();
  // 粗略预估容量，减少 rehash（可选）
  out.reserve(std::count(s.begin(), s.end(), ':'));

  size_t i = 0, n = s.size();
  auto skip_ws = [&](){ while (i < n && is_ws((unsigned char)s[i])) ++i; };

  skip_ws();
  if (i >= n || s[i] != '{') return false;
  ++i;

  while (true) {
    skip_ws();
    if (i < n && s[i] == '}') { ++i; break; } // 空对象 {}
    if (i >= n || s[i] != '"') return false;  // 期待 "id"
    ++i;

    if (i >= n || !std::isdigit((unsigned char)s[i])) return false;
    size_t k0 = i;
    while (i < n && std::isdigit((unsigned char)s[i])) ++i;
    int id = 0;
    try {
      id = std::stoi(s.substr(k0, i - k0));
    } catch (...) { return false; }
    if (i >= n || s[i] != '"') return false;
    ++i;
    skip_ws();
    if (i >= n || s[i] != ':') return false;
    ++i;
    skip_ws();
    if (i >= n || s[i] != '"') return false;
    ++i;
    std::string name;
    name.reserve(16);
    while (i < n) {
      char c = s[i++];
      if (c == '\\') {
        if (i >= n) return false;
        char esc = s[i++];
        switch (esc) {
          case '"': name.push_back('"'); break;
          case '\\': name.push_back('\\'); break;
          case 'n': name.push_back('\n'); break;
          case 't': name.push_back('\t'); break;
          default:  name.push_back(esc);  break; 
        }
      } else if (c == '"') {
        break;
      } else {
        name.push_back(c);
      }
    }

    out[id] = std::move(name);

    skip_ws();
    if (i < n && s[i] == ',') { ++i; continue; }
    if (i < n && s[i] == '}') { ++i; break; }

    size_t j = i;
    while (j < n && is_ws((unsigned char)s[j])) ++j;
    if (j < n && s[j] == '}') { i = j + 1; break; }

    if (j < n) return false;
    break;
  }
  while (i < n && is_ws((unsigned char)s[i])) ++i;
  return i == n;
}

void SFMSensorInterface::classDictCb(const tvss_nav::StringStamped::ConstPtr& msg) {
  ROS_INFO("Received instance_class_dict message.");
  ROS_INFO("[classDictCb] stamp=%u.%09u frame_id=%s payload_size=%zu",
           msg->header.stamp.sec,
           msg->header.stamp.nsec,
           msg->header.frame_id.c_str(),
           msg->data.size());
  ROS_INFO("Parsing (simple) class dict from /instance_class_dict...");

  const std::string& s = msg->data;

  {
    std::ostringstream os;
    os << "[classDictCb] full dump (" << s.size() << " bytes): ";
    os.write(s.data(), s.size());
    ROS_INFO_STREAM(os.str());
  }

  std::unordered_map<int, std::string> tmp;
  if (!parseClassDictSimple(s, tmp)) {
    ROS_WARN("[classDictCb] simple parse failed, keep old mapping");
    return;
  }

  {
    std::lock_guard<std::mutex> lk(class_map_mtx_);
    class_map_.swap(tmp);
  }
  ROS_INFO("[classDictCb] Processed %zu class mappings", class_map_.size());

  for (const auto& kv : class_map_) {
    ROS_INFO("[classDictCb] id %d -> '%s'", kv.first, kv.second.c_str());
  }
}
void SFMSensorInterface::depthCb(const sensor_msgs::ImageConstPtr& msg) {
  ros::WallTime t0 = ros::WallTime::now();
  cv::Mat depth_m;
  if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
    cv::Mat d16(msg->height, msg->width, CV_16UC1,
                const_cast<uint8_t*>(msg->data.data()), msg->step);
    d16.copyTo(depth_m);
    depth_m.convertTo(depth_m, CV_32FC1, 1.0 / 1000.0);  // mm -> m
  } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
    cv::Mat d32(msg->height, msg->width, CV_32FC1,
                const_cast<uint8_t*>(msg->data.data()), msg->step);
    d32.copyTo(depth_m);
  } else {
    ROS_WARN_THROTTLE(5.0, "Unsupported depth encoding: %s", msg->encoding.c_str());
    return;
  }

  {
    std::lock_guard<std::mutex> g(depth_mtx_);
    last_depth_ = depth_m.clone();
    last_depth_stamp_ = msg->header.stamp;
    depth_ready_ = true;
  }
  ROS_INFO("Depth image received, size: %dx%d", depth_m.cols, depth_m.rows);
  buildAgentsFromGSAM(msg->header.stamp);
  ROS_INFO("Depth callback processed, agents built.");
  ROS_WARN("depthCb+buildAgentsFromGSAM dispatched in %.3f ms", (ros::WallTime::now()-t0).toSec()*1e3);
}
// void SFMSensorInterface::buildAgentsFromGSAM(const ros::Time& stamp) {
  
//   if (!cam_ready_ || camera_frame_.empty()) return;
//   // ROS_INFO("BUILDBUILDBUILDBUILDBUILD: buildAgentsFromGSAM");
//   // 快照（避免持锁耗时）
//   cv::Mat mask, depth;
//   ros::Time mask_stamp;
//   // ROS_INFO("p0000000000000000000000000000000000000000000000000000");
//   {
//     std::lock_guard<std::mutex> lk(mask_mtx_);
//     if (last_label_mask_.empty()){
//       ROS_INFO("No label mask available, skipping agent build.");
//       return;
//     }
//     mask = last_label_mask_.clone();
//     mask_stamp = last_label_mask_stamp_;
//   }
//   // ROS_INFO("11111111111111111111111111111111111111111111111111111111111111111");
//   {
//     std::lock_guard<std::mutex> g(depth_mtx_);
//     if (last_depth_.empty()) return;
//     depth = last_depth_.clone();
//   }
//   // ROS_INFO("2222222222222222222222222222222222222222222222222222222222222222");
//   // 时间对齐（可选，避免错配）
//   if (fabs((stamp - mask_stamp).toSec()) > 0.3) {
//     ROS_DEBUG_THROTTLE(2.0, "GSAM build skipped due to large stamp diff: %.3f s",
//                        fabs((stamp - mask_stamp).toSec()));
//     // 仍可继续，不 return 也行；这里保守直接继续
//   }
//   // ROS_INFO("33333333333333333333333333333333333333333333333333333333");
//   // 尺寸不一致时，把深度 resize 到 mask 尺寸（一般本来就对齐）
//   if (depth.size() != mask.size())
//     cv::resize(depth, depth, mask.size(), 0, 0, cv::INTER_NEAREST);
//   // ROS_INFO("44444444444444444444444444444444444444444444444444");
//   // 先粗扫有哪些 instance id（0 当背景）
//   bool present[256] = {false};
//   for (int y = 0; y < mask.rows; y += 2) {
//     const uint8_t* pm = mask.ptr<uint8_t>(y);
//     for (int x = 0; x < mask.cols; x += 2) {
//       uint8_t iid = pm[x];
//       if (iid) present[iid] = true;
//     }
//   }
//   ROS_INFO("5555555555555555555555555555555555555555555555555");
//   //
//   std::vector<sfm::Agent> agents; agents.reserve(32);

//   for (int iid = 1; iid < 256; ++iid) {
//     if (!present[iid]) continue;

//     // 二值图（该 iid 的像素）
//     cv::Mat bin;
//     cv::compare(mask, iid, bin, cv::CMP_EQ);

//     // 质心
//     cv::Moments m = cv::moments(bin, true);
//     if (m.m00 < 20.0) continue;              // 太小的片段跳过
//     int u = static_cast<int>(m.m10 / m.m00);
//     int v = static_cast<int>(m.m01 / m.m00);
//     if (u < 2 || v < 2 || u >= mask.cols - 2 || v >= mask.rows - 2) continue;

//     // 5×5 深度窗口中位数（米）
//     std::vector<float> zs; zs.reserve(25);
//     for (int dy = -2; dy <= 2; ++dy)
//       for (int dx = -2; dx <= 2; ++dx) {
//         float z = depth.at<float>(v + dy, u + dx);
//         if (std::isfinite(z) && z > 0.1f && z < 15.0f) zs.push_back(z);
//       }
//     if (zs.size() < 5) continue;
//     float z_med = medianInVector(zs);
//     // 像素→相机系 3D
//     cv::Point2d uv(u, v);
//     cv::Point3d ray = cam_model_.projectPixelTo3dRay(uv);
//     cv::Point3d xyz_cam(ray.x * z_med, ray.y * z_med, ray.z * z_med);

//     // 相机系→ odom
//     geometry_msgs::PointStamped pc, po;
//     pc.header.frame_id = camera_frame_;
//     pc.header.stamp    = stamp;             // 用深度帧时间
//     pc.point.x = xyz_cam.x; pc.point.y = xyz_cam.y; pc.point.z = xyz_cam.z;

//     try {
//       tf_buffer_->transform(pc, po, odom_frame_, ros::Duration(0.05));
//     } catch (const tf2::TransformException& e) {
//       ROS_DEBUG("TF failed cam->odom for iid %d: %s", iid, e.what());
//       continue;
//     }

//     // 读取 string 标签（没有就 "unknown"）
//     ROS_INFO("READREADREADREADREAD: iid=%d", iid);
//     std::string cls = "unknown";
//     {
//       std::lock_guard<std::mutex> lk(class_map_mtx_);
//       auto it = class_map_.find(iid);
//       if (it != class_map_.end()) cls = it->second;
//     }
//     ROS_INFO("READREADREADREADREAD: iid=%d, cls=%s", iid, cls.c_str());
//     // 组装 agent（半径统一 person_radius_；只打 string 标签）
//     sfm::Agent ag;
//     ag.id = static_cast<int>(agents.size() + 1);     // 1..N，0 留给机器人
//     ag.position.set(po.point.x, po.point.y);
//     ag.velocity.set(0.0, 0.0);
//     ag.linearVelocity = 0.0;
//     ag.yaw = utils::Angle::fromRadian(0.0);
//     ag.radius = person_radius_;
//     ag.teleoperated = false;
//     ag.desiredVelocity = people_velocity_;
//     ag.groupId = -1;

//     ag.semantic_instance_id = iid;
//     ag.semantic_class = cls;
//     ag.category = cls;                      // 你要的 string 标签

//     // 简单 goal（停在原地；后面你再加预测也行）
//     sfm::Goal g;
//     g.center = ag.position;
//     g.radius = ag.radius;
//     ag.goals.clear();
//     ag.goals.push_back(g);

//     agents.push_back(ag);
//   }
//   ROS_INFO("66666666666666666666666666666666666666666666666666");
//   // 把障碍点塞给每个 agent
//   const std::vector<utils::Vector2d> obs_points = obstacles_;
//   for (auto& a : agents) a.obstacles1 = obs_points;
//   ROS_INFO("7777777777777777777777777777777777777777777777777");
//   // 回写全局 agents_：0=机器人，1..=gsam
//   {
//     std::lock_guard<std::mutex> g(agents_mutex_);
//     agents_.resize(agents.size() + 1);
//     // 保留 agents_[0]（机器人位姿由 odomCb 在更新）
//     agents_[0].obstacles1 = obs_points;
//     for (size_t i = 1; i < agents_.size(); ++i) agents_[i] = agents[i - 1];
//   }
// }

void SFMSensorInterface::camInfoCb(const sensor_msgs::CameraInfo::ConstPtr& msg) {
  cam_model_.fromCameraInfo(*msg);
  cam_ready_ = true;
  // ROS_INFO_ONCE("camera info received, camera model ready");
}

void SFMSensorInterface::labelMaskCb(const sensor_msgs::CompressedImage::ConstPtr& msg) {
  ros::WallTime t0 = ros::WallTime::now();
  // ROS_INFO("labelMaskCb called, data size: %zu", msg->data.size());
  try {
    cv::Mat buf(1, msg->data.size(), CV_8UC1, const_cast<uint8_t*>(msg->data.data()));
    cv::Mat mat = cv::imdecode(buf, cv::IMREAD_UNCHANGED);  
    ROS_INFO("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL");
    if (mat.empty()){
      ROS_INFO("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP");
      return;
    }

    if (mat.type() != CV_8UC1) {
      cv::Mat gray;
      if (mat.channels() == 3)
        cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
      else if (mat.channels() == 4)
        cv::cvtColor(mat, gray, cv::COLOR_BGRA2GRAY);
      else
        gray = mat;  
      mat = gray;
    }
    ROS_INFO("SSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS");
    {
      std::lock_guard<std::mutex> lk(mask_mtx_);
      last_label_mask_ = mat.clone();
      last_label_mask_stamp_ = msg->header.stamp;  // 记录 mask 时间戳
    }
    ROS_INFO_ONCE("label mask received");
  } catch (const std::exception& e) {
    ROS_WARN("labelMaskCb decode failed: %s", e.what());
  }
  ROS_WARN("labelMaskCb took %.3f ms", (ros::WallTime::now()-t0).toSec()*1e3);
}

bool SFMSensorInterface::projectToImage(const geometry_msgs::Point& p_odom, int& u, int& v) {
  if (!cam_ready_) return false;
  if (camera_frame_.empty()) return false;

  geometry_msgs::PointStamped pin, pout;
  pin.header.frame_id = odom_frame_;
  pin.header.stamp    = ros::Time(0);
  pin.point           = p_odom;

  try {
    tf_buffer_->transform(pin, pout, camera_frame_, ros::Duration(0.05));
  } catch (const tf2::TransformException& e) {
    // ROS_DEBUG("projectToImage TF failed: %s", e.what());
    return false;
  }

  if (pout.point.z <= 0.0) return false;  

  cv::Point3d xyz(pout.point.x, pout.point.y, pout.point.z);
  cv::Point2d uv = cam_model_.project3dToPixel(xyz);

  u = static_cast<int>(std::round(uv.x));
  v = static_cast<int>(std::round(uv.y));

  std::lock_guard<std::mutex> lk(mask_mtx_);
  if (last_label_mask_.empty()) return false;

  return (u >= 0 && v >= 0 &&
          u < last_label_mask_.cols &&
          v < last_label_mask_.rows);
}

int SFMSensorInterface::instanceIdAt(int u, int v) {
  std::lock_guard<std::mutex> lk(mask_mtx_);
  if (last_label_mask_.empty()){
    ROS_INFO("EMPTY LABEL MASK, RETURNING -1ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss");
    return -1;
  }
  return static_cast<int>(last_label_mask_.at<uint8_t>(v, u));
}

/**
 * @brief  Destructor class
 */
SFMSensorInterface::~SFMSensorInterface() {}

/**
 * @brief  Callback to process the laser scan sensory input.
 * @param laser laserScan message to be processed
 */
void SFMSensorInterface::laserCb(
    const sensor_msgs::LaserScan::ConstPtr &laser) {

  laser_received_ = true;
  last_laser_ = ros::Time::now();

  // ROS_INFO_ONCE("laser received");

  std::vector<utils::Vector2d> points;
  float angle = laser->angle_min;

  for (unsigned int i = 0; i < laser->ranges.size(); i++) {

    if (laser->ranges[i] < max_obstacle_dist_) {

      utils::Vector2d point(laser->ranges[i] * cos(angle),
                            laser->ranges[i] * sin(angle));
      points.push_back(point);
    }
    angle += laser->angle_increment;
    // alpha += angle_inc;
  }

  if (points.empty()) {
    obstacles_ = points;
    return;
  }

  // Transform points to
  // odom_frame_ if necessary
  if (laser->header.frame_id != odom_frame_) {
    geometry_msgs::PointStamped out;
    for (unsigned int i = 0; i < points.size(); i++) {
      geometry_msgs::PointStamped in;
      in.header.frame_id = laser->header.frame_id;
      in.header.stamp = ros::Time(0);
      in.point.x = points[i].getX();
      in.point.y = points[i].getY();
      in.point.z = 0.0;
      try {
        geometry_msgs::PointStamped out =
            tf_buffer_->transform(in, odom_frame_);
        points[i].setX(out.point.x);
        points[i].setY(out.point.y);

      } catch (tf2::TransformException &ex) {
        ROS_WARN("Could NOT transform "
                 "laser point to %s: "
                 "%s",
                 robot_frame_.c_str(), ex.what());
        return;
      }
    }
  }

  // Now check if the points
  // belong to a person or an
  // dynamic obstacle

  // // transform people positions to
  // // robot frame
  // // people_mutex_.lock();
  // people_msgs::People people = people_;
  // // people_mutex_.unlock();

  // std::vector<geometry_msgs::Point> people_points;
  // if (people.header.frame_id != odom_frame_) {
  //   geometry_msgs::PointStamped person_point;
  //   person_point.header = people.header;

  //   for (auto person : people.people) {
  //     person_point.point = person.position;
  //     person_point.header.stamp = ros::Time(0);
  //     try {
  //       geometry_msgs::PointStamped p_point =
  //           tf_buffer_->transform(person_point, odom_frame_);
  //       people_points.push_back(p_point.point);
  //     } catch (tf2::TransformException &ex) {
  //       ROS_WARN("Could NOT transform "
  //                "person point to %s: "
  //                "%s",
  //                odom_frame_.c_str(), ex.what());
  //       return;
  //     }
  //   }
  // } else {
  //   for (auto person : people.people) {
  //     people_points.push_back(person.position);
  //   }
  // }
  // // Remove the points in the
  // // people radius
  // std::vector<utils::Vector2d> points_aux;
  // for (utils::Vector2d p : points) {
  //   bool remove = false;
  //   for (auto person : people_points) {
  //     float dx = p.getX() - person.x;
  //     float dy = p.getY() - person.y;
  //     float d = std::hypotf(dx, dy);
  //     if (d <= person_radius_) {
  //       remove = true;
  //       break;
  //     }
  //   }
  //   if (!remove)
  //     points_aux.push_back(p);
  // }
  // points.clear();
  // points = points_aux;
  // // we can have one point per
  // // sector as much
  // if (points.empty()) {
  //   obstacles_ = points;
  //   return;
  // }

  // // transform dynamic obstacles
  // // positions to robot frame
  // // obs_mutex_.lock();
  // dynamic_obstacle_detector::DynamicObstacles obstacles = dyn_obs_;
  // // obs_mutex_.unlock();

  // std::vector<geometry_msgs::Point> ob_points;
  // if (obstacles.header.frame_id != odom_frame_) {
  //   geometry_msgs::PointStamped ob_point;
  //   ob_point.header = obstacles.header;
  //   // ob_point.stamp = ros::Time();
  //   for (auto obstacle : obstacles.obstacles) {
  //     ob_point.point = obstacle.position;
  //     ob_point.header.stamp = ros::Time(0);
  //     try {
  //       geometry_msgs::PointStamped o_point =
  //           tf_buffer_->transform(ob_point, odom_frame_);
  //       ob_points.push_back(o_point.point);
  //     } catch (tf2::TransformException &ex) {
  //       ROS_WARN("SFM: Could NOT transform "
  //                "obstacle point to %s: "
  //                "%s",
  //                odom_frame_.c_str(), ex.what());
  //       return;
  //     }
  //   }
  // }
  // // Remove the points in the
  // // object radius (approximated
  // // by person radius because we
  // // do not know the radius)
  // points_aux.clear();
  // for (utils::Vector2d p : points) {
  //   bool remove = false;
  //   for (auto ob : ob_points) {
  //     float dx = p.getX() - ob.x;
  //     float dy = p.getY() - ob.y;
  //     float d = std::hypotf(dx, dy);
  //     if (d <= person_radius_) {
  //       remove = true;
  //       break;
  //     }
  //   }
  //   if (!remove)
  //     points_aux.push_back(p);
  // }
  // points.clear();
  // points = points_aux;

  obstacles_ = points;
  publish_obstacle_points(points);
}

/**
 * @brief  publish the transformed points in RViz
 * @param points vector with the coordinates of the points
 * @return none
 */
void SFMSensorInterface::publish_obstacle_points(
    const std::vector<utils::Vector2d> &points) {

  visualization_msgs::Marker m;
  m.header.frame_id = odom_frame_; // robot_frame_
  m.header.stamp = ros::Time::now();
  m.ns = "sfm_obstacle_points";
  m.type = visualization_msgs::Marker::POINTS;
  m.action = visualization_msgs::Marker::ADD;
  m.pose.orientation.x = 0.0;
  m.pose.orientation.y = 0.0;
  m.pose.orientation.z = 0.0;
  m.pose.orientation.w = 1.0;
  m.scale.x = 0.15;
  m.scale.y = 0.15;
  m.scale.z = 0.15;
  m.color.r = 1.0;
  m.color.g = 0.0;
  m.color.b = 0.0;
  m.color.a = 1.0;
  m.id = 1000;
  m.lifetime = ros::Duration(0.3);
  // printf("Published Obstacles: ");
  for (utils::Vector2d p : points) {
    geometry_msgs::Point pt;
    pt.x = p.getX();
    pt.y = p.getY();
    // printf("x: %.2f, y: %.2f -", pt.x, pt.y);
    pt.z = 0.2;
    m.points.push_back(pt);
  }
  // printf("\n");
  points_pub_.publish(m);
}

/**
 * @brief  Callback to process the moving obstacles detected in the robot
 * vecinity.
 * @param obs messages with the obstacles to be processed.
 */
void SFMSensorInterface::dynamicObsCb(
    const dynamic_obstacle_detector::DynamicObstacles::ConstPtr &obs) {
  // ROS_INFO("Dyamic obs receivedddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
  dyn_obs_ = *obs;

  std::vector<sfm::Agent> agents;
  geometry_msgs::PointStamped ps;

  for (unsigned i = 0; i < obs->obstacles.size(); i++) {
    sfm::Agent ag;

    // 位姿到 odom
    ps.header.frame_id = obs->header.frame_id;
    ps.header.stamp    = ros::Time(0);
    ps.point           = obs->obstacles[i].position;
    if (obs->header.frame_id != odom_frame_) {
      try { ps = tf_buffer_->transform(ps, odom_frame_); }
      catch (tf2::TransformException &ex) { ROS_WARN("No transform %s", ex.what()); }
    }
    ag.position.set(ps.point.x, ps.point.y);

    // 语义：投影到图像 → 读 instance_id → 查类别
    // int u_px = -1, v_px = -1;
    // if (projectToImage(ps.point, u_px, v_px)) {
    //   const int iid = instanceIdAt(u_px, v_px);
    //   ag.semantic_instance_id = iid;
    //   std::string cls = "unknown";
    //   if (iid >= 0) {
    //     std::lock_guard<std::mutex> lk(class_map_mtx_);
    //     auto it = class_map_.find(iid);
    //     if (it != class_map_.end()) cls = it->second;
    //   }
    //   ag.semantic_class = cls;
    //   ag.category       = cls;  // 兼容旧字段
    // }

    // 速度到 odom
    geometry_msgs::Vector3 velocity;
    velocity.x = obs->obstacles[i].velocity.x;
    velocity.y = obs->obstacles[i].velocity.y;
    geometry_msgs::Vector3 localV =
        SFMSensorInterface::transformVector(velocity, obs->header.frame_id, odom_frame_);
    ag.yaw = utils::Angle::fromRadian(std::atan2(localV.y, localV.x));
    ag.velocity.set(localV.x, localV.y);
    ag.linearVelocity = ag.velocity.norm();

    // 其它属性
    ag.radius = person_radius_;
    ag.teleoperated = false;

    // 朴素目标
    ag.goals.clear();
    {
      sfm::Goal naiveGoal;
      utils::Vector2d vgoal = ag.position + naive_goal_time_ * ag.velocity;
      naiveGoal.center.set(vgoal.getX(), vgoal.getY());
      naiveGoal.radius = person_radius_;
      ag.goals.push_back(naiveGoal);
    }
    ag.desiredVelocity = people_velocity_;
    ag.groupId = -1;

    // **SFM 内部 id：使用顺序号，不要用语义 instance id**
    ag.id = static_cast<int>(i + 1);

    agents.push_back(ag);
  }

  // 障碍点
  const std::vector<utils::Vector2d> obs_points = obstacles_;
  for (auto &a : agents) a.obstacles1 = obs_points;

  // 回写全局 agents_
  {
    std::lock_guard<std::mutex> g(agents_mutex_);
    agents_.resize(agents.size() + 1);
    agents_[0].obstacles1 = obs_points;   // 机器人
    for (size_t i = 1; i < agents_.size(); ++i) agents_[i] = agents[i - 1];
  }
}


/**
 * @brief  Callback to process the people detected in the robot vecinity.
 * @param people message with the people to be processed
 */
void SFMSensorInterface::peopleCb(const people_msgs::People::ConstPtr &people) {
  // ROS_INFO("People receivedddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
  people_ = *people;

  std::vector<sfm::Agent> agents;
  geometry_msgs::PointStamped ps;

  for (unsigned i = 0; i < people->people.size(); i++) {
    sfm::Agent ag;

    // 位姿到 odom
    ps.header.frame_id = people->header.frame_id;
    ps.header.stamp    = ros::Time(0);
    ps.point           = people->people[i].position;
    if (people->header.frame_id != odom_frame_) {
      try { ps = tf_buffer_->transform(ps, odom_frame_); }
      catch (tf2::TransformException &ex) { ROS_WARN("No transform %s", ex.what()); }
    }
    ag.position.set(ps.point.x, ps.point.y);

    // 速度到 odom
    geometry_msgs::Vector3 velocity;
    velocity.x = people->people[i].velocity.x;
    velocity.y = people->people[i].velocity.y;
    geometry_msgs::Vector3 localV =
        SFMSensorInterface::transformVector(velocity, people->header.frame_id, odom_frame_);
    ag.yaw = utils::Angle::fromRadian(std::atan2(localV.y, localV.x));
    ag.velocity.set(localV.x, localV.y);
    ag.linearVelocity = ag.velocity.norm();

    // 语义：投影到图像 → 读 instance_id → 查类别
    // int u_px = -1, v_px = -1;
    // if (projectToImage(ps.point, u_px, v_px)) {
    //   // ROS_INFO("HAHAHAHAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH");
    //   const int iid = instanceIdAt(u_px, v_px);
    //   ag.semantic_instance_id = iid;
    //   std::string cls = "unknown";
    //   if (iid >= 0) {
    //     std::lock_guard<std::mutex> lk(class_map_mtx_);
    //     auto it = class_map_.find(iid);
    //     if (it != class_map_.end()) cls = it->second;
    //     // ROS_INFO("HOHOHOHOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO");
    //   }
    //   ag.semantic_class = cls;
    //   ag.category       = cls;  // 兼容旧字段
    // }

    // 朴素目标
    ag.goals.clear();
    {
      sfm::Goal naiveGoal;
      utils::Vector2d vgoal = ag.position + naive_goal_time_ * ag.velocity;
      naiveGoal.center.set(vgoal.getX(), vgoal.getY());
      naiveGoal.radius = person_radius_;
      ag.goals.push_back(naiveGoal);
    }
    ag.desiredVelocity = people_velocity_;
    ag.groupId = -1;

    // **SFM 内部 id：使用顺序号**
    ag.id = static_cast<int>(i + 1);

    agents.push_back(ag);
  }

  // 障碍点
  const std::vector<utils::Vector2d> obs_points = obstacles_;
  for (auto &a : agents) a.obstacles1 = obs_points;

  // 回写全局 agents_
  {
    std::lock_guard<std::mutex> g(agents_mutex_);
    agents_.resize(agents.size() + 1);
    agents_[0].obstacles1 = obs_points;   // 机器人
    for (size_t i = 1; i < agents_.size(); ++i) agents_[i] = agents[i - 1];
  }
}


/**
 * @brief  Callback to process the odometry messages with the robot movement.
 * @param odom messages with the obstacles to be processed.
 */
void SFMSensorInterface::odomCb(const nav_msgs::Odometry::ConstPtr &odom) {
  // ROS_INFO_ONCE("Odom received");
  // ROS_INFO("Odom Callback triggered!");

  // if (odom->header.frame_id != odom_frame_) {
  //   ROS_INFO("Odometry frame is %s, it should be %s",
  //            odom->header.frame_id.c_str(), odom_frame_.c_str());
  //   // return;
  // }

  agents_mutex_.lock();
  sfm::Agent agent = agents_[0];
  agents_mutex_.unlock();

  agent.position.set(odom->pose.pose.position.x, odom->pose.pose.position.y);
  agent.yaw = utils::Angle::fromRadian(tf::getYaw(odom->pose.pose.orientation));

  agent.linearVelocity =
      std::sqrt(odom->twist.twist.linear.x * odom->twist.twist.linear.x +
                odom->twist.twist.linear.y * odom->twist.twist.linear.y);
  agent.angularVelocity = odom->twist.twist.angular.z;

  // The velocity in the odom messages is in the robot local frame!!!
  geometry_msgs::Vector3 velocity;
  velocity.x = odom->twist.twist.linear.x;
  velocity.y = odom->twist.twist.linear.y;

  geometry_msgs::Vector3 localV =
      SFMSensorInterface::transformVector(velocity, robot_frame_, odom_frame_);
  agent.velocity.set(localV.x, localV.y);

  // Update agent[0] (the robot) with odom.
  agents_mutex_.lock();
  agents_[0] = agent;
  agents_mutex_.unlock();
}

// void SFMSensorInterface::updateAgents() {
// Get odom
//}

/**
 * @brief  returns the vector of sfm agents
 * @return agents vector
 */
std::vector<sfm::Agent> SFMSensorInterface::getAgents() {
  agents_mutex_.lock();
  std::vector<sfm::Agent> agents = agents_;
  agents_mutex_.unlock();
  return agents;
}

/**
 * @brief  Tranform a coordinate vector from one frame to another
 * @param vector coordinate vector in the origin frame
 * @param from string with the name of the origin frame
 * @param to string with the name of the target frame
 * @return coordinate vector in the target frame
 */
geometry_msgs::Vector3
SFMSensorInterface::transformVector(geometry_msgs::Vector3 &vector,
                                    std::string from, std::string to) {
  geometry_msgs::Vector3 nextVector;

  geometry_msgs::Vector3Stamped v;
  geometry_msgs::Vector3Stamped nv;

  v.header.frame_id = from;

  // we'll just use the most recent transform available for our simple example
  v.header.stamp = ros::Time(0);

  // just an arbitrary point in space
  v.vector.x = vector.x;
  v.vector.y = vector.y;
  v.vector.z = 0.0;

  try {
    nv = tf_buffer_->transform(v, to);
  } catch (tf2::TransformException &ex) {
    ROS_WARN("No transform %s", ex.what());
  }

  nextVector.x = nv.vector.x;
  nextVector.y = nv.vector.y;
  nextVector.z = 0.0;

  return nextVector;
}

} // namespace sfm_controller
} // namespace Navigation
} // namespace Upo