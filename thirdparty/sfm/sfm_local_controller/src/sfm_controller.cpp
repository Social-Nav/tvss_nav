/**
 * Local robot controller based on the Social Force Model
 *
 * Author: Noé Pérez-Higueras
 * Service Robotics Lab, Pablo de Olavide University 2021
 *
 * Software License Agreement (BSD License)
 *
 */

#include <sfm_local_controller/sfm_controller.h>
#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/PointStamped.h>
#include <tvss_nav/SemanticInstanceArray.h> 
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
// #include <sfm_local_controller/include/lightsfm/sfm.hpp>

namespace Upo {
namespace Navigation {
namespace sfm_controller {

/**
 * @brief  Default constructor
 */
SFMController::SFMController()
    : max_lin_vel_(0.7), min_lin_vel_(0.1), max_vel_theta_(0.6),
      min_vel_theta_(0.1), min_rot_in_place_(0.4), max_lin_acc_(1.0),
      max_theta_acc_(1.0), local_goal_dist_(1.0), sfm_goal_weight_(2.0),
      sfm_obstacle_weight_(10.0), sfm_people_weight_(12.0), robot_radius_(0.35),
      person_radius_(0.35), goal_tolerance_(0.15), yaw_tolerance_(0.3),
      robot_frame_(std::string("base_link")),
      planner_frame_(std::string("odom")), a_(2.0) {}

/**
 * @brief  Constructor with parameters
 */
SFMController::SFMController(
    ros::NodeHandle *n, tf2_ros::Buffer *tfBuffer,
    SFMSensorInterface *sensor_iface, CollisionChecker *coll_check,
    float max_lin_vel, float min_lin_vel, float max_vel_theta,
    float min_vel_theta, float min_rot_in_place, float max_lin_acc,
    float max_theta_acc, float local_goal_dist, float sfm_goal_weight,
    float sfm_obstacle_weight, float sfm_people_weight, float robot_radius,
    float person_radius, float goal_tolerance, float yaw_tolerance,
    std::string robot_frame, std::string controller_frame, float a)
    : max_lin_vel_(max_lin_vel), min_lin_vel_(min_lin_vel),
      max_vel_theta_(max_vel_theta), min_vel_theta_(min_vel_theta),
      min_rot_in_place_(min_rot_in_place), max_lin_acc_(max_lin_acc),
      max_theta_acc_(max_theta_acc), local_goal_dist_(local_goal_dist),
      sfm_goal_weight_(sfm_goal_weight),
      sfm_obstacle_weight_(sfm_obstacle_weight),
      sfm_people_weight_(sfm_people_weight), robot_radius_(robot_radius),
      person_radius_(person_radius), goal_tolerance_(goal_tolerance),
      yaw_tolerance_(yaw_tolerance), robot_frame_(robot_frame),
      planner_frame_(controller_frame), a_(a) {
  // Initialize robot agent
  // return true;
  robot_.params.forceFactorObstacle = sfm_obstacle_weight_;
  robot_.params.forceFactorSocial = sfm_people_weight_;

  std::cout << std::endl
            << "SFM_CONTROLLER INITIATED:" << std::endl
            << "max_lin_acc: " << max_lin_acc_ << std::endl
            << "max_rot_acc: " << max_theta_acc_ << std::endl
            << "min_lin_vel: " << min_lin_vel_ << std::endl
            << "max_lin_vel: " << max_lin_vel_ << std::endl
            << "min_rot_vel: " << min_vel_theta_ << std::endl
            << "max_rot_vel: " << max_vel_theta_ << std::endl
            << "local_goal_dist: " << local_goal_dist_ << std::endl
            << "goal_tolerance: " << goal_tolerance_ << std::endl
            << "robot_radius: " << robot_radius_ << std::endl
            << "person_radius: " << person_radius_ << std::endl
            << "robot_frame: " << robot_frame_ << std::endl
            << "planner_frame: " << planner_frame_ << std::endl
            << "sfm_goal_weight: " << sfm_goal_weight_ << std::endl
            << "sfm_obstacle_weight: " << sfm_obstacle_weight_ << std::endl
            << "sfm_social_weight: " << sfm_people_weight_ << std::endl
            << std::endl;

  // Advertise SFM related markers
  robot_markers_pub_ = n->advertise<visualization_msgs::MarkerArray>(
      "/sfm/markers/robot_forces", 1);

  // Adverstise SFM local goal
  sfm_goal_pub_ =
      n->advertise<visualization_msgs::Marker>("/sfm/markers/goal", 1);
  // subgoal_pub_ = n->advertise<visualization_msgs::Marker>("/sfm/markers/subgoal", 1);
  // Initialize sensor interface
  sensor_iface_ = sensor_iface;

  // Initialize collision checker
  collision_checker_ = coll_check;

  last_command_time_ = ros::Time::now();
  goal_reached_ = false;
  rotate_ = false;

  // tf_buffer_ = tfBuffer;
  // n->param("sfm/sem_match_radius", match_radius_, 1.0);  
  // n->param("sfm/sem_frame", sem_frame_, std::string("map"));
  // sem_sub_ = n->subscribe("/instance_array", 1, &SFMController::semCb, this);
}
// void SFMController::semCb(const tvss_nav::SemanticInstanceArray::ConstPtr& msg)
// {
//   sem_objs_.clear();

//   // 语义消息的源坐标系（若空，则用参数 sem_frame_）
//   const std::string src_frame = msg->header.frame_id.empty() ? sem_frame_ : msg->header.frame_id;
//   ROS_INFO("--------------------------------------------------------------Processing semantic instances in frame");
//   for (const auto& inst : msg->instances) {
//     ROS_INFO("000000000000000000000000000000000000000000000000000000000000000000000000000Processing semantic instance: %s, id: %d", inst.class_name.c_str(), inst.instance_id);
//     // 1) 从实例点云中求质心
//     geometry_msgs::Point c_src;
//     if (!centroidFromCloud(inst.cloud, c_src)) {
//       continue;
//     }
//     ROS_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
//     // 2) 统一变换到 sem_frame_，便于和 agents_ 匹配
//     geometry_msgs::Point c_sem;
//     if (!transformPoint(c_src, src_frame, sem_frame_, msg->header.stamp, c_sem)) {
//       continue;
//     }
//     ROS_INFO("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY");
//     // 3) 记录（保留 GSAM2 原始标签与实例 id）
//     SemObj s;
//     s.id    = inst.instance_id;  // ✅ 保留原 instance_id
//     s.cls   = inst.class_name;   // ✅ 保留原 class_name（如 "person"、"doctor" 等）
//     s.p     = c_sem;
//     s.stamp = msg->header.stamp;
//     sem_objs_.push_back(std::move(s));
//   }
// }

// bool SFMController::centroidFromCloud(const sensor_msgs::PointCloud2& cloud,
//                                       geometry_msgs::Point& out)
// {
//   ROS_INFO("Computing centroid from point cloud with %zu points", cloud.width * cloud.height);
//   if (cloud.width * cloud.height == 0){
//     ROS_INFO("No points in the point cloud, cannot compute centroid.");
//     return false;
//   }

//   sensor_msgs::PointCloud2ConstIterator<float> it_x(cloud, "x");
//   sensor_msgs::PointCloud2ConstIterator<float> it_y(cloud, "y");
//   sensor_msgs::PointCloud2ConstIterator<float> it_z(cloud, "z");

//   double sx = 0.0, sy = 0.0, sz = 0.0;
//   size_t n = cloud.width * cloud.height, cnt = 0;
//   for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z) {
//     const float x = *it_x, y = *it_y, z = *it_z;
//     if (std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
//       sx += x; sy += y; sz += z; ++cnt;
//     }
//   }
//   if (cnt == 0){
//     ROS_INFO("No valid points found in the point cloud.");
//     return false;
//   }

//   out.x = sx / cnt;
//   out.y = sy / cnt;
//   out.z = sz / cnt;
//   return true;
// }

// bool SFMController::transformPoint(const geometry_msgs::Point& in,
//                                    const std::string& from_frame,
//                                    const std::string& to_frame,
//                                    const ros::Time& stamp,
//                                    geometry_msgs::Point& out) const
// {
//   ROS_INFO("Transforming point from %s to %s at time %f",
//            from_frame.c_str(), to_frame.c_str(), stamp.toSec());
//   geometry_msgs::PointStamped pin, pout;
//   pin.header.frame_id = from_frame;
//   pin.header.stamp    = stamp;
//   pin.point           = in;
//   try {
//     tf_buffer_->transform(pin, pout, to_frame, ros::Duration(0.05));
//     out = pout.point;
//     return true;
//   } catch (const tf2::TransformException& e) {
//     ROS_WARN_THROTTLE(1.0, "transformPoint %s -> %s failed: %s",
//                       from_frame.c_str(), to_frame.c_str(), e.what());
//     return false;
//   }
// }

// void SFMController::matchAndTagAgents()
// {
//   ROS_INFO("Matching and tagging agents with semantic objects...");
//   if (agents_.empty()) return;

//   // 如果没有任何语义对象，就把所有 agent 置回 unknown（你也可以选择保留旧值）
//   if (sem_objs_.empty()) {
//     for (auto& ag : agents_) {
//       ag.semantic_class        = "unknown";
//       ag.semantic_instance_id  = -1;
//     }
//     // ROS_INFO("11111111111111111111111111111111111111111111111111111111:No semantic objects found, resetting agent tags.");
//     return;
//   }

//   const ros::Time now = ros::Time::now();

//   for (size_t i = 0; i < agents_.size(); ++i) {
//     // 1) 取 agent 在 planner_frame_ 的位置
//     geometry_msgs::Point a_planner, a_sem;
//     a_planner.x = agents_[i].position.getX();
//     a_planner.y = agents_[i].position.getY();
//     a_planner.z = 0.0;

//     // 2) 变换到 sem_frame_ 做空间最近邻匹配
//     if (!transformPoint(a_planner, planner_frame_, sem_frame_, now, a_sem)) {
//       // 变换失败，清空标签（或保留旧值，按需）
//       agents_[i].semantic_class       = "unknown";
//       agents_[i].semantic_instance_id = -1;
//       // ROS_INFO("222222222222222222222222222222222222222222222222:Failed to transform agent position to semantic frame.");
//       continue;
//     }

//     // 3) 最近邻搜索（半径平方）
//     int best = -1;
//     double best_d2 = match_radius_ * match_radius_;
//     for (size_t j = 0; j < sem_objs_.size(); ++j) {
//       const auto& s = sem_objs_[j];
//       const double dx = a_sem.x - s.p.x;
//       const double dy = a_sem.y - s.p.y;
//       const double d2 = dx*dx + dy*dy;
//       if (d2 <= best_d2) { best_d2 = d2; best = (int)j; }
//     }

//     // 4) 写回 agent 内置字段（匹配不到则置 unknown / -1）
//     if (best >= 0) {
//       agents_[i].semantic_class       = sem_objs_[best].cls; // ✅ 直接用 GSAM2 原标签
//       agents_[i].semantic_instance_id = sem_objs_[best].id;  // ✅ 直接用实例 ID
//     } else {
//       agents_[i].semantic_class       = "unknown";
//       agents_[i].semantic_instance_id = -1;
//     }
//   }
// }

/**
 * @brief  Default destructor
 */
SFMController::~SFMController() {
  delete sensor_iface_;
  delete collision_checker_;
}

/**
 * @brief Callback to update the local planner's parameters based on dynamic
 * reconfigure
 */
void SFMController::reconfigure(
    sfm_local_controller::SFMLocalControllerConfig &cfg) {
  sfm_local_controller::SFMLocalControllerConfig config(cfg);
    ROS_INFO("Reconfiguring SFMController with new parameters...");
  configuration_mutex_.lock();

  max_lin_acc_ = config.max_lin_acc;
  max_theta_acc_ = config.max_rot_acc;
  max_lin_vel_ = config.max_lin_vel;
  robot_.desiredVelocity = max_lin_vel_;
  min_lin_vel_ = config.min_lin_vel;
  max_vel_theta_ = config.max_rot_vel;
  min_vel_theta_ = config.min_rot_vel;

  sfm_goal_weight_ = config.sfm_goal_weight;
  sfm_obstacle_weight_ = config.sfm_obstacle_weight;
  sfm_people_weight_ = config.sfm_people_weight;
  robot_.params.forceFactorDesired = sfm_goal_weight_;
  robot_.params.forceFactorObstacle = sfm_obstacle_weight_;
  robot_.params.forceFactorSocial = sfm_people_weight_;

  // min_in_place_vel_th_ = config.min_in_place_rot_vel;
  // goal_lin_tolerance_ = config.xy_goal_tolerance;
  // goal_ang_tolerance_ = config.yaw_goal_tolerance;
  // wp_tolerance_ = config.wp_tolerance;

  sensor_iface_->setMaxLinVel(max_lin_vel_);
  collision_checker_->setVelParams(max_lin_vel_, max_vel_theta_, max_lin_acc_,
                                   max_theta_acc_);

  std::cout << std::endl
            << "SFM_CONTROLLER RECONFIGURED:" << std::endl
            << "max_lin_acc: " << max_lin_acc_ << std::endl
            << "max_rot_acc: " << max_theta_acc_ << std::endl
            << "min_lin_vel: " << min_lin_vel_ << std::endl
            << "max_lin_vel: " << max_lin_vel_ << std::endl
            << "min_rot_vel: " << min_vel_theta_ << std::endl
            << "max_rot_vel: " << max_vel_theta_ << std::endl
            << "sfm_goal_weight: " << sfm_goal_weight_ << std::endl
            << "sfm_obstacle_weight: " << sfm_obstacle_weight_ << std::endl
            << "sfm_social_weight: " << sfm_people_weight_ << std::endl
            << std::endl;

  configuration_mutex_.unlock();
  ROS_INFO("SFMController reconfigured successfully.");
}

/**
 * @brief method to update the scenario situation
 * @param path global path to be followed by the local controller
 * @return True when the goal has been reached, False otherwise
 */
bool SFMController::update(std::vector<geometry_msgs::PoseStamped> path) {


  // if (!path.empty()) {
  //   const auto& global = path.back();
  //   ROS_INFO(
  //     "DEBUG → GLOBAL GOAL: x=%.3f, y=%.3f, z=%.3f",
  //     global.pose.position.x,
  //     global.pose.position.y,
  //     global.pose.position.z
  //   );
  // }
  ROS_INFO("start update...");
  std::vector<sfm::Agent> agents = sensor_iface_->getAgents();

  configuration_mutex_.lock();
  // update robot
  robot_.position = agents[0].position;
  robot_.yaw = agents[0].yaw;
  robot_.linearVelocity = agents[0].linearVelocity;
  robot_.angularVelocity = agents[0].angularVelocity;
  robot_.velocity = agents[0].velocity;
  robot_.obstacles1.clear();
  robot_.obstacles1 = agents[0].obstacles1;

  // Update the rest of agents
  agents_.clear();
  if (!agents.empty())
    agents_.assign(++agents.begin(), agents.end());
  // annotateAgents(agents_);

  // If we have to rotate, we do not look for a new goal
  if (rotate_) {
    sfm::Goal g;
    g.center.set(goal_.position.x, goal_.position.y);
    g.radius = goal_tolerance_;
    robot_.goals.clear();
    robot_.goals.push_back(g);
    if (goal_reached_) {
      goal_reached_ = false;
      rotate_ = false;
      configuration_mutex_.unlock();
      return true;
    }
    configuration_mutex_.unlock();
    return false;
  }

  // Update the robot goal
  geometry_msgs::PoseStamped min;
  float min_dist = 9999.0;
  bool goal_found = false;
  for (unsigned int i = path.size() - 1; i > 0; i--) {
    float dx = path[i].pose.position.x - robot_.position.getX();
    float dy = path[i].pose.position.y - robot_.position.getY();
    float d = sqrt(dx * dx + dy * dy);
    if (d < min_dist) {
      min_dist = d;
      min = path[i];
    }
    if (d <= local_goal_dist_) {
      if (d < goal_tolerance_) {
        configuration_mutex_.unlock();
        // goal reached! - rotate!
        printf("Update. Goal location reached. Rotating...\n");
        rotate_ = true;
        return false;
      }
      sfm::Goal g;
      goal_ = path[i].pose;
      g.center.set(path[i].pose.position.x, path[i].pose.position.y);
      g.radius = goal_tolerance_;
      robot_.goals.clear();
      robot_.goals.push_back(g);
      goal_found = true;
      goal_reached_ = false;
      // printf("Update. Goal found! x: %.2f, y: %.2f\n", g.center.getX(),
      //       g.center.getY());
      publishSFMGoal(path[i]);
      configuration_mutex_.unlock();
      return false;
    }
  }
  // printf("Goal not found closer than %.2f,\n", local_goal_dist_);
  // printf("the closest is %.2f m\n", min_dist);
  if (!path.empty()) {
    sfm::Goal g;
    goal_ = min.pose;
    g.center.set(min.pose.position.x, min.pose.position.y);
    g.radius = goal_tolerance_;
    robot_.goals.clear();
    robot_.goals.push_back(g);
    goal_reached_ = false;
    publishSFMGoal(min);
  } else {
    printf("Update. Goal not found. Received path size: %i\n",
           (int)path.size());
  }
  configuration_mutex_.unlock();
  ROS_INFO("Update completed, no goal found or reached.");
  return false;
}

/**
 * @brief method to compute the velocity command of the robot
 * @param cmd_vel velocity message to be filled
 * @return True if a command vel was found
 */
// bool SFMController::computeAction(geometry_msgs::Twist &cmd_vel) {
//   // sfm::SFM.computeForces(robot_, agents_);
//   // ROS_DEBUG(">> computed social forces (norm=%.3f)", robot_.forces.socialForce.norm());

//   ROS_INFO("Start Compute Action...");
//   configuration_mutex_.lock();
//   double dt = (ros::Time::now() - last_command_time_).toSec();
//   // printf("dt: %.4f\n", dt);
//   if (dt > 0.2)
//     dt = 0.1;

//   last_command_time_ = ros::Time::now();

//   // We must rotate to reach the goal position
//   if (rotate_) {
//     float ang_diff = robot_.yaw.toRadian() - tf::getYaw(goal_.orientation);
//     ang_diff = normalizeAngle(ang_diff, -M_PI, M_PI);
//     if (fabs(ang_diff) < yaw_tolerance_) {
//       printf("Angdiff (%.2f) < yaw_tolerance (%.2f)\n", fabs(ang_diff),
//              yaw_tolerance_);
//       cmd_vel.linear.x = 0.0;
//       cmd_vel.linear.y = 0.0;
//       cmd_vel.linear.z = 0.0;
//       cmd_vel.angular.x = 0.0;
//       cmd_vel.angular.y = 0.0;
//       cmd_vel.angular.z = 0.0;
//       goal_reached_ = true;
//       configuration_mutex_.unlock();
//       return true;
//     } else if (ang_diff > 0.0) {
//       cmd_vel.angular.z = -min_rot_in_place_;
//       // printf("Rotating with vel: %.2f\n", cmd_vel.angular.z);
//     } else {
//       cmd_vel.angular.z = min_rot_in_place_;
//       // printf("Rotating with vel: %.2f\n", cmd_vel.angular.z);
//     }
//     configuration_mutex_.unlock();
//     return true;
//   }
//   ROS_INFO("Not rotating, computing forces...");
//   // Set the factor of the obstacle force to 0.2
//   robot_.params.forceFactorObstacle = 10;
//   // robot_.params.forceFactorDesired = 0;
//   for (auto &ag : agents_) {
//     ag.params.forceFactorObstacle = 1;
//   }

//   // Compute Social Forces
//   // matchAndTagAgents();
// if (agents_.size()>=2) {
//   ROS_INFO("==== Agents before computeForces (N=%zu) ====", agents_.size());

//   if (!agents_.empty()) {
//     const auto& r = agents_[0];  // 机器人在 0 号位
//     ROS_INFO("Robot: iid=%d, class=%s, pos=(%.2f, %.2f), vel=(%.2f, %.2f)",
//             r.semantic_instance_id,
//             r.semantic_class.c_str(),
//             r.position.getX(), r.position.getY(),
//             r.velocity.getX(), r.velocity.getY());
//   }

//   // 外部 agent（从 1 开始，跳过机器人）
//   for (size_t i = 1; i < agents_.size(); ++i) {
//     const auto& ag = agents_[i];
//     ROS_INFO("Agent[%zu]: iid=%d, class=%s, pos=(%.2f, %.2f), vel=(%.2f, %.2f)",
//             i,
//             ag.semantic_instance_id,
//             ag.semantic_class.c_str(),
//             ag.position.getX(), ag.position.getY(),
//             ag.velocity.getX(), ag.velocity.getY());
//   }
//   ROS_INFO("============================================");
// }

//   ROS_INFO("Compute Forces for robot and agents...");
//   sfm::SFM.computeForces(robot_, agents_);
//   ROS_INFO("End Compute Forces for robot and agents...");
//   // ROS_INFO("SocialForce norm5 = %.3f", robot_.forces.socialForce.norm());




//   // Compute velocity of the robot
//   robot_.velocity += robot_.forces.globalForce * dt;
//   if (robot_.velocity.norm() > robot_.desiredVelocity) {
//     robot_.velocity.normalize();
//     robot_.velocity *= robot_.desiredVelocity;
//   }

//   // The resultant total velocity is expressed in the odom frame. Transform
//   // to robot_frame
//   geometry_msgs::Vector3 velocity;
//   velocity.x = robot_.velocity.getX();
//   velocity.y = robot_.velocity.getY();
//   geometry_msgs::Vector3 localV =
//       sensor_iface_->transformVector(velocity, planner_frame_, robot_frame_);

//   utils::Vector2d vel;
//   vel.set(localV.x, localV.y);
//   cmd_vel.linear.x = vel.norm();

//   // Decrease speed to approach the goal
//   float dx = goal_.position.x - robot_.position.getX();
//   float dy = goal_.position.y - robot_.position.getY();
//   float d = sqrt(dx * dx + dy * dy);
//   if (d < 1.0)
//     cmd_vel.linear.x =
//         (vel.norm() * d) < min_lin_vel_ ? min_lin_vel_ : (vel.norm() * d);

//   cmd_vel.linear.y = 0.0;
//   cmd_vel.linear.z = 0.0;
//   // Wz = std::atan2(localV.y,localV.x)/2.0;
//   double angle = std::atan2(localV.y, localV.x);
//   cmd_vel.angular.z = getVel(max_vel_theta_, a_, angle);
//   cmd_vel.angular.x = 0.0;
//   cmd_vel.angular.y = 0.0;

//   // Prevent the robot for turning around:
//   // If the angle difference between the desired force
//   // and the global force is almost opossite,
//   // stop the robot instead of turning around
//   double angle_deg =
//       robot_.forces.desiredForce.angleTo(robot_.forces.globalForce).toDegree();
//   if ((180.0 - fabs(angle_deg)) < 25.0) {
//     // printf("\nStopping robot. angle_deg: %.3f!!!!!\n", angle_deg);
//     cmd_vel.linear.x = 0.0;
//     cmd_vel.angular.z = 0.0;
//   }

//   publishForces();
//   // ROS_WARN("cx: %.2f, tx%.2f", robot_.linearVelocity, cmd_vel.linear.x);
//   if (!collision_checker_->checkCommand(
//           robot_.linearVelocity, 0.0, robot_.angularVelocity, cmd_vel.linear.x,
//           0.0, cmd_vel.angular.z, 0.11)) {
//     ROS_WARN("Possible collision detected! Sending opposite command!");
//     cmd_vel.linear.x = -cmd_vel.linear.x;
//     cmd_vel.angular.z = 0.0;
//   }

//   // ROS_INFO("LV: %f; AV: %f", cmd_vel.linear.x, cmd_vel.angular.z);
//   configuration_mutex_.unlock();
//   return true;
// }
bool SFMController::computeAction(geometry_msgs::Twist &cmd_vel) {
  ROS_INFO("Start Compute Action...");

  // ===== ① 在短锁里拿快照，尽快解锁 =====
  sfm::Agent robot_s;
  std::vector<sfm::Agent> agents_s;
  geometry_msgs::Pose goal_s;
  double min_lin_vel_s, max_vel_theta_s, yaw_tol_s, min_rot_in_place_s, a_s;
  std::string planner_frame_s, robot_frame_s;
  double dt_s;
  bool rotate_s;
  double linVel_s, angVel_s;   // 用于碰撞检查

  {
    configuration_mutex_.lock();

    dt_s = (ros::Time::now() - last_command_time_).toSec();
    if (dt_s > 0.2) dt_s = 0.1;
    last_command_time_ = ros::Time::now();

    robot_s           = robot_;
    agents_s          = agents_;
    goal_s            = goal_;
    min_lin_vel_s     = min_lin_vel_;
    max_vel_theta_s   = max_vel_theta_;
    yaw_tol_s         = yaw_tolerance_;
    min_rot_in_place_s= min_rot_in_place_;
    a_s               = a_;
    planner_frame_s   = planner_frame_;
    robot_frame_s     = robot_frame_;
    rotate_s          = rotate_;
    linVel_s          = robot_.linearVelocity;
    angVel_s          = robot_.angularVelocity;

    configuration_mutex_.unlock();
  }

  // ===== ② 重活都在锁外做 =====
  if (rotate_s) {
    float ang_diff = robot_s.yaw.toRadian() - tf::getYaw(goal_s.orientation);
    ang_diff = normalizeAngle(ang_diff, -M_PI, M_PI);
    if (std::fabs(ang_diff) < yaw_tol_s) {
      cmd_vel = geometry_msgs::Twist();   // 全零
      // 短锁设置标志
      configuration_mutex_.lock();
      goal_reached_ = true;
      configuration_mutex_.unlock();
      return true;
    } else {
      cmd_vel.angular.z = (ang_diff > 0.0) ? -min_rot_in_place_s : min_rot_in_place_s;
      return true;
    }
  }

  ROS_INFO("Not rotating, computing forces...");
  robot_s.params.forceFactorObstacle = 10;
  for (auto &ag : agents_s) ag.params.forceFactorObstacle = 1;

  ROS_INFO("Compute Forces for robot and agents...");
  sfm::SFM.computeForces(robot_s, agents_s);
  ROS_INFO("End Compute Forces for robot and agents...");

  // 速度积分
  robot_s.velocity += robot_s.forces.globalForce * dt_s;
  if (robot_s.velocity.norm() > robot_s.desiredVelocity) {
    robot_s.velocity.normalize();
    robot_s.velocity *= robot_s.desiredVelocity;
  }

  // 坐标变换到机体系
  geometry_msgs::Vector3 v;
  v.x = robot_s.velocity.getX();
  v.y = robot_s.velocity.getY();
  geometry_msgs::Vector3 localV =
      sensor_iface_->transformVector(v, planner_frame_s, robot_frame_s);

  utils::Vector2d vel(localV.x, localV.y);
  cmd_vel.linear.x = vel.norm();

  // 接近目标降速
  const double dx = goal_s.position.x - robot_s.position.getX();
  const double dy = goal_s.position.y - robot_s.position.getY();
  const double d  = std::sqrt(dx*dx + dy*dy);
  if (d < 1.0) cmd_vel.linear.x = std::max(min_lin_vel_s, vel.norm() * d);

  // 角速度
  cmd_vel.angular.z = getVel(max_vel_theta_s, a_s, std::atan2(localV.y, localV.x));

  // 避免反向掉头
  const double angle_deg =
      robot_s.forces.desiredForce.angleTo(robot_s.forces.globalForce).toDegree();
  if ((180.0 - std::fabs(angle_deg)) < 25.0) {
    cmd_vel.linear.x = 0.0;
    cmd_vel.angular.z = 0.0;
  }

  // 碰撞检查也放锁外
  if (!collision_checker_->checkCommand(
          linVel_s, 0.0, angVel_s,
          cmd_vel.linear.x, 0.0, cmd_vel.angular.z, 0.11)) {
    ROS_WARN("Possible collision detected! Sending opposite command!");
    cmd_vel.linear.x = -cmd_vel.linear.x;
    cmd_vel.angular.z = 0.0;
  }

  // ===== ③ 短锁回写并发布 =====
  {
    configuration_mutex_.lock();
    robot_ = robot_s;          // 用计算结果回写
    publishForces();           // 发布 Marker（如果担心发布阻塞，也可以把 publishForces 改成接收快照参数）
    configuration_mutex_.unlock();
  }
  ROS_INFO("Compute Action completed successfully.");
  return true;
}


/**
 * @brief check if the current scenario leads to a possible collision
 * @return True if a possible collision was detected, False otherwise
 */
bool SFMController::fastCollisioncheck() {
  // return sensor_iface_->collisionCheck();
  // ROS_INFO("Fast collision check performed.");
  return true;
}

/**
 * @brief Publish in RViz the local goal followed by the controller
 * @param g goal position
 * @return none
 */
void SFMController::publishSFMGoal(const geometry_msgs::PoseStamped &g) {
  // ROS_INFO("Publishing SFM goal in RViz...");
  visualization_msgs::Marker marker;
  marker.header.frame_id = g.header.frame_id;
  marker.header.stamp = ros::Time::now();
  marker.ns = "sfm_goal";
  marker.id = 1;
  marker.action = visualization_msgs::Marker::ADD;
  marker.type = visualization_msgs::Marker::ARROW;
  marker.color = getColor(0.0, 0.0, 1.0, 1.0);
  marker.lifetime = ros::Duration(0.15);
  marker.scale.x = 0.4;
  marker.scale.y = 0.1;
  marker.scale.z = 0.1;
  marker.pose = g.pose;
  sfm_goal_pub_.publish(marker);
  // ROS_INFO("SFM goal published in RViz.");
}
/**
 * @brief Publish an arrow marker in Rviz representing a force
 * @param index id of the marker
 * @param color RGB color of the marker
 * @param force force to be represented
 * @param markers markerArray in which the arrow will be added
 * @return none
 */
void SFMController::publishForceMarker(
    unsigned index, const std_msgs::ColorRGBA &color,
    const utils::Vector2d &force, visualization_msgs::MarkerArray &markers, const std::string &ns) {
  //  ROS_INFO("Publishing force marker %d in RViz...", index);
  visualization_msgs::Marker marker;
  // ROS_WARN("NS: %s", ns.c_str());
  marker.header.frame_id = planner_frame_;
  marker.header.stamp = ros::Time::now();
  marker.ns = ns;
  marker.id = index;
  // marker.action = force.norm() > 1e-4 ? 0 : 2;
  marker.color = color;
  marker.lifetime = ros::Duration(1.0);
  marker.scale.x = std::max(1e-4, force.norm());
  marker.scale.y = 0.1;
  marker.scale.z = 0.1;
  marker.pose.position.x = robot_.position.getX();
  marker.pose.position.y = robot_.position.getY();
  marker.pose.position.z = 0;
  marker.pose.orientation =
      tf::createQuaternionMsgFromRollPitchYaw(0, 0, force.angle().toRadian());
  markers.markers.push_back(marker);
  // ROS_INFO("Force marker %d published in RViz.", index);
}

/**
 * @brief fill a ColorRGBA message
 * @param r value of the red componet [0-1]
 * @param g value of the green component [0-1]
 * @param b value of the blue component [0-1]
 * @param a transparency of the color [0-1]
 * @return a ROS ColorRGBA message
 */
std_msgs::ColorRGBA SFMController::getColor(double r, double g, double b,
                                            double a) {
  std_msgs::ColorRGBA color;
  color.r = r;
  color.g = g;
  color.b = b;
  color.a = a;
  return color;
}

/**
 * @brief Publish the set of SFM forces in RViz匹
 * @return none
 */
void SFMController::publishForces() {
  // ROS_INFO("Publishing forces in RViz...");
  visualization_msgs::MarkerArray markers;
  publishForceMarker(0, getColor(1, 0, 0, 1), robot_.forces.obstacleForce,
                     markers, "obstacle_force");
  publishForceMarker(1, getColor(0, 0, 1, 1), robot_.forces.socialForce,
                     markers, "social_force");
  publishForceMarker(2, getColor(0, 1, 1, 1), robot_.forces.groupForce,
                     markers, "group_force");
  publishForceMarker(3, getColor(0, 1, 0, 1), robot_.forces.desiredForce,
                     markers, "desired_force");
  publishForceMarker(4, getColor(1, 1, 1, 1), robot_.forces.globalForce,
                     markers, "global_force");
  publishForceMarker(5, getColor(1, 1, 0, 1), robot_.velocity, markers, "velocity");
  robot_markers_pub_.publish(markers);
  // ROS_INFO_STREAM("Goal: " << robot_.forces.desiredForce.norm()
  //                         << ", Obstacle: "
  //                         << robot_.forces.obstacleForce.norm() << ", People:
  //                         "
  //                         << robot_.forces.socialForce.norm() << std::endl);
}

} // namespace sfm_controller
} // namespace Navigation
} // namespace Upo