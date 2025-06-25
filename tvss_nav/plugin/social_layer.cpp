#include <tvss_nav/layers/social_layer.h>
#include <pluginlib/class_list_macros.h>
#include <costmap_2d/costmap_math.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

using costmap_2d::NO_INFORMATION;
using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::FREE_SPACE;

namespace tvss_nav
{

SocialLayer::~SocialLayer() {
  if (dsrv_)
    delete dsrv_;
}

void SocialLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_), g_nh;
  rolling_window_ = layered_costmap_->isRolling();

  default_value_ = 64;

  SocialLayer::matchSize();
  current_ = true;

  global_frame_ = layered_costmap_->getGlobalFrameID();

  std::string topics_string;
  nh.param("observation_sources", topics_string, std::string(""));
  ROS_INFO("    Subscribed to Topics: %s", topics_string.c_str());

  std::stringstream ss(topics_string);
  std::string source;
  while (ss >> source)
  {
    ros::NodeHandle source_node(nh, source);

    std::string topic;
    bool clearing, marking;

    source_node.param("topic", topic, source);
    source_node.param("clearing", clearing, false);
    source_node.param("marking", marking, false);

    ROS_DEBUG("Creating a social buffer for source %s, topic %s", source.c_str(), topic.c_str());

    observation_buffers_.push_back(
      boost::shared_ptr<SocialBuffer>(new SocialBuffer(source_node, topic))
    );

    if (marking)
      marking_buffers_.push_back(observation_buffers_.back());

    if (clearing)
      clearing_buffers_.push_back(observation_buffers_.back());
  }

  dsrv_ = NULL;
  setupDynamicReconfigure(nh);
  activate();
}

void SocialLayer::updateBounds(double robot_x, double robot_y, double robot_yaw,
                                double* min_x, double* min_y, double* max_x, double* max_y)
{
  // std::vector<SocialObservation> observations;
  // if (!getObservations(marking_buffers_, observations)) return;

  // for (const auto& obs : observations)
  // {
  //   for (const auto& instance : obs)
  //   {
  //     pcl::PointCloud<pcl::PointXYZ> cloud, cloud_transformed;
  //     pcl::fromROSMsg(instance.cloud, cloud);
      
  //     geometry_msgs::TransformStamped tf_transform;
  //     try {
  //       tf_transform = tf_->lookupTransform(
  //           global_frame_, 
  //           instance.cloud.header.frame_id, 
  //           instance.cloud.header.stamp, 
  //           ros::Duration(0.5));
        
  //       Eigen::Affine3d tf_eigen;
  //       tf_eigen = tf2::transformToEigen(tf_transform);
  //       Eigen::Matrix4f tf_mat = tf_eigen.matrix().cast<float>();

  //       pcl::transformPointCloud(cloud, cloud_transformed, tf_mat);
  //     } catch (const tf2::TransformException& ex) {
  //       ROS_WARN("Transform failed in updateBounds: %s", ex.what());
  //       continue;
  //     }

  //     double r = instance.inflation_radius > 0 ? instance.inflation_radius : config_.default_inflation_radius;

  //     for (const auto& pt : cloud_transformed)
  //     {
  //       if (pt.z < config_.min_height || pt.z > config_.max_height) continue;
  //       *min_x = std::min(*min_x, pt.x - r);
  //       *min_y = std::min(*min_y, pt.y - r);
  //       *max_x = std::max(*max_x, pt.x + r);
  //       *max_y = std::max(*max_y, pt.y + r);
  //     }
  //   }
  // }

  *min_x = origin_x_;
  *min_y = origin_y_;
  *max_x = origin_x_ + size_x_ * resolution_;
  *max_y = origin_y_ + size_y_ * resolution_;
}

void SocialLayer::updateCosts(costmap_2d::Costmap2D& master_grid,
                               int min_i, int min_j, int max_i, int max_j)
{
  // // clearing logic
  // for (const auto& buffer : clearing_buffers_)
  // {
  //   SocialObservation data = buffer->getData();
  //   for (const auto& instance : data)
  //   {
  //     pcl::PointCloud<pcl::PointXYZ> cloud, cloud_transformed;
  //     pcl::fromROSMsg(instance.cloud, cloud);

  //     geometry_msgs::TransformStamped tf_transform;
  //     try {
  //       tf_transform = tf_->lookupTransform(
  //           global_frame_, 
  //           instance.cloud.header.frame_id, 
  //           instance.cloud.header.stamp, 
  //           ros::Duration(0.5));
        
  //       Eigen::Affine3d tf_eigen;
  //       tf_eigen = tf2::transformToEigen(tf_transform);
  //       Eigen::Matrix4f tf_mat = tf_eigen.matrix().cast<float>();

  //       pcl::transformPointCloud(cloud, cloud_transformed, tf_mat);
  //     } catch (const tf2::TransformException& ex) {
  //       ROS_WARN("Transform failed in updateCosts: %s", ex.what());
  //       continue;
  //     }

  //     for (const auto& pt : cloud_transformed)
  //     {
  //       if (pt.z < config_.min_height || pt.z > config_.max_height) continue;

  //       unsigned int mx, my;
  //       if (!master_grid.worldToMap(pt.x, pt.y, mx, my)) continue;

  //       master_grid.setCost(mx, my, default_value_);
  //     }
  //   }
  // }

  for (int x = min_i; x < max_i; ++x)
  {
    for (int y = min_j; y < max_j; ++y)
    {
      int bias_value = int((costmap_2d::LETHAL_OBSTACLE - default_value_) * master_grid.getCost(x, y) / costmap_2d::LETHAL_OBSTACLE) + default_value_;
      master_grid.setCost(x, y, bias_value);
    }
  }

  // marking logic
  std::vector<SocialObservation> observations;
  if (!getObservations(marking_buffers_, observations)) return;

  for (const auto& obs : observations)
  {
    for (const auto& instance : obs)
    {
      pcl::PointCloud<pcl::PointXYZ> cloud, cloud_transformed;
      pcl::fromROSMsg(instance.cloud, cloud);

      geometry_msgs::TransformStamped tf_transform;
      try {
        tf_transform = tf_->lookupTransform(
            global_frame_, 
            instance.cloud.header.frame_id, 
            instance.cloud.header.stamp, 
            ros::Duration(0.5));
        
        Eigen::Affine3d tf_eigen;
        tf_eigen = tf2::transformToEigen(tf_transform);
        Eigen::Matrix4f tf_mat = tf_eigen.matrix().cast<float>();

        pcl::transformPointCloud(cloud, cloud_transformed, tf_mat);
      } catch (const tf2::TransformException& ex) {
        ROS_WARN("Transform failed in updateCosts: %s", ex.what());
        continue;
      }

      int base_cost = instance.cost_value > 0 ? instance.cost_value : config_.default_cost_value;
      double r = instance.inflation_radius > 0 ? instance.inflation_radius : config_.default_inflation_radius;
      double decay = instance.decay_rate > 0 ? instance.decay_rate : config_.default_decay_rate;

      int radius_cells = std::round(r / resolution_);

      for (const auto& pt : cloud_transformed)
      {
        if (pt.z < config_.min_height || pt.z > config_.max_height) continue;

        unsigned int mx, my;
        if (!master_grid.worldToMap(pt.x, pt.y, mx, my)) continue;

        switch (config_.combination_method)
        {
          case 0:  // Overwrite
            master_grid.setCost(mx, my, base_cost);
            break;
          case 1:  // Maximum
            master_grid.setCost(mx, my, std::max(master_grid.getCost(mx, my), (unsigned char)base_cost));
            break;
          default:
            break;
        }

        for (int dx = -radius_cells; dx <= radius_cells; ++dx)
        {
          for (int dy = -radius_cells; dy <= radius_cells; ++dy)
          {
            int nx = mx + dx, ny = my + dy;
            if (nx < 0 || ny < 0 || nx >= (int)master_grid.getSizeInCellsX() || ny >= (int)master_grid.getSizeInCellsY()) continue;

            double dist = hypot(dx, dy) * resolution_;
            int inflated = std::round(base_cost * exp(-decay * dist));

            switch (config_.combination_method)
            {
              case 0:  // Overwrite
                master_grid.setCost(nx, ny, inflated);
                break;
              case 1:  // Maximum
                master_grid.setCost(nx, ny, std::max(master_grid.getCost(nx, ny), (unsigned char)inflated));
                break;
              default:
                break;
            }
          }
        }
      }
    }
  }
}

void SocialLayer::activate()
{
  for (const auto& buffer : observation_buffers_)
    buffer->subscribe();
}

void SocialLayer::deactivate()
{
  for (const auto& buffer : observation_buffers_)
    buffer->unsubscribe();
}

void SocialLayer::reset()
{
  deactivate();
  resetMaps();
  current_ = true;
  activate();
}

bool SocialLayer::getObservations(std::vector<boost::shared_ptr<SocialBuffer>>& buffers, std::vector<SocialObservation>& observations) const
{
  bool current = true;
  for (const auto& buffer: buffers)
  {
    observations.push_back(buffer->getData());
    current &= buffer->getStamp().isValid();
  }
  return current;
}

void SocialLayer::setupDynamicReconfigure(ros::NodeHandle& nh)
{
  dsrv_ = new dynamic_reconfigure::Server<tvss_nav::SocialPluginConfig>(nh);
  dynamic_reconfigure::Server<tvss_nav::SocialPluginConfig>::CallbackType cb =
      boost::bind(&SocialLayer::reconfigureCB, this, _1, _2);
  dsrv_->setCallback(cb);
}

void SocialLayer::reconfigureCB(tvss_nav::SocialPluginConfig &config, uint32_t level)
{
  config_ = config;
  enabled_ = config.enabled;
}

boost::any SocialLayer::dump(LayerType& type)
{
  tvss_nav::SemanticInstanceArray msg = last_received_instances_;
  msg.header.stamp = ros::Time::now();
  msg.header.frame_id = global_frame_;
  return boost::any(msg);
}

PLUGINLIB_EXPORT_CLASS(tvss_nav::SocialLayer, costmap_2d::Layer)

}  // namespace costmap_2d
