#ifndef COSTMAP_2D_SOCIAL_LAYER_H_
#define COSTMAP_2D_SOCIAL_LAYER_H_

#include <ros/ros.h>
#include <tvss_nav/layers/layer.h>
#include <costmap_2d/costmap_layer.h>
#include <costmap_2d/layered_costmap.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <tvss_nav/SemanticInstance.h>
#include <tvss_nav/SemanticInstanceArray.h>

#include <nav_msgs/OccupancyGrid.h>

#include <dynamic_reconfigure/server.h>
#include <tvss_nav/SocialPluginConfig.h>

namespace tvss_nav
{

typedef std::vector<tvss_nav::SemanticInstance> SocialObservation;

class SocialBuffer{
public:
  SocialBuffer(ros::NodeHandle _nh, std::string _topic){
    nh = _nh;
    stamp = ros::Time(0);
    topic = _topic;
  };

  void subscribe(){
    sub = nh.subscribe(
      topic,
      1,
      &SocialBuffer::callback,
      this
    );
  };

  void unsubscribe(){
    if(sub)
      sub.shutdown();
  };

  ~SocialBuffer(){
    unsubscribe();
  };

  std::string getTopic() { return topic; }
  ros::Time getStamp(){ return stamp; }
  SocialObservation getData(){ return data; }

private:
  void callback(const tvss_nav::SemanticInstanceArrayConstPtr& msg){
    stamp = msg->header.stamp;
    data = msg->instances;
  }

  ros::NodeHandle nh;
  std::string topic;

  ros::Time stamp;
  SocialObservation data;

  ros::Subscriber sub;
};

class SocialLayer : public costmap_2d::CostmapLayer
{
public:
  SocialLayer()
  {
    costmap_ = NULL;
  }

  virtual ~SocialLayer();
  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y ,
                            double* max_x, double* max_y) override;
  virtual void updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j) override;

  virtual void activate();
  virtual void deactivate();
  virtual void reset();

  boost::any dump(LayerType& type) override;

protected:
  virtual void setupDynamicReconfigure(ros::NodeHandle& nh);

  bool getObservations(std::vector<boost::shared_ptr<SocialBuffer>>& buffers, std::vector<SocialObservation>& observations) const;

  std::string global_frame_;

  std::vector<boost::shared_ptr<SocialBuffer>> observation_buffers_;
  std::vector<boost::shared_ptr<SocialBuffer>> marking_buffers_;
  std::vector<boost::shared_ptr<SocialBuffer>> clearing_buffers_;

  bool rolling_window_;
  dynamic_reconfigure::Server<tvss_nav::SocialPluginConfig> *dsrv_;
  tvss_nav::SocialPluginConfig config_;
  tvss_nav::SemanticInstanceArray last_received_instances_;
  int combination_method_;

private:
  void reconfigureCB(tvss_nav::SocialPluginConfig &config, uint32_t level);
};

}  // namespace costmap_2d

#endif  // COSTMAP_2D_SOCIAL_LAYER_H_