// seg_node.cpp (parametrized topics + auto format detection)
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Header.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <tvss_nav/SemanticInstance.h>
#include <tvss_nav/SemanticInstanceArray.h>
#include <tvss_nav/utils/geometry_utils.h>

#include <unordered_map>
#include <vector>
#include <string>
#include <set>

class PointCloudSegNode
{
public:
    PointCloudSegNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), tf_buffer_(), tf_listener_(tf_buffer_)
    {
        pnh.param<std::string>("pointcloud_topic", pointcloud_topic_, "/camera/depth/color/points");
        pnh.param<std::string>("mask_topic", mask_topic_, "/segmented_image/mask");
        pnh.param<std::string>("rgb_info_topic", rgb_info_topic_, "/camera/color/camera_info");

        pnh.param<std::string>("semantic_instances_topic", instance_array_topic_, "/instance_array");
        pnh.param<std::string>("visual_cloud_topic", visual_cloud_topic_, "/masked_cloud");

        pnh.param<double>("voxel_leaf_size", voxel_leaf_size_, 0.1);

        cloud_sub_.subscribe(nh_, pointcloud_topic_, 1);
        mask_sub_.subscribe(nh_, mask_topic_, 1);
        rgb_info_sub_ = nh_.subscribe(rgb_info_topic_, 1, &PointCloudSegNode::handleRgbInfo, this);

        sync_.reset(new Sync(ExactSyncPolicy(10), cloud_sub_, mask_sub_));
        sync_->registerCallback(boost::bind(&PointCloudSegNode::processData, this, _1, _2));

        semantic_pub_ = nh_.advertise<tvss_nav::SemanticInstanceArray>(instance_array_topic_, 1);
        visual_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(visual_cloud_topic_, 1);
    }

private:
    ros::NodeHandle nh_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<sensor_msgs::CompressedImage> mask_sub_;
    ros::Subscriber rgb_info_sub_;
    ros::Publisher semantic_pub_;
    ros::Publisher visual_cloud_pub_;

    std::string pointcloud_topic_, mask_topic_, rgb_info_topic_;
    std::string instance_array_topic_, visual_cloud_topic_;
    double voxel_leaf_size_;

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::PointCloud2, sensor_msgs::CompressedImage> ExactSyncPolicy;
    typedef message_filters::Synchronizer<ExactSyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    cv::Mat rgb_intrinsics_;
    cv::Mat rgb_distortion_;
    cv::Size rgb_image_size_ = cv::Size(640, 480);

    void handleRgbInfo(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        rgb_intrinsics_ = cv::Mat(3, 3, CV_64F, (void*)msg->K.data()).clone();
        rgb_distortion_ = cv::Mat(msg->D).clone();
        rgb_image_size_ = cv::Size(msg->width, msg->height);
    }

    void processData(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                     const sensor_msgs::CompressedImageConstPtr& mask_msg)
    {
        if (rgb_intrinsics_.empty()) {
            ROS_WARN("Waiting for camera intrinsics...");
            return;
        }

        cv::Mat mask = cv::imdecode(cv::Mat(mask_msg->data), cv::IMREAD_GRAYSCALE);
        if (mask.empty()) {
            ROS_ERROR("cv::imdecode failed: received empty mask.");
            return;
        }

        if (mask.cols != rgb_image_size_.width || mask.rows != rgb_image_size_.height) {
            ROS_ERROR("Mask size (%d, %d) does not match RGB image size (%d, %d)",
                      mask.cols, mask.rows, rgb_image_size_.width, rgb_image_size_.height);
            return;
        }

        geometry_msgs::TransformStamped tf_transform;
        try {
            tf_transform = tf_buffer_.lookupTransform(
                cloud_msg->header.frame_id,
                mask_msg->header.frame_id,
                cloud_msg->header.stamp,
                ros::Duration(0.5)  // timeout
            );
        } catch (tf2::TransformException& ex) {
            ROS_WARN("TF2 transform failed: %s", ex.what());
            return;
        }

        std::set<std::string> fields;
        for (const auto& field : cloud_msg->fields) fields.insert(field.name);
        bool has_rgb = fields.count("r") && fields.count("g") && fields.count("b");

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_r(*cloud_msg, "r");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_g(*cloud_msg, "g");
        sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_b(*cloud_msg, "b");

        std::unordered_map<int, pcl::PointCloud<pcl::PointXYZRGB>> point_clusters;
        pcl::PointCloud<pcl::PointXYZRGB> all_points;

        for (int i = 0; i < cloud_msg->width * cloud_msg->height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            geometry_msgs::PointStamped pt_in, pt_out;
            pt_in.header.frame_id = cloud_msg->header.frame_id;
            pt_in.point.x = *iter_x;
            pt_in.point.y = *iter_y;
            pt_in.point.z = *iter_z;

            tf2::doTransform(pt_in, pt_out, tf_transform);

            int u, v;
            std::tie(u, v) = tvss_nav::projectToImage(pt_out.point.x, pt_out.point.y, pt_out.point.z, rgb_intrinsics_, rgb_distortion_);
            if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
                int label = mask.at<uchar>(v, u);
                if (label == 0 || label == 255) continue;

                pcl::PointXYZRGB pt;
                pt.x = *iter_x;
                pt.y = *iter_y;
                pt.z = *iter_z;

                if (has_rgb) {
                    pt.r = *iter_r;
                    pt.g = *iter_g;
                    pt.b = *iter_b;
                    ++iter_r; ++iter_g; ++iter_b;
                } else {
                    pt.r = pt.g = pt.b = 128;
                }

                point_clusters[label].push_back(pt);
                all_points.push_back(pt);
            } else if (has_rgb) {
                ++iter_r; ++iter_g; ++iter_b;
            }
        }

        for (auto& pair : point_clusters) {
            pcl::VoxelGrid<pcl::PointXYZRGB> voxel;
            voxel.setInputCloud(pair.second.makeShared());
            voxel.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
            pcl::PointCloud<pcl::PointXYZRGB> filtered;
            voxel.filter(filtered);
            pair.second.swap(filtered);
        }

        sensor_msgs::PointCloud2 merged_msg;
        pcl::PointCloud<pcl::PointXYZRGB> merged_cloud;

        tvss_nav::SemanticInstanceArray semantic_instances;
        for (const auto& cluster : point_clusters) {
            AddInstance(cluster.first, cluster.second, cloud_msg->header, semantic_instances);
            merged_cloud += cluster.second;
        }

        semantic_instances.header = cloud_msg->header;
        semantic_pub_.publish(semantic_instances);

        pcl::toROSMsg(merged_cloud, merged_msg);
        merged_msg.header = cloud_msg->header;
        visual_cloud_pub_.publish(merged_msg);
    }

    void AddInstance(int label,
                     const pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                     const std_msgs::Header& header,
                     tvss_nav::SemanticInstanceArray& array,
                     int cost_value = 254,
                     float inflation_radius = 1.0,
                     float decay_rate = 2.7685,
                     const std::string& class_name = "unknown")
    {
        tvss_nav::SemanticInstance instance;
        instance.instance_id = label;
        instance.cost_value = cost_value;
        instance.inflation_radius = inflation_radius;
        instance.decay_rate = decay_rate;
        instance.class_name = class_name;

        pcl::toROSMsg(cloud, instance.cloud);
        instance.cloud.header = header;

        array.instances.push_back(instance);
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "seg_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    PointCloudSegNode node(nh, pnh);
    ros::spin();
    return 0;
}
