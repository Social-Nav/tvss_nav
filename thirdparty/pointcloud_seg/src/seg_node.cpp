// seg_node.cpp (parametrized topics + auto format detection)
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/Header.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>

#include <tf/transform_listener.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <unordered_map>
#include <vector>

class PointCloudSegNode
{
public:
    PointCloudSegNode(ros::NodeHandle& nh, ros::NodeHandle& pnh)
        : nh_(nh), tf_listener_()
    {
        pnh.param<std::string>("pointcloud_topic", pointcloud_topic_, "/camera/depth/color/points");
        pnh.param<std::string>("mask_topic", mask_topic_, "/segmented_image/mask");
        pnh.param<std::string>("rgb_info_topic", rgb_info_topic_, "/camera/color/camera_info");

        ROS_INFO("pointcloud_seg node launched with:");
        ROS_INFO("  pointcloud_topic: %s", pointcloud_topic_.c_str());
        ROS_INFO("  mask_topic: %s", mask_topic_.c_str());

        cloud_sub_.subscribe(nh_, pointcloud_topic_, 1);
        mask_sub_.subscribe(nh_, mask_topic_, 1);

        rgb_info_sub_ = nh_.subscribe(rgb_info_topic_, 1, &PointCloudSegNode::handleRgbInfo, this);

        sync_.reset(new Sync(ApproxSyncPolicy(100), cloud_sub_, mask_sub_));
        sync_->registerCallback(boost::bind(&PointCloudSegNode::processData, this, _1, _2));

        depth_image_pub_ = nh_.advertise<sensor_msgs::Image>("/rgb_depth_image", 1);
    }

private:
    ros::NodeHandle nh_;
    tf::TransformListener tf_listener_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub_;
    message_filters::Subscriber<sensor_msgs::Image> mask_sub_;
    ros::Subscriber rgb_info_sub_;
    ros::Publisher depth_image_pub_;

    std::string pointcloud_topic_, mask_topic_, rgb_info_topic_;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image> ApproxSyncPolicy;
    typedef message_filters::Synchronizer<ApproxSyncPolicy> Sync;
    boost::shared_ptr<Sync> sync_;

    std::unordered_map<int, ros::Publisher> instance_pubs_;
    cv::Mat rgb_intrinsics_;
    cv::Size rgb_image_size_ = cv::Size(640, 480);

    void handleRgbInfo(const sensor_msgs::CameraInfoConstPtr& msg)
    {
        rgb_intrinsics_ = cv::Mat(3, 3, CV_64F, (void*)msg->K.data()).clone();
        rgb_image_size_ = cv::Size(msg->width, msg->height);
    }

    void processData(const sensor_msgs::PointCloud2ConstPtr& cloud_msg,
                     const sensor_msgs::ImageConstPtr& mask_msg)
    {
        if (rgb_intrinsics_.empty()) {
            ROS_WARN("Waiting for camera intrinsics...");
            return;
        }

        cv::Mat mask;
        bool decoded = false;
        try {
            // Attempt raw conversion first
            cv_bridge::CvImageConstPtr cv_mask_raw = cv_bridge::toCvShare(mask_msg, sensor_msgs::image_encodings::MONO8);
            mask = cv_mask_raw->image;
            decoded = true;
            ROS_DEBUG_ONCE("Mask received as raw sensor_msgs::Image");
        } catch (cv_bridge::Exception& e) {
            ROS_WARN_ONCE("Raw decoding failed, trying PNG decode (maybe this is a compressed PNG Image masquerading as raw)...");
            try {
                std::vector<uint8_t> data = mask_msg->data;
                mask = cv::imdecode(cv::Mat(data), cv::IMREAD_GRAYSCALE);
                decoded = !mask.empty();
                if (!decoded) ROS_ERROR("cv::imdecode failed: empty result.");
                else ROS_DEBUG_ONCE("Successfully decoded PNG-compressed mask image");
            } catch (const std::exception& ex) {
                ROS_ERROR("Mask decoding failed: %s", ex.what());
                return;
            }
        }

        if (!decoded) {
            ROS_ERROR("Unable to decode mask image");
            return;
        }

        tf::StampedTransform tf_transform;
        try {
            tf_listener_.waitForTransform(cloud_msg->header.frame_id, mask_msg->header.frame_id, ros::Time(0), ros::Duration(1.0));
            tf_listener_.lookupTransform(cloud_msg->header.frame_id, mask_msg->header.frame_id, ros::Time(0), tf_transform);
        } catch (tf::TransformException& ex) {
            ROS_WARN("TF transform failed: %s", ex.what());
            return;
        }

        Eigen::Matrix4f transform_mat = convertTfToMatrix(tf_transform);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud_msg, "z");

        std::unordered_map<int, std::vector<std::array<float, 3>>> point_clusters;
        std::vector<std::array<float, 3>> all_points;

        cv::Mat depth_map(rgb_image_size_, CV_32FC1, std::numeric_limits<float>::infinity());

        for (int i = 0; i < cloud_msg->width * cloud_msg->height; ++i, ++iter_x, ++iter_y, ++iter_z) {
            Eigen::Vector4f point(*iter_x, *iter_y, *iter_z, 1.0f);
            Eigen::Vector4f point_rgb = transform_mat * point;

            int u, v;
            std::tie(u, v) = projectToImage(point_rgb[0], point_rgb[1], point_rgb[2], rgb_intrinsics_);
            if (u >= 0 && u < mask.cols && v >= 0 && v < mask.rows) {
                int label = mask.at<uchar>(v, u);
                if (label > 0) {
                    point_clusters[label].push_back({*iter_x, *iter_y, *iter_z});
                    all_points.push_back({*iter_x, *iter_y, *iter_z});
                }
                if (point_rgb[2] < depth_map.at<float>(v, u)) {
                    depth_map.at<float>(v, u) = point_rgb[2];
                }
            }
        }

        for (const auto& cluster : point_clusters) {
            publishInstanceCloud(cluster.first, cluster.second, cloud_msg->header);
        }
        if (!all_points.empty()) {
            publishInstanceCloud(0, all_points, cloud_msg->header);
        }

        // cv::Mat binary_mask;
        // cv::threshold(mask, binary_mask, 0, 255, cv::THRESH_BINARY);
        // cv::imshow("Mask Visualization", binary_mask);
        // cv::imshow("Depth Visualization", depth_map);
        // cv::waitKey(1);

        cv_bridge::CvImage out_image;
        out_image.header = mask_msg->header;
        out_image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        depth_map.setTo(0, depth_map == std::numeric_limits<float>::infinity());
        out_image.image = depth_map;
        // depth_image_pub_.publish(out_image.toImageMsg());
    }

    std::pair<int, int> projectToImage(float x, float y, float z, const cv::Mat& K)
    {
        float fx = K.at<double>(0, 0);
        float fy = K.at<double>(1, 1);
        float cx = K.at<double>(0, 2);
        float cy = K.at<double>(1, 2);

        int u = static_cast<int>((x / z) * fx + cx);
        int v = static_cast<int>((y / z) * fy + cy);
        return {u, v};
    }

    Eigen::Matrix4f convertTfToMatrix(const tf::StampedTransform& tf)
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

    void publishInstanceCloud(int label, const std::vector<std::array<float, 3>>& points, const std_msgs::Header& header)
    {
        std::string topic = "/segmented_cloud/" + std::to_string(label);
        if (instance_pubs_.find(label) == instance_pubs_.end()) {
            instance_pubs_[label] = nh_.advertise<sensor_msgs::PointCloud2>(topic, 1);
            ROS_INFO("Created topic: %s", topic.c_str());
        }

        sensor_msgs::PointCloud2 cloud;
        cloud.header = header;
        cloud.height = 1;
        cloud.width = points.size();

        sensor_msgs::PointCloud2Modifier modifier(cloud);
        modifier.setPointCloud2FieldsByString(1, "xyz");
        modifier.resize(points.size());

        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

        for (const auto& pt : points) {
            *iter_x = pt[0];
            *iter_y = pt[1];
            *iter_z = pt[2];
            ++iter_x; ++iter_y; ++iter_z;
        }

        instance_pubs_[label].publish(cloud);
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
