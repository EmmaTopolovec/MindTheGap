#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <geometry_msgs/msg/pose.hpp>  // <-- Add this for geometry_msgs::msg::Pose
#include <thread>


namespace gazebo
{
  class BotResetPlugin : public ModelPlugin
  {
  public:
    BotResetPlugin() : ModelPlugin() {}

    void Load(physics::ModelPtr model, sdf::ElementPtr /*sdf*/) override
    {
      model_ = model;

      if (!rclcpp::ok())
      {
        rclcpp::init(0, nullptr);
      }

      ros_node_ = std::make_shared<rclcpp::Node>("bot_reset_plugin_node");

      pose_sub_ = ros_node_->create_subscription<geometry_msgs::msg::Pose>(
        "/set_bot_position", 10,
        std::bind(&BotResetPlugin::SetPositionCallback, this, std::placeholders::_1));

      ros_spin_thread_ = std::make_shared<std::thread>(&BotResetPlugin::SpinRosNode, this);
      
      RCLCPP_INFO(ros_node_->get_logger(), "BotResetPlugin loaded and listening to /set_bot_position.");
    }

    void SetPositionCallback(const geometry_msgs::msg::Pose::SharedPtr msg)
    {
      // Convert quaternion to yaw only, then reconstruct flat quaternion
      ignition::math::Quaterniond input_quat(
        msg->orientation.w,
        msg->orientation.x,
        msg->orientation.y,
        msg->orientation.z);

      ignition::math::Vector3d rpy = input_quat.Euler();
      double roll = rpy.X();
      double pitch = rpy.Y();
      double yaw = rpy.Z();        

      // Rebuild a flat yaw-only quaternion
      ignition::math::Quaterniond flat_quat(0, 0, yaw);

      ignition::math::Vector3d position(msg->position.x, msg->position.y, msg->position.z);
      ignition::math::Pose3d pose(position, flat_quat);

      model_->SetWorldPose(pose);

      auto link = model_->GetLink("base_link");
      if (link)
      {
        link->SetLinearVel(ignition::math::Vector3d::Zero);
        link->SetAngularVel(ignition::math::Vector3d::Zero);
      }

      RCLCPP_INFO(
        ros_node_->get_logger(),
        "Set bot pose to (x: %f, y: %f, z: %f, yaw: %f)",
        pose.Pos().X(), pose.Pos().Y(), pose.Pos().Z(), yaw);
    }

    void SpinRosNode()
    {
      // Spin the ROS node in its own thread
      rclcpp::spin(ros_node_);
    }

    ~BotResetPlugin()
    {
      if (ros_spin_thread_ && ros_spin_thread_->joinable())
      {
        ros_spin_thread_->join();
      }
    }

  private:
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
    std::shared_ptr<std::thread> ros_spin_thread_;

    physics::ModelPtr model_;
  };

  GZ_REGISTER_MODEL_PLUGIN(BotResetPlugin)
}
