#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <thread>

namespace gazebo
{
  class TrainResetPlugin : public ModelPlugin
  {
  public:
    TrainResetPlugin() : ModelPlugin(), active_(false), doors_active_(false), x_velocity_(-24.72), deceleration_(1.3) {}

    void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
    {
      model_ = model;

      if (!rclcpp::ok())
        rclcpp::init(0, nullptr);

      ros_node_ = std::make_shared<rclcpp::Node>("train_reset_plugin_node");

      if (sdf->HasElement("link_name"))
      {
        link_name_ = sdf->Get<std::string>("link_name");
      }
      else
      {
        RCLCPP_ERROR(ros_node_->get_logger(), "No <link_name> specified in plugin.");
        return;
      }

      link_ = model_->GetLink(link_name_);
      if (!link_)
      {
        RCLCPP_ERROR(ros_node_->get_logger(), "Link %s not found!", link_name_.c_str());
        return;
      }

      reset_sub_ = ros_node_->create_subscription<std_msgs::msg::Empty>(
        "/train/reset", 10,
        std::bind(&TrainResetPlugin::OnReset, this, std::placeholders::_1));

      update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&TrainResetPlugin::OnUpdate, this));

      ros_spin_thread_ = std::make_shared<std::thread>(&TrainResetPlugin::SpinRosNode, this);

      RCLCPP_INFO(ros_node_->get_logger(), "TrainResetPlugin loaded and listening to /train/reset.");
    }

    void OnReset(const std_msgs::msg::Empty::SharedPtr /*msg*/)
    {
      if (!link_)
        return;

      // Teleport to x = 235
      ignition::math::Pose3d pose = link_->WorldPose();
      pose.Pos().X(235.0);
      link_->SetWorldPose(pose);

      // Set initial velocity
      link_->SetLinearVel(ignition::math::Vector3d(x_velocity_, 0, 0));
      link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

      // Activate motion logic
      start_time_ = model_->GetWorld()->SimTime();
      active_ = true;
      doors_active_ = false;
      RCLCPP_INFO(ros_node_->get_logger(), "Train reset triggered.");
    }

    void OnUpdate()
    {
      if (doors_active_) {
        return;
      }

      if (!active_ || !link_)
        return;

      auto current_time = model_->GetWorld()->SimTime();
      double elapsed = (current_time - start_time_).Double();

      // v = v0 - a * t
      double v = -(x_velocity_ + deceleration_ * elapsed );  // x_velocity_ is negative
      RCLCPP_INFO(ros_node_->get_logger(), "Elapsed time: %fs", elapsed);
      RCLCPP_INFO(ros_node_->get_logger(), "Train velocity is v = %f", v);
      v = std::max(v, 0.0); // prevent reversing

      link_->SetLinearVel(ignition::math::Vector3d(-v, 0, 0));

      double current_x = link_->WorldPose().Pos().X();
      if (current_x <= 0.0 || v == 0.0)
      {
        link_->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
        active_ = false;
        if (link_name_ == "left_link" || link_name_ == "right_link") {
          doors_active_ = true;
        }
        RCLCPP_INFO(ros_node_->get_logger(), "Train stopped at x = %f", current_x);
      }
    }

    void SpinRosNode()
    {
      rclcpp::spin(ros_node_);
    }

  private:
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
    std::shared_ptr<std::thread> ros_spin_thread_;

    physics::ModelPtr model_;
    physics::LinkPtr link_;
    event::ConnectionPtr update_connection_;

    std::string link_name_;
    bool active_;
    bool doors_active_;
    double x_velocity_;
    double deceleration_;
    gazebo::common::Time start_time_;
  };

  GZ_REGISTER_MODEL_PLUGIN(TrainResetPlugin)
}