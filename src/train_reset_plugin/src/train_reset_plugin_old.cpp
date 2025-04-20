#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/PID.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include <thread>

namespace gazebo
{
  class TrainResetPlugin : public ModelPlugin
  {
  public:
    TrainResetPlugin() : ModelPlugin(), active_(false), x_velocity_(-16.12), deceleration_(1.3) {}

    void Load(physics::ModelPtr model, sdf::ElementPtr /*sdf*/) override
    {
      model_ = model;

      if (!rclcpp::ok())
        rclcpp::init(0, nullptr);

      ros_node_ = std::make_shared<rclcpp::Node>("train_reset_plugin_node");

      train_link_ = model_->GetLink("train_link");
      if (!train_link_) {
        RCLCPP_ERROR(ros_node_->get_logger(), "Train link not found!");
        return;
      }

      left_door_link_ = model_->GetLink("left_door_link");
      if (!left_door_link_) {
        RCLCPP_ERROR(ros_node_->get_logger(), "Left door link not found!");
        return;
      }

      right_door_link_ = model_->GetLink("right_door_link");
      if (!right_door_link_) {
        RCLCPP_ERROR(ros_node_->get_logger(), "Right door link not found!");
        return;
      }

      joint_controller_ = model_->GetJointController();
      if (!joint_controller_) {
        RCLCPP_ERROR(ros_node_->get_logger(), "Failed to get JointController");
        return;
      }

      // Set up joints and PID
      left_joint_name_ = "train::left_door_joint";
      right_joint_name_ = "train::right_door_joint";

      joint_controller_->SetPositionPID(left_joint_name_, common::PID(10.0, 0.0, 1.0));
      joint_controller_->SetPositionPID(right_joint_name_, common::PID(10.0, 0.0, 1.0));

      reset_sub_ = ros_node_->create_subscription<std_msgs::msg::Empty>(
        "/train/reset", 10,
        std::bind(&TrainResetPlugin::OnReset, this, std::placeholders::_1));

      update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&TrainResetPlugin::OnUpdate, this));

      ros_spin_thread_ = std::make_shared<std::thread>(&TrainResetPlugin::SpinRosNode, this);

      RCLCPP_INFO(ros_node_->get_logger(), "TrainResetPlugin loaded.");
    }

    void OnReset(const std_msgs::msg::Empty::SharedPtr /*msg*/)
    {
      joint_controller_->Reset();
      
      train_link_->SetKinematic(false);
      left_door_link_->SetKinematic(true);
      right_door_link_->SetKinematic(true);

      train_link_->SetWorldPose(ignition::math::Pose3d(100, 0, 0, 0, 0, 0));
      train_link_->SetLinearVel(ignition::math::Vector3d(x_velocity_, 0, 0));
      train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

      start_time_ = model_->GetWorld()->SimTime();
      door_timer_start_ = start_time_;
      door_state_ = 0;
      active_ = true;

      joint_controller_->SetPositionTarget(left_joint_name_, 0.0);
      joint_controller_->SetPositionTarget(right_joint_name_, 0.0);

      RCLCPP_INFO(ros_node_->get_logger(), "Train reset initiated.");
    }

    void OnUpdate()
    {
      auto sim_time = model_->GetWorld()->SimTime();
      double t = (sim_time - door_timer_start_).Double();

      if (door_state_ == 0 && active_)
      {
        double elapsed = (sim_time - start_time_).Double();
        double v = -(x_velocity_ + deceleration_ * elapsed);
        v = std::max(v, 0.0);

        train_link_->SetLinearVel(ignition::math::Vector3d(-v, 0, 0));

        double current_x = train_link_->WorldPose().Pos().X();
        if (current_x <= 0.0 || v == 0.0) {
          train_link_->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
          train_link_->SetWorldPose(ignition::math::Pose3d(0, 0, 0, 0, 0, 0));
          train_link_->SetKinematic(true);
          left_door_link_->SetKinematic(false);
          right_door_link_->SetKinematic(false);


          active_ = false;
          door_state_ = 1;
          door_timer_start_ = sim_time;
          RCLCPP_INFO(ros_node_->get_logger(), "Train stopped at station.");
        }

        return;
      }

      switch (door_state_) {
        case 1:
          if (t > 2.0) {
            door_state_ = 2;
            joint_controller_->SetPositionTarget(left_joint_name_, -0.75);
            joint_controller_->SetPositionTarget(right_joint_name_, 0.75);
            RCLCPP_INFO(ros_node_->get_logger(), "Opening doors.");
            door_timer_start_ = sim_time;
          }
          break;

        case 2:
          if (IsJointAtTarget(left_joint_name_, -0.75) && IsJointAtTarget(right_joint_name_, 0.75)) {
            door_state_ = 3;
            door_timer_start_ = sim_time;
            RCLCPP_INFO(ros_node_->get_logger(), "Doors opened.");
          }
          break;

        case 3:
          if (t > 3.0) {
            door_state_ = 4;
            joint_controller_->SetPositionTarget(left_joint_name_, 0.0);
            joint_controller_->SetPositionTarget(right_joint_name_, 0.0);
            RCLCPP_INFO(ros_node_->get_logger(), "Closing doors.");
            door_timer_start_ = sim_time;
          }
          break;

        case 4:
          if (IsJointAtTarget(left_joint_name_, 0.0) && IsJointAtTarget(right_joint_name_, 0.0)) {
            door_state_ = 0;
            RCLCPP_INFO(ros_node_->get_logger(), "Doors closed. Sequence complete.");
          }
          break;
      }
    }

    bool IsJointAtTarget(const std::string &joint_name, double target)
    {
      auto joint = model_->GetJoint(joint_name);
      if (!joint)
      {
        RCLCPP_ERROR(ros_node_->get_logger(), "Joint %s not found!", joint_name.c_str());
        return false;
      }

      double current = joint->Position(0);  // 0 = axis index
      return std::abs(current - target) < 0.01;
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
    physics::LinkPtr train_link_;
    physics::LinkPtr left_door_link_;
    physics::LinkPtr right_door_link_;
    physics::JointControllerPtr joint_controller_;
    std::string left_joint_name_, right_joint_name_;

    event::ConnectionPtr update_connection_;

    bool active_;
    int door_state_ = 0;
    double x_velocity_, deceleration_;
    gazebo::common::Time start_time_, door_timer_start_;
  };

  GZ_REGISTER_MODEL_PLUGIN(TrainResetPlugin)
}



// #include <gazebo/common/Plugin.hh>
// #include <gazebo/gazebo.hh>
// #include <gazebo/physics/physics.hh>
// #include <rclcpp/rclcpp.hpp>
// #include <std_msgs/msg/empty.hpp>
// #include <thread>

// namespace gazebo
// {
//   class TrainResetPlugin : public ModelPlugin
//   {
//   public:
//     TrainResetPlugin() : ModelPlugin(), active_(false), door_state_(false), x_velocity_(-16.12), deceleration_(1.3) {}

//     void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
//     {
//       model_ = model;

//       if (!rclcpp::ok())
//         rclcpp::init(0, nullptr);

//       ros_node_ = std::make_shared<rclcpp::Node>("train_reset_plugin_node");

//       if (sdf->HasElement("link_name"))
//       {
//         link_name_ = sdf->Get<std::string>("link_name");
//       }
//       else
//       {
//         RCLCPP_ERROR(ros_node_->get_logger(), "No <link_name> specified in plugin.");
//         return;
//       }

//       link_ = model_->GetLink(link_name_);
//       if (!link_)
//       {
//         RCLCPP_ERROR(ros_node_->get_logger(), "Link %s not found!", link_name_.c_str());
//         return;
//       }

//       reset_sub_ = ros_node_->create_subscription<std_msgs::msg::Empty>(
//         "/train/reset", 10,
//         std::bind(&TrainResetPlugin::OnReset, this, std::placeholders::_1));

//       update_connection_ = event::Events::ConnectWorldUpdateBegin(
//         std::bind(&TrainResetPlugin::OnUpdate, this));

//       ros_spin_thread_ = std::make_shared<std::thread>(&TrainResetPlugin::SpinRosNode, this);

//       RCLCPP_INFO(ros_node_->get_logger(), "TrainResetPlugin loaded and listening to /train/reset.");
//     }

//     void OnReset(const std_msgs::msg::Empty::SharedPtr /*msg*/)
//     {
//       if (!link_)
//         return;

//       // Teleport to x = 100
//       ignition::math::Pose3d pose = link_->WorldPose();
//       pose.Pos().X(100.0);
//       link_->SetWorldPose(pose);

//       // Set initial velocity
//       link_->SetLinearVel(ignition::math::Vector3d(x_velocity_, 0, 0));
//       link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

//       // Activate motion logic
//       start_time_ = model_->GetWorld()->SimTime();
//       active_ = true;
//       door_state_ = false;
//       RCLCPP_INFO(ros_node_->get_logger(), "Train reset triggered.");
//     }

//     void OnUpdate()
//     {
//       auto sim_time = model_->GetWorld()->SimTime();

//       if (door_state_) {
//         double t = (sim_time - door_timer_start_).Double();
//         double speed = 0.25;

//         if (door_state_ == 1 && t < 1.0) {}
//         else if (door_state_ == 2) { // Opening doors
//           // link_->SetKinematic(false);
//           RCLCPP_INFO(ros_node_->get_logger(), "Opening doors. Mind the gap.");
//           if (link_name_ == "left_link")
//             link_->SetLinearVel(ignition::math::Vector3d(-speed, 0, 0));
//           if (link_name_ == "right_link")
//             link_->SetLinearVel(ignition::math::Vector3d(speed, 0, 0));
          
//           double moved = std::abs(link_->WorldPose().Pos().X());
//           RCLCPP_INFO(ros_node_->get_logger(), "moved=%f", moved);
//           if (moved >= 0.75) {
//             link_->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
//             link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
//             // link_->SetKinematic(true);

//             if (link_name_ == "left_link")
//               link_->SetWorldPose(ignition::math::Pose3d(-0.75, 0, 0, 0, 0, 0));
//             if (link_name_ == "right_link")
//               link_->SetWorldPose(ignition::math::Pose3d(0.75, 0, 0, 0, 0, 0));
//             door_state_ = 3;
//             door_timer_start_ = sim_time;
//             RCLCPP_INFO(ros_node_->get_logger(), "Doors open.");
//           }
//         }
//         else if (door_state_ == 3 && t < 1.0) {} // Wait with doors open
//         else if (door_state_ == 4) { // Close doors
//           // link_->SetKinematic(false);
//           if (link_name_ == "left_link")
//             link_->SetLinearVel(ignition::math::Vector3d(speed, 0, 0));
//           if (link_name_ == "right_link")
//             link_->SetLinearVel(ignition::math::Vector3d(-speed, 0, 0));
          
//           double moved = std::abs(link_->WorldPose().Pos().X());
//           if (moved <= 0.01) {
//             link_->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
//             link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
//             // link_->SetKinematic(true);

//             if (link_name_ == "left_link")
//               link_->SetWorldPose(ignition::math::Pose3d(0, 0, 0, 0, 0, 0));
//             if (link_name_ == "right_link")
//               link_->SetWorldPose(ignition::math::Pose3d(0, 0, 0, 0, 0, 0));
//             door_state_ = 0;
//             RCLCPP_INFO(ros_node_->get_logger(), "Doors closed. Sequence complete.");
//           }
//         } 
//         else {
//           door_state_++;
//           RCLCPP_INFO(ros_node_->get_logger(), "door_state_++;");

//         }
//         return;
//       }

//       if (!active_ || !link_)
//         return;

//       auto current_time = model_->GetWorld()->SimTime();
//       double elapsed = (current_time - start_time_).Double();

//       // v = v0 - a * t
//       double v = -(x_velocity_ + deceleration_ * elapsed );  // x_velocity_ is negative
//       // RCLCPP_INFO(ros_node_->get_logger(), "Elapsed time: %fs", elapsed);
//       // RCLCPP_INFO(ros_node_->get_logger(), "Train velocity is v = %f", v);
//       v = std::max(v, 0.0);

//       link_->SetLinearVel(ignition::math::Vector3d(-v, 0, 0));

//       double current_x = link_->WorldPose().Pos().X();
//       if (current_x <= 0.0 || v == 0.0)
//       {
//         link_->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
//         link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
//         // link_->SetKinematic(true);
//         link_->SetWorldPose(ignition::math::Pose3d(0, 0, 0, 0, 0, 0));
//         active_ = false;
//         if (link_name_ == "left_link" || link_name_ == "right_link") {
//           door_state_ = 1;
//           door_timer_start_ = sim_time;;
//         }
//         RCLCPP_INFO(ros_node_->get_logger(), "Train stopped at x = %f", current_x);
//       }
//     }

//     void SpinRosNode()
//     {
//       rclcpp::spin(ros_node_);
//     }

//   private:
//     rclcpp::Node::SharedPtr ros_node_;
//     rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
//     std::shared_ptr<std::thread> ros_spin_thread_;

//     physics::ModelPtr model_;
//     physics::LinkPtr link_;
//     event::ConnectionPtr update_connection_;

//     std::string link_name_;
//     bool active_;
//     int door_state_;
//     double x_velocity_;
//     double deceleration_;
//     gazebo::common::Time start_time_;
//     gazebo::common::Time door_timer_start_;
//   };

//   GZ_REGISTER_MODEL_PLUGIN(TrainResetPlugin)
// }