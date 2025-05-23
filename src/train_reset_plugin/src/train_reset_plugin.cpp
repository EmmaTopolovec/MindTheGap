#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/empty.hpp>
#include "std_msgs/msg/bool.hpp"
#include <thread>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>

namespace gazebo
{
  class TrainResetPlugin : public ModelPlugin
  {
  public:
    TrainResetPlugin() : ModelPlugin(), active_(false), solid_train_active_(false), door_state_(0), x_velocity_(-16.12), deceleration_(1.3) {}

    void MoveUp(const physics::LinkPtr &link)
    {
      ignition::math::Pose3d current_pose = link->WorldPose();

      double z_offset = 9999.0;
      current_pose.Pos().Z(current_pose.Pos().Z() + z_offset);  // hide

      link->SetWorldPose(current_pose);
    }

    void SpawnInvisibleTrain()
    {
      RCLCPP_INFO(ros_node_->get_logger(), "SPAWNING STATIC TRAIN...");
      if (invisible_train_spawned_) {
        RCLCPP_INFO(ros_node_->get_logger(), "STATIC TRAIN ALREADY SPAWNED.");
        return;
      }

      gazebo::msgs::Factory msg;
      msg.set_sdf_filename("model://invisible_train");
      gazebo::msgs::Set(msg.mutable_pose(), ignition::math::Pose3d(0, 0, 0.0, 0, 0, 0));  // Adjust position as needed
      RCLCPP_INFO(ros_node_->get_logger(), "PUBLISHING MESSAGE TO SPAWN STATIC TRAIN.");
      factory_pub_->Publish(msg);
      invisible_train_spawned_ = true;
    }
    
    void DeleteInvisibleTrain()
    {
      RCLCPP_INFO(ros_node_->get_logger(), "DELETING STATIC TRAIN...");
      if (!invisible_train_spawned_) {
        RCLCPP_INFO(ros_node_->get_logger(), "STATIC TRAIN ALREADY DELETED.");
        return;
      }
    
      auto delete_msg = gazebo::msgs::CreateRequest("entity_delete", "invisible_train");
      RCLCPP_INFO(ros_node_->get_logger(), "PUBLISHING MESSAGE TO DELETE STATIC TRAIN.");

      request_pub_->Publish(*delete_msg);
      invisible_train_spawned_ = false;
    }
    
    void Load(physics::ModelPtr model, sdf::ElementPtr /*sdf*/) override
    {
      model_ = model;

      gz_node_ = gazebo::transport::NodePtr(new gazebo::transport::Node());
      gz_node_->Init(model_->GetWorld()->Name());
      factory_pub_ = gz_node_->Advertise<gazebo::msgs::Factory>("~/factory");
      request_pub_ = gz_node_->Advertise<gazebo::msgs::Request>("~/request");


      if (!rclcpp::ok())
        rclcpp::init(0, nullptr);

      ros_node_ = std::make_shared<rclcpp::Node>("train_reset_plugin_node");

      train_leaving_pub_ = ros_node_->create_publisher<std_msgs::msg::Bool>("/train/leaving", 10);

      solid_train_link_ = model_->GetLink("solid_train_link");
      // train_link_ = model_->GetLink("train_link");
      left_door_link_ = model_->GetLink("left_link");
      right_door_link_ = model_->GetLink("right_link");

      // if (!solid_train_link_ || !train_link_ || !left_door_link_ || !right_door_link_)
      // {
      //   RCLCPP_ERROR(ros_node_->get_logger(), "One or more required links not found in the model.");
      //   return;
      // }

      // MoveUp(train_link_);
      MoveUp(left_door_link_);
      MoveUp(right_door_link_);

      // train_link_->SetKinematic(true);
      left_door_link_->SetKinematic(true);
      right_door_link_->SetKinematic(true);

      solid_train_link_->SetKinematic(false);

      reset_sub_ = ros_node_->create_subscription<std_msgs::msg::Empty>(
        "/train/reset", 10,
        std::bind(&TrainResetPlugin::OnReset, this, std::placeholders::_1));

      start_sub_ = ros_node_->create_subscription<std_msgs::msg::Empty>(
        "/train/start", 10,
        std::bind(&TrainResetPlugin::OnStart, this, std::placeholders::_1));

      update_connection_ = event::Events::ConnectWorldUpdateBegin(
        std::bind(&TrainResetPlugin::OnUpdate, this));

      ros_spin_thread_ = std::make_shared<std::thread>(&TrainResetPlugin::SpinRosNode, this);

      RCLCPP_INFO(ros_node_->get_logger(), "TrainResetPlugin loaded and listening to /train/reset and /train/start.");
    }

    void OnReset(const std_msgs::msg::Empty::SharedPtr /*msg*/)
    {
      solid_train_active_ = false;
      active_ = false;
      door_state_ = 0;

      DeleteInvisibleTrain();

      // MoveUp(train_link_);
      MoveUp(left_door_link_);
      MoveUp(right_door_link_);

      // train_link_->SetLinearVel({0, 0, 0});
      left_door_link_->SetLinearVel({0, 0, 0});
      right_door_link_->SetLinearVel({0, 0, 0});

      // train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
      left_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
      right_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

      // train_link_->SetKinematic(true);
      left_door_link_->SetKinematic(true);
      right_door_link_->SetKinematic(true);

      solid_train_link_->SetKinematic(false);

      // Move solid_train to x = 100
      solid_train_link_->SetWorldPose(ignition::math::Pose3d(100.0, 0, 0, 0, 0, 0));
      solid_train_link_->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
      solid_train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

      
      RCLCPP_INFO(ros_node_->get_logger(), "Train reset triggered.");
    }

    void OnStart(const std_msgs::msg::Empty::SharedPtr /*msg*/)
    {
      solid_train_active_ = true;
      solid_train_link_->SetLinearVel(ignition::math::Vector3d(x_velocity_, 0, 0));
      solid_train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
      start_time_ = model_->GetWorld()->SimTime();
      RCLCPP_INFO(ros_node_->get_logger(), "Train start triggered.");
    }

    void OnUpdate()
    {
      auto sim_time = model_->GetWorld()->SimTime();

      if (solid_train_active_)
      {
        
        auto current_time = model_->GetWorld()->SimTime();
        double elapsed = (current_time - start_time_).Double();
        double v = -(x_velocity_ + deceleration_ * elapsed);
        v = std::max(v, 0.0);
        solid_train_link_->SetLinearVel(ignition::math::Vector3d(-v, 0, 0));

        double current_x = solid_train_link_->WorldPose().Pos().X();
        if (current_x <= 0.0 || v == 0.0)
        {
          solid_train_link_->SetLinearVel({0, 0, 0});
          solid_train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
          solid_train_link_->SetWorldPose({0, 0.005, 0, 0, 0, 0});
          
          MoveUp(solid_train_link_);
          solid_train_link_->SetKinematic(true);

          // Show stationary train
          // train_link_->SetWorldPose({0, 0.005, 0, 0, 0, 0});
          left_door_link_->SetWorldPose({0, -0.01, 0.01, 0, 0, 0});
          right_door_link_->SetWorldPose({0, -0.01, 0.01, 0, 0, 0});

          // train_link_->SetKinematic(true);
          left_door_link_->SetKinematic(false);
          right_door_link_->SetKinematic(false);
          // train_link_->SetLinearVel({0, 0, 0});
          left_door_link_->SetLinearVel({0, 0, 0});
          right_door_link_->SetLinearVel({0, 0, 0});
          // train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
          left_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
          right_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

          SpawnInvisibleTrain();

          door_state_ = 1;
          door_timer_start_ = sim_time;

          solid_train_active_ = false;

          RCLCPP_INFO(ros_node_->get_logger(), "Solid train arrived. Switching to kinematic train and starting door sequence.");
        }
        return;
      }

      // Door animation logic
      if (door_state_ > 0)
      {

        double t = (sim_time - door_timer_start_).Double();
        double speed = 0.25;

        if (door_state_ == 1 && t < 1.0) {
          left_door_link_->SetWorldPose({0, -0.01, 0.01, 0, 0, 0});
          right_door_link_->SetWorldPose({0, -0.01, 0.01, 0, 0, 0});
          left_door_link_->SetLinearVel({0, 0, 0});
          right_door_link_->SetLinearVel({0, 0, 0});
          left_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
          right_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
        }
        else if (door_state_ == 2)
        {
          if (left_door_link_)
            left_door_link_->SetLinearVel({-speed, 0, 0});
          if (right_door_link_)
            right_door_link_->SetLinearVel({speed, 0, 0});

          double moved = std::abs(left_door_link_->WorldPose().Pos().X());
          if (moved >= 0.75)
          {
            // train_link_->SetWorldPose({0, 0.005, 0, 0, 0, 0});
            left_door_link_->SetWorldPose({-0.75, -0.01, 0.01, 0, 0, 0});
            right_door_link_->SetWorldPose({0.75, -0.01, 0.01, 0, 0, 0});

            // train_link_->SetLinearVel({0, 0, 0});
            left_door_link_->SetLinearVel({0, 0, 0});
            right_door_link_->SetLinearVel({0, 0, 0});

            // train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
            left_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
            right_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

            door_state_ = 3;
            door_timer_start_ = sim_time;
            RCLCPP_INFO(ros_node_->get_logger(), "Doors opened. Mind the gap!");
          }
        }
        else if (door_state_ == 3 && t < 5.0) {}
        else if (door_state_ == 4)
        {
          if (left_door_link_)
            left_door_link_->SetLinearVel({speed, 0, 0});
          if (right_door_link_)
            right_door_link_->SetLinearVel({-speed, 0, 0});

          double moved = std::abs(left_door_link_->WorldPose().Pos().X());
          if (moved <= 0.01)
          {
            // train_link_->SetWorldPose({0, 0.005, 0, 0, 0, 0});
            left_door_link_->SetWorldPose({0, -0.01, 0.01, 0, 0, 0});
            right_door_link_->SetWorldPose({0, -0.01, 0.01, 0, 0, 0});

            // train_link_->SetLinearVel({0, 0, 0});
            left_door_link_->SetLinearVel({0, 0, 0});
            right_door_link_->SetLinearVel({0, 0, 0});

            // train_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
            left_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
            right_door_link_->SetAngularVel(ignition::math::Vector3d(0, 0, 0));

            door_state_ = 0;
            RCLCPP_INFO(ros_node_->get_logger(), "Doors closed. Sequence complete.");

            std_msgs::msg::Bool msg;
            msg.data = true;
            train_leaving_pub_->publish(msg);
          }
        }
        else
        {
          door_state_++;
          door_timer_start_ = sim_time;
        }

        return;
      }
    }

    void SpinRosNode()
    {
      rclcpp::spin(ros_node_);
    }

  private:
    rclcpp::Node::SharedPtr ros_node_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_sub_;
    rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr start_sub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr train_leaving_pub_;
    // rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_pub_;
    std::shared_ptr<std::thread> ros_spin_thread_;

    physics::ModelPtr model_;
    physics::LinkPtr solid_train_link_;
    // physics::LinkPtr train_link_;
    physics::LinkPtr left_door_link_;
    physics::LinkPtr right_door_link_;

    gazebo::transport::NodePtr gz_node_;
    gazebo::transport::PublisherPtr factory_pub_;
    gazebo::transport::PublisherPtr request_pub_;
    bool invisible_train_spawned_ = false;


    event::ConnectionPtr update_connection_;

    bool active_;
    bool solid_train_active_;
    int door_state_;
    double x_velocity_;
    double deceleration_;
    gazebo::common::Time start_time_;
    gazebo::common::Time door_timer_start_;
  };

  GZ_REGISTER_MODEL_PLUGIN(TrainResetPlugin)
}
