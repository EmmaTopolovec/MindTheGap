#include <string>
#include <gazebo/common/Plugin.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include "std_msgs/msg/bool.hpp"
#include <thread>
#include <rclcpp/rclcpp.hpp>

namespace gazebo
{
  class ContactPlugin : public SensorPlugin
  {
    public:
      ContactPlugin() : SensorPlugin() {}

      void Load(sensors::SensorPtr _sensor, sdf::ElementPtr /*_sdf*/)
      {

        if (!rclcpp::ok())
          rclcpp::init(0, nullptr);

        ros_node_ = std::make_shared<rclcpp::Node>("contact_plugin_node");
        collision_pub_ = ros_node_->create_publisher<std_msgs::msg::Bool>("/train/collision", 10);

        // Get the parent sensor.
        this->parentSensor =
          std::dynamic_pointer_cast<sensors::ContactSensor>(_sensor);
      
        // Make sure the parent sensor is valid.
        if (!this->parentSensor)
        {
          gzerr << "ContactPlugin requires a ContactSensor.\n";
          return;
        }
      
        // Connect to the sensor update event.
        this->updateConnection = this->parentSensor->ConnectUpdated(
            std::bind(&ContactPlugin::OnUpdate, this));
      
        // Make sure the parent sensor is active.
        this->parentSensor->SetActive(true);

        ros_spin_thread_ = std::make_shared<std::thread>(&ContactPlugin::SpinRosNode, this);
        RCLCPP_INFO(ros_node_->get_logger(), "ContactPlugin loaded and listening to /train/collision.");
      }
      
      /////////////////////////////////////////////////
      void OnUpdate()
      {
        // Get all the contacts.
        msgs::Contacts contacts;
        contacts = this->parentSensor->Contacts();
        // for (unsigned int i = 0; i < contacts.contact_size(); ++i)
        // {
        //   std::cout << "Collision between[" << contacts.contact(i).collision1()
        //             << "] and [" << contacts.contact(i).collision2() << "]\n";
      
        //   for (unsigned int j = 0; j < contacts.contact(i).position_size(); ++j)
        //   {
        //     std::cout << j << "  Position:"
        //               << contacts.contact(i).position(j).x() << " "
        //               << contacts.contact(i).position(j).y() << " "
        //               << contacts.contact(i).position(j).z() << "\n";
        //     std::cout << "   Normal:"
        //               << contacts.contact(i).normal(j).x() << " "
        //               << contacts.contact(i).normal(j).y() << " "
        //               << contacts.contact(i).normal(j).z() << "\n";
        //     std::cout << "   Depth:" << contacts.contact(i).depth(j) << "\n";
        //   }
        // }
        if (contacts.contact_size() > 0) {
          std_msgs::msg::Bool msg;
          msg.data = true;  // Collision detected
          collision_pub_->publish(msg);
        }
      }

      void SpinRosNode()
      {
        rclcpp::spin(ros_node_);
      }
    private:
      rclcpp::Node::SharedPtr ros_node_;
      sensors::ContactSensorPtr parentSensor;
      event::ConnectionPtr updateConnection;
      rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr collision_pub_;
      std::shared_ptr<std::thread> ros_spin_thread_;
  };
  GZ_REGISTER_SENSOR_PLUGIN(ContactPlugin)
}