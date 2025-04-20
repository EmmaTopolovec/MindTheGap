#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>

namespace gazebo
{
  enum TrainState { IDLE, MOVING, ARRIVED, DOOR_OPENING, DOOR_WAITING, DOOR_CLOSING };

  class TrainControllerPlugin : public WorldPlugin
  {
  public:
    void Load(physics::WorldPtr _world, sdf::ElementPtr _sdf) override
    {
      this->world = _world;
      this->train = _world->ModelByName("train");
      this->leftDoor = _world->ModelByName("left");
      this->rightDoor = _world->ModelByName("right");

      if (!train || !leftDoor || !rightDoor)
      {
        gzerr << "Train or door models not found!\n";
        return;
      }

      // Setup transport node
      this->node = transport::NodePtr(new transport::Node());
      this->node->Init();

      auto sub = this->node->Subscribe("~/train/reset", &TrainControllerPlugin::OnReset, this);
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&TrainControllerPlugin::OnUpdate, this));
    }

    void OnReset(ConstEmptyPtr &msg)
    {
      train->SetWorldPose(ignition::math::Pose3d(100, 0, 0, 0, 0, 0));
      train->SetLinearVel(ignition::math::Vector3d(-initialVelocity, 0, 0));
      this->currentState = MOVING;
      this->startTime = world->SimTime();
    }

    void OnUpdate()
    {
      common::Time simTime = world->SimTime();
      double dt = (simTime - lastUpdate).Double();
      lastUpdate = simTime;

      ignition::math::Pose3d trainPose = train->WorldPose();

      if (currentState == MOVING)
      {
        double elapsed = (simTime - startTime).Double();
        double velocity = std::max(0.0, initialVelocity + acceleration * elapsed);
        double dx = initialVelocity * elapsed + 0.5 * acceleration * elapsed * elapsed;
        double x = 100 - dx;

        if (x <= 0.0)
        {
          train->SetWorldPose(ignition::math::Pose3d(0, 0, 0, 0, 0, 0));
          train->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
          currentState = ARRIVED;
          waitStart = simTime;
        }
        else
        {
          train->SetWorldPose(ignition::math::Pose3d(x, 0, 0, 0, 0, 0));
          train->SetLinearVel(ignition::math::Vector3d(-velocity, 0, 0));
        }

        UpdateDoorPos(x);
      }
      else if (currentState == ARRIVED && (simTime - waitStart).Double() > 1.0)
      {
        currentState = DOOR_OPENING;
        doorStart = simTime;
      }
      else if (currentState == DOOR_OPENING)
      {
        double progress = std::min(1.0, (simTime - doorStart).Double());
        leftDoor->SetWorldPose(train->WorldPose() + ignition::math::Pose3d(-0.75 * progress, 0, 0, 0, 0, 0));
        rightDoor->SetWorldPose(train->WorldPose() + ignition::math::Pose3d(0.75 * progress, 0, 0, 0, 0, 0));

        if (progress >= 1.0)
        {
          currentState = DOOR_WAITING;
          doorStart = simTime;
        }
      }
      else if (currentState == DOOR_WAITING && (simTime - doorStart).Double() > 1.0)
      {
        currentState = DOOR_CLOSING;
        doorStart = simTime;
      }
      else if (currentState == DOOR_CLOSING)
      {
        double progress = std::min(1.0, (simTime - doorStart).Double());
        leftDoor->SetWorldPose(train->WorldPose() + ignition::math::Pose3d(-0.75 * (1.0 - progress), 0, 0, 0, 0, 0));
        rightDoor->SetWorldPose(train->WorldPose() + ignition::math::Pose3d(0.75 * (1.0 - progress), 0, 0, 0, 0, 0));

        if (progress >= 1.0)
        {
          currentState = IDLE;
        }
      }
    }

    void UpdateDoorPos(double trainX)
    {
      leftDoor->SetWorldPose(ignition::math::Pose3d(trainX - 0.1, 0, 0, 0, 0, 0));
      rightDoor->SetWorldPose(ignition::math::Pose3d(trainX + 0.1, 0, 0, 0, 0, 0));
    }

  private:
    physics::WorldPtr world;
    physics::ModelPtr train, leftDoor, rightDoor;
    event::ConnectionPtr updateConnection;
    transport::NodePtr node;
    common::Time startTime, doorStart, waitStart, lastUpdate;
    double initialVelocity = 16.12;
    double acceleration = -1.3;
    TrainState currentState = IDLE;
  };
  GZ_REGISTER_WORLD_PLUGIN(TrainControllerPlugin)
}