import gym
from gym import spaces
import numpy as np
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState


class TrainEnv(gym.Env):
    def __init__(self):
        super(TrainEnv, self).__init__()

        # Initialize ROS
        rclpy.init()

        # Node to send velocity commands
        self.node = Node('turtlebot3_env_node')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.set_entity_state_cli = self.node.create_client(SetEntityState, '/gazebo/set_entity_state')

        while not self.set_entity_state_cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for /gazebo/set_entity_state service...')

        # Action space: forward, left, right
        self.action_space = spaces.Discrete(3)

        # Observation space: (x, y, distance to goal)
        self.observation_space = spaces.Box(low=np.array([-100, -100, 0]), high=np.array([100, 100, 100]), dtype=np.float32)

        # Goal position on the platform (define where you want the robot to go)
        self.goal_x = 10.0  # Example position along x-axis
        self.goal_y = 0.0   # Example position along y-axis
        self.goal_distance = math.sqrt(self.goal_x ** 2 + self.goal_y ** 2)

        # Initialize robot position on the train platform
        self.reset()

    def reset(self):
        # Reset the robot's initial position (start on the train platform)
        self.x = 0.0  # You can adjust this to your specific train platform's position
        self.y = 0.0
        self.angle = 0.0  # Assume the robot starts facing the "train" direction

        # Set the robot's position in Gazebo
        self.set_model_state('turtlebot3_burger', self.x, self.y, 0.0)

        # Distance to the goal (goal is the point on the platform)
        self.distance_to_goal = math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)

        # Return the initial observation (robot position and distance to goal)
        return np.array([self.x, self.y, self.distance_to_goal], dtype=np.float32)

    def set_model_state(self, model_name, x, y, z):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = model_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0  # Set orientation to 0 (facing forward)
        req.state = state
        future = self.set_entity_state_cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

    def step(self, action):
        # Create a Twist message for controlling the robot
        msg = Twist()

        # Apply the action: move forward, left, or right
        if action == 0:  # Move forward
            msg.linear.x = 0.5  # Move forward with a speed of 0.5 m/s
            msg.angular.z = 0.0  # No rotation
        elif action == 1:  # Turn left
            msg.linear.x = 0.0  # No movement forward
            msg.angular.z = 0.5  # Turn left
        elif action == 2:  # Turn right
            msg.linear.x = 0.0  # No movement forward
            msg.angular.z = -0.5  # Turn right

        # Publish the velocity command to control the robot
        self.cmd_vel_pub.publish(msg)

        # Simulate the robot's movement based on action (simple kinematic model)
        self.x += msg.linear.x * np.cos(self.angle)
        self.y += msg.linear.x * np.sin(self.angle)
        self.angle += msg.angular.z  # Update the robot's orientation

        # Ensure the angle is between 0 and 2π
        self.angle = self.angle % (2 * np.pi)

        # Calculate the new distance to the goal
        self.distance_to_goal = math.sqrt((self.goal_x - self.x) ** 2 + (self.goal_y - self.y) ** 2)

        # Reward is the negative distance to the goal (closer to the goal is better)
        reward = -self.distance_to_goal

        # If the robot is close to the goal, mark the task as done
        done = False
        if self.distance_to_goal < 1.0:  # Reached the goal if within 1 meter
            done = True
            reward = 100  # Give a large reward for reaching the goal

        # Return the new observation, reward, and done flag
        observation = np.array([self.x, self.y, self.distance_to_goal], dtype=np.float32)
        return observation, reward, done, {}

    def render(self, mode='human'):
        # Optionally, you can render the environment here (visualize it)
        # For simplicity, you can just print the current state
        print(f"Current position: ({self.x}, {self.y}), Distance to goal: {self.distance_to_goal}")

    def close(self):
        # Shut down the ROS node when done
        self.node.destroy_node()
        rclpy.shutdown()
