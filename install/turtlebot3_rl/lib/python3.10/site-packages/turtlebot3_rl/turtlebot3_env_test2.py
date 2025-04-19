import gym
from gym import spaces
import numpy as np
import random
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64


class TrainEnv(gym.Env):
    def __init__(self):
        super(TrainEnv, self).__init__()

        # Initialize ROS
        rclpy.init()

        # Node to send velocity commands
        self.node = Node('turtlebot3_env_node')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.train_vel_pub = self.node.create_publisher(Twist, '/train/x_vel', 10)
        self.train_teleport_pub = self.node.create_publisher(Twist, '/train/teleport_x', 10)

        # Action space: forward, left, right
        self.action_space = spaces.Discrete(3)

        # Observation space: (x, y, distance to goal)
        self.observation_space = spaces.Box(low=np.array([-100, -100, 0]), high=np.array([100, 100, 100]), dtype=np.float32)

        self.reset()

    def reset(self):
        teleport_msg = Float64()
        teleport_msg.data = 1000.0
        self.train_teleport_pub.publish(teleport_msg)

        # Set robot to a random position between (-100, -100) and (100, 100)
        self.x = random.uniform(-100, 100)
        self.y = random.uniform(-100, 100)
        self.angle = random.uniform(0, 2 * np.pi)  # random orientation between 0 and 2π

        # Distance to the goal (0, 0)
        self.distance_to_goal = math.sqrt(self.x ** 2 + self.y ** 2)

        # Return the initial observation
        return np.array([self.x, self.y, self.distance_to_goal], dtype=np.float32)

    def train_step(self):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.0
        self.train_vel_pub.publish(msg)

    def step(self, action):
        self.train_step()
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
        # Move in the direction of the current angle
        self.x += msg.linear.x * np.cos(self.angle)
        self.y += msg.linear.x * np.sin(self.angle)
        self.angle += msg.angular.z  # Update the robot's orientation

        # Ensure the angle is between 0 and 2π
        self.angle = self.angle % (2 * np.pi)

        # Calculate the new distance to the goal
        self.distance_to_goal = math.sqrt(self.x ** 2 + self.y ** 2)

        # Reward is the negative distance to the goal (closer to 0,0 is better)
        reward = -self.distance_to_goal

        # If we are close to the goal, consider it done
        done = False
        if self.distance_to_goal < 1.0:  # Reached the goal if within 1 meter
            done = True
            reward = 100  # Give a large reward for reaching the goal

        # Return the new observation, reward, and done flag
        observation = np.array([self.x, self.y, self.distance_to_goal], dtype=np.float32)
        print(observation)
        return observation, reward, done, {}

    def render(self, mode='human'):
        # Optionally, you can render the environment here (visualize it)
        # For simplicity, you can just print the current state
        print(f"Current position: ({self.x}, {self.y}), Distance to goal: {self.distance_to_goal}")

    def close(self):
        # Shut down the ROS node when done
        self.node.destroy_node()
        rclpy.shutdown()

