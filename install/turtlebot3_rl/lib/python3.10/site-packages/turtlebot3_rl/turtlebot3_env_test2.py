import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from std_msgs.msg import Bool
import random
import math

def euler_to_quaternion(roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - \
            math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + \
            math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - \
            math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + \
            math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        return qx, qy, qz, qw
    
def quaternion_to_yaw(x, y, z, w):
    # This gives yaw only (simplified for 2D use cases)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

class TrainEnv(gym.Env):
    def __init__(self):
        super(TrainEnv, self).__init__()

        # Init ROS
        rclpy.init()
        self.node = Node('train_env_node')

        self.train_collision = False
        self.node.create_subscription(
            Bool,
            '/train/collision',
            self._collision_callback,
            10
        )

        # Velocity publisher
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Reset train
        self.reset_pub = self.node.create_publisher(Empty, '/train/reset', 10)

        # Set turtlebot3 pose
        self.bot_pose_pub =  self.node.create_publisher(Pose, '/set_bot_position', 10)

        # Subscribe to train arrival flag
        self.train_leaving = False
        self.node.create_subscription(Bool, '/train/leaving', self._train_leaving_cb, 10)

        # Subscribe to odometry for bot position
        self.node.create_subscription(Odometry, '/odom', self._odom_cb, 10)

        self.laser_data = np.zeros(24, dtype=np.float32)  # initialize with zeros
        self.node.create_subscription(
            LaserScan,
            '/scan',
            self._laser_cb,
            10
        )

        # Observation
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([-10, -3.0, 0, 0.0]),     # pose: x, y, z, angle
                np.zeros(24)                            # lidar min distances
            )),
            high=np.concatenate((
                np.array([100, 3.0, 2.0, 2 * np.pi]), # pose
                np.full(24, 10.0)                       # lidar max range
            )),
            dtype=np.float32
        )

        self.action_space = spaces.Discrete(3)

    def _collision_callback(self, msg):
        self.train_collision = msg.data

    def _train_leaving_cb(self, msg):
        self.train_leaving = msg.data

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        self.angle = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )

    def _laser_cb(self, msg):
     self.laser_data = np.array(msg.ranges, dtype=np.float32)

    def get_observation(self):
        pose_obs = np.array([self.x, self.y, self.z, self.angle], dtype=np.float32)
        if self.laser_data is not None:
            lidar_obs = np.clip(self.laser_data, 0, 10)  # clip distances for safety
            combined = np.concatenate((pose_obs, lidar_obs))
        else:
            combined = pose_obs  # fallback if no data yet
        return combined

    def reset(self):
        reset_msg = Empty()
        self.reset_pub.publish(reset_msg)
        self.node.get_logger().info("Published reset message to /train/reset")

        # Set a random initial position within the platform's valid range
        self.x = random.uniform(10, 81.75)  # Ensure these values respect the platform
        self.y = random.uniform(-2.5, -0.5)  # Ensure this is in the goal range of y
        self.z = 1.17  # Ensure z > 1.15 (the platform height)
        self.angle = random.uniform(0, 2 * np.pi)  # random orientation between 0 and 2π

        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = self.z

        qx, qy, qz, qw = euler_to_quaternion(0, 0, self.angle)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        self.bot_pose_pub.publish(pose)
        self.node.get_logger().info(f"Published new bot position {pose} /set_bot_position")
        
        # Distance to the goal (in y range [1, 3])
        self.distance_to_goal = self.y - 2  # If goal is between 1 and 3, goal is y = 2
        self.prev_distance_to_goal = abs(self.y - 2)

        self.train_leaving = False
        
        # Return the initial observation
        return self.get_observation()

    def step(self, action):
        # Step the agent based on the action chosen by the RL algorithm

        # self.render()

        # Apply the action: move forward, left, or right
        msg = Twist()
        
        if action == 0:  # Move forward
            msg.linear.x = 0.5
            msg.angular.z = 0.0
        elif action == 1:  # Turn left
            msg.linear.x = 0.0
            msg.angular.z = 0.5
        elif action == 2:  # Turn right
            msg.linear.x = 0.0
            msg.angular.z = -0.5

        # Publish the velocity command
        self.cmd_vel_pub.publish(msg)

        rclpy.spin_once(self.node, timeout_sec=0.1)

        # Compute distance to goal (y = 2)
        distance_to_goal = abs(self.y - 2)
        reward = -distance_to_goal
        done = False  # <- initialize done to False

        if distance_to_goal > self.prev_distance_to_goal:
            reward -= 5  # Penalty for moving away

        self.prev_distance_to_goal = distance_to_goal

        # Check failure conditions
        if self.z < 1.15:
            reward = -100
            done = True
            print("[RL]: Bot Fell! (Z<1.15)")
        elif self.check_collision():
            reward = -100
            done = True
            print("[RL]: COLLISION DETECTED!")

        # Check success condition (train boarded)
        elif 1 <= self.y <= 3:
            reward = 100
            done = True
            print("[RL]: TRAIN BOARDED!")

        elif self.train_leaving:
            reward = -100
            done = True
            print("[RL]: FAILED TO BOARD TRAIN BEFORE DOORS CLOSED!")
        
        observation = self.get_observation()
        return observation, reward, done, {}


    def render(self):
        print(f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})")

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def check_collision(self):
        # Collision from ROS topic
        if self.train_collision:
            print("[RL]: Collision flag from /train/collision topic.")
            return True

        # Collision based on Y position (bot left the valid area)
        if self.y < -2.75:
            print(f"[RL]: Collision due to Y position ({self.y:.2f} < -2.75)")
            return True

        return False

