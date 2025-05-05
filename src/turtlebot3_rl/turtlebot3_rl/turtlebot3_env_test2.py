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
from rclpy.executors import MultiThreadedExecutor
import random
import math
import time
import os
import threading
from rosgraph_msgs.msg import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.callback_groups import ReentrantCallbackGroup

# Convert x, y, z angle to quaternion
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
    
# Convert quaternion to x, y, and z angles and extract z angle (yaw)
def quaternion_to_yaw(x, y, z, w):
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

# OpenAI Gym Environment
class TrainEnv(gym.Env):
    def __init__(self):
        super(TrainEnv, self).__init__()

        # Clean files that sometimes break when interrupting the program
        # os.system("sudo rm -rf /dev/shm/fastrtps_* /dev/shm/fastdds*")

        # Init ROS
        rclpy.init()
        self.node = Node('train_env_node')

        my_callback_group = ReentrantCallbackGroup()

        self.sim_time = 0.0

        self.last_reset_time = 0

        # Default values
        self.x = 0
        self.y = 0
        self.z = 0
        self.angle = 0

        self.executor = MultiThreadedExecutor()
        self.executor.add_node(self.node)
        self.executor_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.executor_thread.start()

        # Subscribe to get sim time
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.node.create_subscription(
            Clock,
            '/clock',
            self._clock_callback,
            qos_profile,
            callback_group=my_callback_group
        )

        self.lock = threading.Lock()


        # Subscribe to hear when bot collides with train
        self.train_collision = False
        self.node.create_subscription(
            Bool,
            '/train/collision',
            self._collision_callback,
            10,
            callback_group=my_callback_group
        )

        # Send velocity commands publisher
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Reset train publisher
        self.reset_pub = self.node.create_publisher(Empty, '/train/reset', 10)
        
        # Start train publisher
        self.start_train_pub = self.node.create_publisher(Empty, '/train/start', 10)

        # Set turtlebot3 pose publisher
        self.bot_pose_pub =  self.node.create_publisher(Pose, '/set_bot_position', 10)

        # Subscribe to hear when train leaves (reset sim)
        self.train_leaving = False
        self.node.create_subscription(
            Bool, 
            '/train/leaving', 
            self._train_leaving_cb, 
            10,
            callback_group=my_callback_group
        )

        # Subscribe to hear odometry for bot position
        self.node.create_subscription(
            Odometry, 
            '/odom', 
            self._odom_cb, 
            10, 
            callback_group=my_callback_group
            )

        # Subscribe to hear LiDAR data
        self.laser_data = np.zeros(24, dtype=np.float32)
        self.node.create_subscription(
            LaserScan,
            '/scan',
            self._laser_cb,
            10,
            callback_group=my_callback_group
        )

        # Observation Space
        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([-10, -3.0, 0, -1.0, -1.0]),
                np.zeros(24)
            )),
            high=np.concatenate((
                np.array([100, 3.0, 2.0, 1.0, 1.0]), # pose
                np.full(24, 3.5)
            )),
            dtype=np.float32
        )

        # Action space (left, right, forward, stop)
        # self.action_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)

    def _clock_callback(self, msg):
        # Store simulation time (from /clock)
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        # self.node.get_logger().info(f"[Clock]: sim_time={self.sim_time}")

    def get_sim_time(self):
        with self.lock:
            return self.sim_time

    def _collision_callback(self, msg):
        with self.lock:
            self.train_collision = self.train_collision or msg.data
            # print("COLLISION RECEIVED IN ENV")

    def _train_leaving_cb(self, msg):
        with self.lock:
            self.train_leaving = msg.data

    def _odom_cb(self, msg):
        with self.lock:
            self.x = msg.pose.pose.position.x
            self.y = msg.pose.pose.position.y
            self.z = msg.pose.pose.position.z

            self.angle = quaternion_to_yaw(
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            )

            # print(self.angle)

    def _laser_cb(self, msg):
     with self.lock:
        self.laser_data = np.array(msg.ranges, dtype=np.float32)
        self.laser_data[np.isinf(self.laser_data)] = 3.5

    def get_observation(self):
        with self.lock:
            pose_obs = np.array([self.x, self.y, self.z, math.sin(self.angle), math.cos(self.angle)], dtype=np.float32)
            if self.laser_data is not None:
                lidar_obs = np.clip(self.laser_data, 0, 10)  # clip distances for safety
                combined = np.concatenate((pose_obs, lidar_obs))
            else:
                combined = pose_obs

            return combined

    def reset(self):
        self.last_reset_time = time.time()

        print("Resetting...")

        os.system("ros2 service call /pause_physics std_srvs/srv/Empty '{}' > /dev/null 2>&1")
        time.sleep(0.5)

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)
        time.sleep(0.5)

        os.system("ros2 service call /reset_simulation std_srvs/srv/Empty '{}' > /dev/null 2>&1")
        time.sleep(0.5)

        reset_msg = Empty()
        self.reset_pub.publish(reset_msg)
        self.node.get_logger().info("Published reset message to /train/reset")

        # Set a random initial position on the platform's
        # self.x = random.uniform(10, 81.75)
        # self.y = random.uniform(-2.0, -0.5)
        # self.z = 1.151
        # self.angle = random.uniform(0, 2 * np.pi)

        # Hard-coded starting position
        # self.x = 21.5
        # self.y = -0.5
        # self.z = 1.151
        # self.angle = np.pi/2

        self.x = 10.0
        self.y = -2.0
        self.z = 1.151
        self.angle = np.pi/2

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
        # self.node.get_logger().info(f"Published new bot position {pose} /set_bot_position")
        
        self.distance_to_goal = abs(self.y - 2)
        self.prev_distance_to_goal = abs(self.y - 2)

        self.train_leaving = False
        self.train_collision = False

        self.ignore_collision_until = time.time() + 1.0

        time.sleep(0.1)

        os.system("ros2 service call /unpause_physics std_srvs/srv/Empty '{}' > /dev/null 2>&1")

        time.sleep(0.1)

        start_train_msg = Empty()
        self.start_train_pub.publish(start_train_msg)
        self.node.get_logger().info("Published start message to /train/start")
        
        return self.get_observation()

    def step(self, action):
        # rclpy.spin_once(self.node, timeout_sec=0.001)

        now = time.time()
        if now - self.last_reset_time < 2.0:
            # Prevent duplicate resets
            return self.get_observation(), 0, True, {}

        msg = Twist()
        
        # if action == 0: # Move forward
        #     msg.linear.x = 0.9
        #     msg.angular.z = 0.0
        # elif action == 1: # Turn left
        #     msg.linear.x = 0.0
        #     msg.angular.z = 0.5
        # elif action == 2: # Turn right
        #     msg.linear.x = 0.0
        #     msg.angular.z = -0.5
        # elif action == 3: # Stop
        #     msg.linear.x = 0.0
        #     msg.angular.z = 0.0

        if action == 0: # Move forward
            msg.linear.x = 0.9
            msg.angular.z = 0.0
        elif action == 1: # Stop
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        # Publish the velocity command
        self.cmd_vel_pub.publish(msg)

        # rclpy.spin_once(self.node, timeout_sec=0.1)

        # Compute distance to goal
        distance_to_goal = abs(self.y - 2)
        reward = 5
        done = False

        reward += (self.prev_distance_to_goal - distance_to_goal) * 5.0

        # Lower reward if close to hitting something
        if np.min(self.laser_data) < 0.3 and np.sum(self.laser_data) > 0:
            reward -= (0.3 - np.min(self.laser_data)) * 10
            print("[RL]: CLOSE TO WALL")

        # Higher reward for facing goal
        # facing_goal_reward = -math.cos((self.angle + math.pi/2 + math.pi) % (2 * math.pi) - math.pi)
        # reward += facing_goal_reward
        # print(f"Facing Reward: {facing_goal_reward}")

        self.prev_distance_to_goal = distance_to_goal

        # Check success/failure conditions
        if self.z < 1:
            reward += -10000 - distance_to_goal
            done = True
            print("[RL]: Bot Fell! (Z<1)")
        elif 1 <= self.y <= 3:
            reward += 100000
            done = True
            print("[RL]: TRAIN BOARDED!\n[RL]: TRAIN BOARDED!\n[RL]: TRAIN BOARDED!\n[RL]: TRAIN BOARDED!\n[RL]: TRAIN BOARDED!")
        elif self.check_collision():
            reward += -10000 - distance_to_goal
            done = True
            print("[RL]: COLLISION DETECTED!")
        elif self.train_leaving:
            if time.time() < self.ignore_collision_until:
                self.train_leaving = False
            reward += -1000 - distance_to_goal
            done = True
            print("[RL]: FAILED TO BOARD TRAIN BEFORE DOORS CLOSED!")
        
        # print(f"[RL_reward]: {reward}")

        observation = self.get_observation()
        return observation, reward, done, {}


    def render(self):
        print(f"({self.x:.2f}, {self.y:.2f}, {self.z:.2f})")

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def check_collision(self):
        with self.lock:
            # Prevent weird behavior upon reset
            if time.time() < self.ignore_collision_until:
                self.train_collision = False
                return False

            # Bot collides with something (Poorly named from previous implementation)
            if self.train_collision:
                print("[RL]: Collision flag from /train/collision")
                self.train_collision = False
                return True

            # Collision with platform wall
            if self.y < -2.6:
                print(f"[RL]: Collision with platform wall ({self.y:.4f} < -2.5)")
                return True

        return False

