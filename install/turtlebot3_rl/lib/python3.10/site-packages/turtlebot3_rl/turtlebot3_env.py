import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

class TurtleBot3Env(gym.Env):
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.node = rclpy.create_node('rl_env')

        self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.node.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.scan = np.ones(360)

        self.reset_client = self.node.create_client(Empty, '/reset_simulation')
        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info('Waiting for /reset_simulation service...')

        self.action_space = spaces.Discrete(3)  # forward, left, right
        self.observation_space = spaces.Box(low=0.0, high=3.5, shape=(24,), dtype=np.float32)

    def lidar_callback(self, msg):
        self.scan = np.array(msg.ranges)

    def reset(self):
        req = Empty.Request()
        future = self.reset_client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)
        return self.get_state()

    def get_state(self):
        rclpy.spin_once(self.node, timeout_sec=0.1)
        scan = np.clip(self.scan, 0.0, 3.5)
        return scan[::15]  # downsample

    def step(self, action):
        msg = Twist()
        if action == 0:
            msg.linear.x = 0.2
        elif action == 1:
            msg.angular.z = 0.5
        elif action == 2:
            msg.angular.z = -0.5
        self.publisher.publish(msg)

        rclpy.spin_once(self.node, timeout_sec=0.1)
        obs = self.get_state()

        done = np.min(obs) < 0.2
        reward = -1.0 if done else 0.1
        return obs, reward, done, {}

    def close(self):
        self.node.destroy_node()
        rclpy.shutdown()

