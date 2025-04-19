import gym
from gym import spaces
import numpy as np
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from std_srvs.srv import Empty
from rclpy.qos import QoSProfile


class TrainEnv(gym.Env):
    def __init__(self):
        super(TrainEnv, self).__init__()

        # Initialize ROS
        rclpy.init()

        # Node to send velocity commands
        self.node = Node('turtlebot3_env_node')
        self.cmd_vel_pub = self.node.create_publisher(Twist, '/cmd_vel', 10)

        # Action space: forward, left, right
        self.action_space = spaces.Discrete(3)

        # Observation space: (x, y, distance to goal)
        self.observation_space = spaces.Box(low=np.array([-100, -100, 0]), high=np.array([100, 100, 100]), dtype=np.float32)

        # Goal position on the platform (define where you want the robot to go)
        self.goal_x = 10.0  # Example position along x-axis
        self.goal_y = 0.0   # Example position along y-axis
        self.goal_distance = math.sqrt(self.goal_x ** 2 + self.goal_y ** 2)

        # Train stuff

        # Initial values
        self.train_name = 'train'
        self.left_doors_name = 'left'
        self.right_doors_name = 'right'

        self.initial_offset_m = 235.0  # meters
        self.train_speed = 24.72  # m/s (89km/h)
        self.deceleration = 1.3  # m/s^2
        self.update_rate = 20  # Hz

        self.train_x = self.initial_offset_m
        self.train_velocity = self.train_speed

        self.door_slide_distance = 0.75  # meters
        self.door_slide_duration = 4.0  # seconds

        # Simulation clock
        self.sim_clock = self.node.get_clock()

        # Reset
        self.reset_srv = self.node.create_service(Empty, 'reset_train', self.reset_callback)

        # Service to run train
        self.start_train_srv = self.node.create_service(Empty, 'start_train', self.start_train_callback)

        # Publisher for train state
        self.train_state_pub = self.node.create_publisher(EntityState, '/train/state', QoSProfile(depth=10))

        # Initialize positions of train and bot
        self.reset()

    def reset(self):

        # Train stuff

        self.train_x = self.initial_offset_m
        self.train_velocity = self.train_speed
        self.set_model_state(self.train_name, self.train_x, 0.0, 0.0, self.train_velocity)
        self.set_model_state(self.left_doors_name, self.train_x, 0.0, 0.0, self.train_velocity)
        self.set_model_state(self.right_doors_name, self.train_x, 0.0, 0.0, self.train_velocity)

        # Bot stuff

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

    def set_model_state(self, model_name, x, y, z, x_vel=0.0):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = model_name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0  # Set orientation to 0 (facing forward)

        if model_name == self.train_name:
            state.twist.linear.x = x_vel
            self.train_state_pub.publish(state) 
        
        req.state = state
        future = self.node.cli.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

    def move_train_step(self):
        dt = 1.0 / self.update_rate
        if self.train_x > 0.0 and self.train_velocity > 0.0:
            self.train_x -= self.train_velocity * dt
            self.train_velocity -= self.deceleration * dt
            self.train_velocity = max(0.0, self.train_velocity)

            # Update train and door positions
            self.set_model_state(self.train_name, self.train_x, 0.0, 0.0, self.train_velocity)
            self.set_model_state(self.left_doors_name, self.train_x, 0.0, 0.0, self.train_velocity)
            self.set_model_state(self.right_doors_name, self.train_x, 0.0, 0.0, self.train_velocity)

    def step(self, action):
        self.move_train_step()

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
    
    def move_train_to_platform(self):
        self.get_logger().info('Moving train to platform...')
        x_pos = self.initial_offset_m
        velocity = self.train_speed
        dt = 1.0 / self.update_rate

        while x_pos > 0.0:
            x_pos -= velocity * dt
            velocity -= self.deceleration * dt
            velocity = max(0.0, velocity)

            self.set_model_state(self.train_name, x_pos, 0.0, 0.0, velocity)
            self.set_model_state(self.left_doors_name, x_pos, 0.0, 0.0, velocity)
            self.set_model_state(self.right_doors_name, x_pos, 0.0, 0.0, velocity)
            
            start_time = self.sim_clock.now()
            duration = rclpy.duration.Duration(seconds=dt)

            while self.sim_clock.now() - start_time < duration:
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info('Train stopped at platform.')

    def animate_doors(self, open=True):
        direction = 1 if open else -1
        step_count = int(self.door_slide_duration * self.update_rate)
        dt = 1.0 / self.update_rate

        for step in range(step_count + 1):
            offset = direction * (step / step_count) * self.door_slide_distance
            self.set_model_state(self.left_doors_name, 0.0 - offset, 0.0, 0.0)
            self.set_model_state(self.right_doors_name, 0.0 + offset, 0.0, 0.0)
            
            start_time = self.sim_clock.now()
            duration = rclpy.duration.Duration(seconds=duration)

            while self.sim_clock.now() - start_time < duration:
                rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f'Doors {"opened" if open else "closed"}.')

    def reset_callback(self, request, response):
        self.get_logger().info('Resetting train...')
        self.set_model_state(self.train_name, self.initial_offset_m, 0.0, 0.0, 0.0)
        self.set_model_state(self.left_doors_name, self.initial_offset_m, 0.0, 0.0, 0.0)
        self.set_model_state(self.right_doors_name, self.initial_offset_m, 0.0, 0.0, 0.0)
        return response

    def start_train_callback(self, request, response):
        self.get_logger().info('Starting train sequence...')
        self.run_sequence()
        return response
    
    def run_sequence(self):
        # Step 1: Move train and doors to platform
        self.move_train_to_platform()

        # Step 2: Wait 3 seconds
        self.get_logger().info('Waiting 3 seconds before opening doors...')
        
        start_time = self.sim_clock.now()
        duration = rclpy.duration.Duration(seconds=3)

        while self.sim_clock.now() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Step 3: Open doors
        self.animate_doors(open=True)

        # Step 4: Wait 10 seconds
        self.get_logger().info('Doors open. Waiting 10 seconds...')
        
        start_time = self.sim_clock.now()
        duration = rclpy.duration.Duration(seconds=10)

        while self.sim_clock.now() - start_time < duration:
            rclpy.spin_once(self, timeout_sec=0.1)

        # Step 5: Close doors
        self.animate_doors(open=False)

        self.get_logger().info('Sequence complete.')

    def render(self, mode='human'):
        # Optionally, you can render the environment here (visualize it)
        # For simplicity, you can just print the current state
        print(f"Current position: ({self.x}, {self.y}), Distance to goal: {self.distance_to_goal}")

    def close(self):
        # Shut down the ROS node when done
        self.node.destroy_node()
        rclpy.shutdown()
