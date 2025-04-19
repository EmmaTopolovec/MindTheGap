import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.clock import ClockType
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist
from std_srvs.srv import Empty
import time
import math

class TrainControlNode(Node):
    def __init__(self):
        super().__init__('train_control_node')

        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /gazebo/set_entity_state service...')

        # Initial values
        self.train_name = 'train'
        self.left_doors_name = 'left'
        self.right_doors_name = 'right'

        self.initial_offset_m = 235.0  # meters
        self.train_speed = 24.72  # m/s (89km/h)
        self.deceleration = 1.3  # m/s^2
        self.update_rate = 20  # Hz

        self.door_slide_distance = 0.75  # meters
        self.door_slide_duration = 4.0  # seconds

        # Simulation clock
        self.sim_clock = self.get_clock()

        # Reset
        self.reset_srv = self.create_service(Empty, 'reset_train', self.reset_callback)

        # Service to run train
        self.start_train_srv = self.create_service(Empty, 'start_train', self.start_train_callback)

        # Publisher for train state
        self.train_state_pub = self.create_publisher(EntityState, '/train/state', QoSProfile(depth=10))

        # Start simulation - possibly remove and just use service?
        self.run_sequence()

    def set_model_state(self, name, x, y, z, x_vel=0.0):
        req = SetEntityState.Request()
        state = EntityState()
        state.name = name
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0
        state.twist.linear.x = x_vel
        req.state = state

        if name == self.train_name:
            self.train_state_pub.publish(state) 

        future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        return future.result()

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

def main(args=None):
    rclpy.init(args=args)
    node = TrainControlNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
