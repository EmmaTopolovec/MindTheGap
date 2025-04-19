import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist
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
        self.left_doors_name = 'left_doors'
        self.right_doors_name = 'right_doors'

        self.initial_offset_m = 235.0  # meters
        self.train_speed = 24.72  # m/s (89km/h)
        self.deceleration = 1.3  # m/s^2
        self.update_rate = 20  # Hz

        self.door_slide_distance = 0.75  # meters
        self.door_slide_duration = 4.0  # seconds

        # Start simulation
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

            self.set_model_state(self.train_name, x_pos, 0, 0, velocity)
            self.set_model_state(self.left_doors_name, x_pos, 0, 0, velocity)
            self.set_model_state(self.right_doors_name, x_pos, 0, 0, velocity)
            time.sleep(dt)

        self.get_logger().info('Train stopped at platform.')

    def animate_doors(self, open=True):
        direction = 1 if open else -1
        step_count = int(self.door_slide_duration * self.update_rate)
        dt = 1.0 / self.update_rate

        for step in range(step_count + 1):
            offset = direction * (step / step_count) * self.door_slide_distance
            self.set_model_state(self.left_doors_name, 0.0 - offset, 0, 0)
            self.set_model_state(self.right_doors_name, 0.0 + offset, 0, 0)
            time.sleep(dt)

        self.get_logger().info(f'Doors {"opened" if open else "closed"}.')

    def run_sequence(self):
        # Step 1: Move train and doors to platform
        self.move_train_to_platform()

        # Step 2: Wait 3 seconds
        self.get_logger().info('Waiting 3 seconds before opening doors...')
        time.sleep(3)

        # Step 3: Open doors
        self.animate_doors(open=True)

        # Step 4: Wait 10 seconds
        self.get_logger().info('Doors open. Waiting 10 seconds...')
        time.sleep(10)

        # Step 5: Close doors
        self.animate_doors(open=False)

        self.get_logger().info('Sequence complete.')

def main(args=None):
    rclpy.init(args=args)
    node = TrainControlNode()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
