import time
from turtlebot3_env_test2 import TrainEnv
from geometry_msgs.msg import Twist

num_tests = 10

def run_tests():
    env = TrainEnv()

    successes = 0
    failures = 0

    for test in range(num_tests):
        print(f"Running test {test + 1}/{num_tests}...")
        
        # Reset the environment
        obs = env.reset()
        
        while env.get_sim_time() < 15:
            time.sleep(0.1)
            # print(f"Sim Time={env.get_sim_time()}")

        msg = Twist()
        msg.linear.x = 0.9
        msg.angular.z = 0.0
        env.cmd_vel_pub.publish(msg)
        print("Moving robot forward...")
        
        while env.get_sim_time() < 17:
            # print(f"Sim Time={env.get_sim_time()}")
            time.sleep(0.01)

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        env.cmd_vel_pub.publish(msg)
        print("Stopping robot...")

        while not env.train_leaving:
            time.sleep(0.1)
        
        # Check if the test is a success or failure
        if 1 <= env.y <= 3:
            print(f"Test {test + 1}: SUCCESS - Robot boarded the train!")
            successes += 1
        elif env.check_collision():
            print(f"Test {test + 1}: FAILURE - Collision detected!")
            failures += 1
        elif env.z < 1:
            print(f"Test {test + 1}: FAILURE - Robot fell off the platform!")
            failures += 1
        else:
            print(f"Test {test + 1}: FAILURE - Train left before boarding!")
            failures += 1
        
        time.sleep(1)

    print(f"Testing complete. {successes} successes, {failures} failures.")

if __name__ == "__main__":
    run_tests()
