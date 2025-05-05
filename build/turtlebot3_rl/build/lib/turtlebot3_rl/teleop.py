import time
from turtlebot3_env_test2 import TrainEnv

# Define the number of tests
num_tests = 10

def run_tests():
    env = TrainEnv()  # Initialize the environment

    successes = 0
    failures = 0

    for test in range(num_tests):
        print(f"Running test {test + 1}/{num_tests}...")
        
        # Reset the environment at the start of each test
        obs = env.reset()
        
        # Wait until the simulation time is greater than 15 seconds
        while env.get_sim_time() < 15:
            time.sleep(0.1)

        # Set forward velocity of the robot to 0.9 for 1 second
        action_forward = 0  # Action 0 corresponds to moving forward
        obs, reward, done, info = env.step(action_forward)
        time.sleep(1.0)  # Wait for 1 second

        # Set velocity to zero (stop moving)
        action_stop = 3  # Action 3 corresponds to stop
        obs, reward, done, info = env.step(action_stop)
        
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
        elif env.train_leaving:
            print(f"Test {test + 1}: FAILURE - Train left before boarding!")
            failures += 1
        
        # Wait before starting the next test
        time.sleep(1)

    print(f"Testing complete. {successes} successes, {failures} failures.")

if __name__ == "__main__":
    run_tests()
