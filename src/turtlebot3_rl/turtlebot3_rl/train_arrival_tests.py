import time
from turtlebot3_env_test2 import TrainEnv

num_tests = 10

def run_tests():
    env = TrainEnv()

    successes = 0

    for test in range(num_tests):
        print(f"Running test {test + 1}/{num_tests}...")

        # Reset the environment
        env.reset()

        # Wait until the train starts leaving
        while not env.train_leaving:
            time.sleep(0.1)

        print(f"Test {test + 1}: SUCCESS.")
        successes += 1

        time.sleep(1)

    print(f"Testing complete. {successes} successes, 0 failures.")

if __name__ == "__main__":
    run_tests()
