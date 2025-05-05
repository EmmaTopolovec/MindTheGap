from stable_baselines3 import PPO
from turtlebot3_env_test2 import TrainEnv

# Load the trained model
# model_time = "2025-05-05 17:03:38" 
model_time = "2025-05-05 17:31:51" 
model = PPO.load(f"model_{model_time}")

env = TrainEnv()

num_runs = 10

successes = 0
failures = 0

for run in range(num_runs):
    print(f"Running Test {run+1}/{num_runs}...")

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if done:
            if 'TRAIN BOARDED!' in info.get('message', ''):
                print(f"Success! Trial {run+1} completed.")
                successes += 1
            else:
                print(f"Failure! Trial {run+1} completed.")
                failures += 1

print(f"\nTesting complete. Successes: {successes}, Failures: {failures}")
