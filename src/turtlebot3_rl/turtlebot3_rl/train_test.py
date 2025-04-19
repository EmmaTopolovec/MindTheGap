import gym
from stable_baselines3 import PPO
from turtlebot3_env_test import TrainEnv
from datetime import datetime

# Create the environment
env = TrainEnv()

# Create RL model
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
model.learn(total_timesteps=100)

# Save the model
model_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model.save(f"model_{model_time}")

# Test the model
obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")

