import gym
from stable_baselines3 import PPO, DQN
from turtlebot3_env_test2 import TrainEnv
from datetime import datetime

# Create the environment
env = TrainEnv()

# Create RL model
# model = PPO("MlpPolicy", env, verbose=1)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=32,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
)

# model = DQN(
#     "MlpPolicy",
#     env,
#     learning_rate=1e-3,
#     buffer_size=10000,
#     learning_starts=1000,
#     batch_size=64,
#     verbose=1
# )

# Train the model
model.learn(total_timesteps=100_000)

# Save the model
model_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
model.save(f"model_{model_time}")

# # Test the model
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     print(f"Action: {action}, Reward: {reward}")

