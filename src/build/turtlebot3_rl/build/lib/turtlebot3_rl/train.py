from turtlebot3_env import TurtleBot3Env
from stable_baselines3 import PPO

if __name__ == "__main__":
    env = TurtleBot3Env()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    env.close()

