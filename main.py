from stable_baselines3 import PPO
from environment import UltrakillEnv
import os
import time

def train():
    # Initialize environment
    env = UltrakillEnv()

    # Define the model (Proximal Policy Optimization)
    # CnnPolicy is used because our observations are images
    model = PPO("CnnPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

    print("Starting training in 5 seconds... Switch to Ultrakill!")
    time.sleep(5)

    # Training loop
    try:
        # We'll train in chunks and save
        for i in range(1, 100):
            model.learn(total_timesteps=10000, reset_num_timesteps=False)
            model.save(f"ultrakill_ppo_model_{i}")
            print(f"Model saved: ultrakill_ppo_model_{i}")
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving current model...")
        model.save("ultrakill_ppo_model_interrupted")

if __name__ == "__main__":
    train()
