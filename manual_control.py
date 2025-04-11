import gym
import babyai_text  # Make sure this package is installed
import matplotlib.pyplot as plt
import numpy as np

# Create the BabyAI-Text environment
env = gym.make("BabyAI-MixedTrainLocal-v0")

# Reset the environment
obs, info = env.reset()

# Define action mapping for manual control
action_mapping = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}

print("\nBabyAI-Text Manual Control Started")
print("Control the agent using:", ", ".join(action_mapping.keys()), "\n")


# Show the game board
def render_board(env):
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()  # Clear plot for next render

# Initial render
render_board(env)

# Main Loop
while True:
    print("\nMission:", obs.get("mission", "Unknown Mission"))

    descriptions = info.get("descriptions", ["No description available."])
    print("BabyAI-Text sees:") # Environment
    for desc in descriptions:
        print(f"- {desc}")

    user_input = input("Enter action: ").strip().lower() # Manual Control Action
    # Results
    if user_input in action_mapping:
        action = action_mapping[user_input]
        obs, reward, done, info = env.step(action)

        render_board(env)

        if reward > 0:
            print("Reward received:", reward)

        if done:
            print("Mission Accomplished")
            break
    else:
        print("Invalid command! Valid actions are:", ", ".join(action_mapping.keys()))

env.close()
