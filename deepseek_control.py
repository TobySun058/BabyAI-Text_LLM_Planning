import gym
import babyai_text
import matplotlib.pyplot as plt
from ollama import chat as ollama_chat
from ollama import ChatResponse

# Show the Game Board
def render_board(env):
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.5)
    plt.clf()

# Deepseek Prompt
def generate_deepseek_prompt(mission, descriptions):
    base_prompt = '''You are playing a grid-based board game on a 6x6 square grid. Various objects are placed on the board, 
and your goal is to complete a specific mission by reaching a target object.

I will provide you with:
- The mission objective
- A description of your surroundings
- The available moves you can make

Valid actions are: left, right, forward, pickup, drop, toggle, done

After each move you choose, I will update you with the new state of the environment. Continue selecting moves until you complete the mission.

To complete the mission, you must move to a position directly next to the target object and face it. Avoid walking into walls. If you suspect 
there's a wall ahead, consider turning first. If you don't have any information in BabyAI-text sees information, you need to turn and find.

Here are few examples:

When you see this: 
Mission: go to the grey box
BabyAI-Text sees:
- You see a wall 1 step forward
- You see a wall 3 steps left
- You see a blue ball 2 steps right
- You see a grey box 3 steps right
You are facing the wall, so you need to turn right to find grey box. When "You see a wall 1 step forward", you have to turn. If you can turn to 
the object direction, that would be better.

When you see this:
Mission: pick up a grey box
BabyAI-Text sees:
- You see a wall 1 step right
- You see a red key 3 steps left and 1 step forward
- You see a grey box 2 steps left and 4 steps forward
- You see a yellow box 2 steps left and 2 steps forward
- You see a blue box 2 steps left
- You see a grey box 4 steps forward
You can go forward because the grey box is 4 steps forward.

Now, give your answer based on the mission and descriptions I give you in the following:

Mission: {mission}
BabyAI-Text sees:
{descriptions}

Only respond with one word: either "left", "right", "forward", "pickup", "drop", "toggle", or "done".
Do not explain your reasoning. Just return the selected move.'''

    formatted_descriptions = "\n".join(f"- {desc}" for desc in descriptions)
    return base_prompt.format(mission=mission, descriptions=formatted_descriptions)


# Get DeepSeek Response 
def get_deepseek_response(prompt):
    response: ChatResponse = ollama_chat(
        model='deepseek-v2',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content'].strip().lower()


# Setup BabyAI-Text Environment
env = gym.make("BabyAI-MixedTrainLocal-v0")
obs, info = env.reset()

# Save original descriptions for context
original_descriptions = info.get("descriptions", ["No description available."])

action_mapping = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}

print("Initial Mission:", obs.get("mission", "Unknown Mission")) # Print the mission
render_board(env) # Show the game board


# Main Loop
while True:
    mission = obs.get("mission", "Unknown Mission")
    current_descriptions = info.get("descriptions", ["No description available."])

    print("DeepSeek Instructions:")
    for desc in original_descriptions:
        print(f"- {desc}")

    # Generate and show prompt
    deepseek_prompt = generate_deepseek_prompt(mission, current_descriptions)
    print("\n DeepSeek Prompt Sent to Model \n")
    print(deepseek_prompt)

    # Ask DeepSeek
    action_str = get_deepseek_response(deepseek_prompt)
    print(f"\n DeepSeek model chose action: {action_str}")

    # Results of the game
    if action_str in action_mapping:
        action = action_mapping[action_str]
        obs, reward, done, info = env.step(action)

        render_board(env)

        if reward > 0:
            print("Reward received:", reward)

        if done:
            print("Mission Accomplished!")
            break
    else:
        print("Invalid action from model:", action_str)
        break

env.close()
