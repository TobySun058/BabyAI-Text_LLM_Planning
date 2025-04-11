# PPO training gemma-3-1b-it by getting reward for each episode
import gym
import babyai_text
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.optim import Adam
import random

# Configs
model_id = "google/gemma-3-1b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model with hidden states and quantization
bnb_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    output_hidden_states=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()

# Freeze model parameters
for param in model.parameters():
    param.requires_grad = False

# Value Head
class ValueHead(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 1)
        )

    def forward(self, hidden):
        return self.head(hidden).squeeze(-1)

value_head = ValueHead(hidden_size=1152).to(device)
optimizer = Adam(list(value_head.parameters()), lr=1e-4)

# Action mapping
action_mapping = {
    "left": 0,
    "right": 1,
    "forward": 2,
    "pickup": 3,
    "drop": 4,
    "toggle": 5,
    "done": 6
}
valid_actions = list(action_mapping.keys())
valid_token_ids = [tokenizer.encode(a, add_special_tokens=False)[0] for a in valid_actions]

# Prompt generator
def generate_prompt(mission, descriptions, history):
    history_str = ""
    for i, step in enumerate(history):
        history_str += f"Step {i+1}:\n"
        history_str += "BabyAI-Text sees:\n"
        history_str += "\n".join(f"- {d}" for d in step["descriptions"]) + "\n"
        history_str += f"Gemma chose action: {step['action']}\n\n"

    return f'''
You are playing a 6x6 grid-based board game. Objects such as keys, balls, doors, and boxes are placed on the board. You are an agent that must complete a mission by navigating toward a target object using simple movement actions.

I will give you:
- A mission objective
- A textual description of what your agent currently sees
- A history of what the agent has previously seen and done

You must return exactly one valid action to take next.

You must choose the next move based on your current surroundings. Use the following rules to guide your decision:
1. If the target object is directly in front of you, move **forward** to get closer.
2. If you are holding the object and need to drop it, use **drop**.
3. If the object is in front of you and can be picked up, use **pickup**.
4. If there’s a wall directly ahead, use **left** or **right** to turn before moving.
5. If the target object is not in your field of view, **turn left or right** to look for it.
6. Once you believe the mission is complete, use **done**.

Valid actions (respond with ONE word only):
- **left**: turn left without moving
- **right**: turn right without moving
- **forward**: move forward by one step
- **pickup**: pick up the object in front of you
- **drop**: drop the object you're holding
- **toggle**: open or close a door, or activate a switch
- **done**: signal that the mission is complete

===========================
Mission Objective:
{mission}

Current Observations from BabyAI-Text:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Recent History:
{history_str if history_str else "None yet."}
===========================

Respond with ONLY ONE action word.
'''

# Run one episode
def run_episode(env, model, tokenizer, value_head, max_steps=20):
    obs, info = env.reset()
    mission = obs.get("mission", "Unknown Mission")
    history, log_probs, state_values = [], [], []
    final_reward = torch.tensor([0.0], device=device)

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        descriptions = info.get("descriptions", [])
        prompt = generate_prompt(mission, descriptions, history)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that replies with only a valid BabyAI move."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        tokenized = tokenizer.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in tokenized.items()}

        # Forward pass
        outputs = model(**inputs)
        hidden = outputs.hidden_states[-1][:, -1, :]  # last token hidden state
        logits = outputs.logits[:, -1, :]              # logits for next token
        logits = torch.clamp(logits, -50, 50)

        # Mask to only valid actions
        mask = torch.full_like(logits, -float("inf"))
        for token_id in valid_token_ids:
            mask[:, token_id] = logits[:, token_id]
        probs = torch.softmax(mask, dim=-1)

        sampled_token = torch.multinomial(probs, num_samples=1)
        action_str = tokenizer.decode(sampled_token[0], skip_special_tokens=True).strip().lower()
        print(f"Model selected action: {action_str}")

        if action_str not in action_mapping:
            print("Invalid action. Ending episode.")
            break

        log_prob = torch.log(probs[0, sampled_token[0]])
        value = value_head(hidden.float())

        log_probs.append(log_prob)
        state_values.append(value)
        history.append({"descriptions": descriptions, "action": action_str})

        action = action_mapping[action_str]
        obs, reward, done, info = env.step(action)
        final_reward = torch.tensor([reward], device=device)

        if done:
            print("Episode completed.")
            break

    return log_probs, state_values, final_reward

# Training loop
env = gym.make("BabyAI-MixedTrainLocal-v0")
num_episodes = 100

for episode in range(num_episodes):
    print(f"\n=========== Episode {episode} ===========")
    model.eval()
    log_probs, state_values, reward = run_episode(env, model, tokenizer, value_head)

    if not log_probs or len(state_values) == 0:
        print("Skipping PPO update (no valid steps).")
        continue

    log_probs = torch.stack(log_probs).float()
    values = torch.cat(state_values).squeeze().float()
    returns = reward.expand_as(values).detach().float()
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratios = torch.exp(log_probs - log_probs.detach())
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.empty_cache()

    print(f"\n[Episode {episode}] Reward: {reward.item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | Total Loss: {loss.item():.4f}")

# Save
model.save_pretrained("./ppo-gemma-3-1b-it-100-episode")
tokenizer.save_pretrained("./ppo-gemma-3-1b-it-100-episode")
torch.save(value_head.state_dict(), "./ppo-value-head.pt")
print("\nModel and value head saved.") 