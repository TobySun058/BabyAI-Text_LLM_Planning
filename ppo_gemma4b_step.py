# PPO train gemma-4b by getting reward after each episode
import gym
import babyai_text
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from torch.optim import Adam

# Config
model_id = "google/gemma-3-4b-it"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = 0.99  # Discount factor for returns

# Load model and processor
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_id)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last 3 transformer layers
for i in range(-3, 0):
    for param in model.language_model.model.layers[i].parameters():
        param.requires_grad = True

# Optimizer
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = Adam(trainable_params, lr=1e-5)

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
valid_token_ids = [processor.tokenizer.encode(a, add_special_tokens=False)[0] for a in valid_actions]

# PPO loss function
def ppo_loss_step(new_log_probs, old_log_probs, values, returns, clip_eps=0.2, vf_coef=0.5):
    values = values.squeeze()
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    ratios = torch.exp(new_log_probs - old_log_probs.detach())
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + vf_coef * value_loss
    return loss, policy_loss, value_loss

# Prompt generator
def generate_prompt(mission, descriptions, history):
    history_str = ""
    for i, step in enumerate(history[-5:]):
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

You must return exactly one valid action.

Use these rules:
1. **forward**: move 1 step forward.
2. **pickup**: pickup the project in front of you.
3. **drop**: drop the object.
4. **left**: turn left but stay in the same grid
5. **right**: turn right but stay in the same grid
6. **left** or **right** if there's a wall ahead or need to explore.
7. Use **done** to end when complete.
8. Use **right** or **left** then move **forward**, you can move to different direction
8. Using these actions and environment descriptions to accomplish the mission.

Valid actions:
- left
- right
- forward
- pickup
- drop
- toggle
- done

===========================
Mission Objective:
{mission}

Current Observations:
{chr(10).join(f"- {desc}" for desc in descriptions)}

Recent History:
{history_str if history_str else "None yet."}
===========================

Respond with ONE action word.
'''

# Run one episode
def run_episode(env, model, processor, max_steps=10):
    obs, info = env.reset()
    mission = obs.get("mission", "Unknown Mission")
    history, values, prompts, actions, old_log_probs, rewards = [], [], [], [], [], []

    for step in range(max_steps):
        descriptions = info.get("descriptions", [])
        prompt = generate_prompt(mission, descriptions, history)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that replies with only a valid BabyAI move."}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.logits[:, -1, :]
            hidden = outputs.hidden_states[-1][:, -1, :]

        value = hidden.mean(dim=-1)
        values.append(value)

        mask = torch.full_like(logits, -float("inf"))
        for token_id in valid_token_ids:
            mask[:, token_id] = logits[:, token_id]
        probs = torch.softmax(mask, dim=-1)

        sampled_token = torch.multinomial(probs, num_samples=1)
        action_str = processor.tokenizer.decode(sampled_token[0], skip_special_tokens=True).strip().lower()
        if action_str not in action_mapping:
            break

        log_prob = torch.log(probs[0, sampled_token[0]])

        action = action_mapping[action_str]
        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        old_log_probs.append(log_prob.detach())
        prompts.append(messages)
        actions.append(sampled_token[0].item())
        history.append({"descriptions": descriptions, "action": action_str})

        if done:
            break

    # Compute discounted returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns, device=device).float()
    values = torch.stack(values)
    old_log_probs = torch.stack(old_log_probs)

    return prompts, actions, old_log_probs, values, returns

# Training loop
env = gym.make("BabyAI-MixedTrainLocal-v0")
num_episodes = 100

for episode in range(num_episodes):
    print(f"\n=========== Episode {episode} ===========")
    prompts, actions, old_log_probs, values, returns = run_episode(env, model, processor)

    if len(old_log_probs) == 0:
        print("Skipping PPO update (no valid steps).")
        continue

    new_log_probs = []
    for msg, act in zip(prompts, actions):
        inputs = processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        new_log_prob = torch.log(probs[0, act])
        new_log_probs.append(new_log_prob)

    new_log_probs = torch.stack(new_log_probs)

    loss, policy_loss, value_loss = ppo_loss_step(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        values=values,
        returns=returns
    )

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    print(f"[Episode {episode}] Total Reward: {returns.sum().item():.4f} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f} | Total Loss: {loss.item():.4f}")

# Save
model.save_pretrained("./ppo-gemma-3-4b-step-100episode")
processor.save_pretrained("./ppo-gemma-3-4b-step-100episode")
print("Model and tokenizer saved.")
