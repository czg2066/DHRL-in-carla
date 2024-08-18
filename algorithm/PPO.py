import torch
import torch.nn as nn
from torch.distributions import Categorical

class CustomPPO:
    def __init__(self, model, optimizer, clip_param=0.2, ppo_epochs=10, mini_batch_size=64):
        self.model = model
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size

    def compute_returns(self, next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = torch.randperm(batch_size)[:mini_batch_size]
            yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                action_probs, state_value = self.model(state)
                dist = Categorical(action_probs)
                new_log_probs = dist.log_prob(action)
                ratio = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = -torch.min(surr1, surr2).mean()
                critic_loss = (return_ - state_value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

"""
def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=3e-4)

    for epoch in range(10):  # Train for 10 epochs for simplicity
        state = env.reset()
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []
        entropy = 0

        for _ in range(200):  # Collect data from 200 timesteps
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = policy(state)

            dist = Categorical(action_probs)
            action = dist.sample()

            next_state, reward, done, _ = env.step(action.numpy()[0])
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            masks.append(torch.tensor([1-done], dtype=torch.float32))
            states.append(state)
            actions.append(action)

            state = next_state

            if done:
                break
        
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        _, next_value = policy(next_state)
        returns = compute_returns(next_value, rewards, masks)
        
        states = torch.cat(states)
        actions = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns)
        values = torch.cat(values)
        
        advantages = returns - values

        ppo_update(policy, optimizer, 4, 64, states, actions, log_probs, returns, advantages)

    env.close()

if __name__ == '__main__':
    train()
"""
