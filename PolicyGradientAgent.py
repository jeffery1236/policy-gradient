import numpy as np
import torch as T
from policy import PolicyNet
from replay_memory import ReplayMemory

class PolicyGradientAgent():
    def __init__(self, lr, gamma, state_dims, num_actions, env_name,
                 checkpoint_dir):
        
        self.lr = lr
        self.gamma = gamma
        self.state_dims = state_dims
        self.num_actions = num_actions

        self.policy = PolicyNet(lr=lr, num_actions=num_actions,
                                input_dims=state_dims, fc1_dims=256, fc2_dims=256,
                                 name=env_name+'_policy_gradients',
                                 checkpoint_dir=checkpoint_dir)
        # self.action_logprob_history = T.autograd.Variable(T.Tensor()).to(self.policy.device)
        self.action_logprob_history = []
        self.reward_history = []
        self.loss_history = []

    def get_action(self, state):
        action_probs = self.policy(state)
        action_probs = T.distributions.Categorical(action_probs) #define action_probs as a multinoulli distribution

        action = action_probs.sample()

        action_logprob = action_probs.log_prob(action)
        
        # if self.action_logprob_history.size()[0] == 0:
        #     self.action_logprob_history = action_logprob
        # else:
        #     T.cat((self.action_logprob_history, action_logprob))
        self.action_logprob_history.append(action_logprob)

        return action.cpu().item()


    def learn(self): 
        self.policy.optimizer.zero_grad()
        reward_history = T.tensor(self.reward_history, dtype=T.float32).to(self.policy.device)
        
        
        #Calculate Return values
        # R_list = T.zeros_like(reward_history)
        R_list = np.zeros_like(self.reward_history)
        for t in range(len(self.reward_history)):
            R_t = 0
            discount = 1
            for k in range(t, len(self.reward_history)):
                R_t += self.reward_history[k] * discount
                discount = self.gamma * discount
            R_list[t] = R_t

        std = R_list.std() if R_list.std() > 0 else 1 #handles exception where std == 0
        R_list = (R_list - R_list.mean()) / std

        #Calculate loss and backprop
        # loss = T.sum(T.mul(R_list, self.action_logprob_history)).to(self.policy.device)
        # loss *= -1.0
        loss = 0
        for R, action_logprob in zip(R_list, self.action_logprob_history):
            loss += -R * action_logprob

        loss.backward()
        self.policy.optimizer.step()

        # self.action_logprob_history = T.autograd.Variable(T.Tensor()).to(self.policy.device)
        self.action_logprob_history = []
        self.reward_history = []

    def save_model(self):
        self.policy.save()
    
    def load_model(self):
        self.policy.load()
