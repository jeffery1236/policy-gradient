import gym
import numpy as np
import matplotlib.pyplot as plt
from PolicyGradientAgent import PolicyGradientAgent
from utils import plot_learning_curve
plt.style.use('seaborn-whitegrid')

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    print(env.observation_space.shape, env.action_space.n)
    env.reset()

    test_mode = False
    num_games = 2500
    best_score = -np.inf
    scores = []
    eps_history = []

    state_dims = env.observation_space.shape[0]
    num_actions = env.action_space.n
    lr = 0.001
    gamma = 0.99
    agent = PolicyGradientAgent(lr=lr, gamma=gamma, state_dims=state_dims,
                                num_actions=num_actions, env_name='lunar_lander',
                                 checkpoint_dir='temp/')
    if test_mode:
        agent.load_model()

    # env = gym.wrappers.Monitor(env, 'temp/lunar_lander',
    #                             video_callable=lambda episode_id: True, force=True)

    for count in range(num_games):
        state = env.reset()
        done = False
        score = 0

        while not done:
            env.render()
            action = agent.get_action(state)
            new_state, reward, done, _ = env.step(action)
            agent.reward_history.append(reward)

            score += reward
            state = new_state
        
        if not test_mode:
            agent.learn()
        scores.append(score)
        
        if count % 100 == 0:
            avg_score = np.mean(scores[-100:])
            if score > best_score:
                best_score = score
                if not test_mode:
                    agent.save_model()

            print(f"Episode: {count+1}, score: {score}, current average score: {avg_score}")
    env.close()
    x = range(1, num_games+1)
    plt.plot(x, scores)
    plt.show()