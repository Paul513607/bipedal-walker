import gym
import numpy as np
from actor_critic_agent import ActorCriticAgent
from ploting import plot_agent_learning


if __name__ == '__main__':
    load_agent = False
    testing_agent = False
    plot_data = True
    render_walker = True
    if render_walker:
        env = gym.make("BipedalWalker-v3", render_mode='human')
    else:
        env = gym.make("BipedalWalker-v3")
    agent = ActorCriticAgent(env=env, checkpoint_directory=r'temp/tensorflow_ddpg_3500')
    nr_of_games = 3500
    plot_path = r'plots/tensor_ddpg_' + str(nr_of_games) + '.png'

    best_score = env.reward_range[0]
    score_history = []

    if load_agent:
        nr_steps = 0
        while nr_steps <= agent.batch_size:
            env_state = env.reset()
            env_state = env_state[0]
            action = env.action_space.sample()
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.add_to_buffer(env_state, action, reward, new_state, terminated or truncated)
            nr_steps += 1
        agent.learn()
        agent.load_model()

    for i in range(nr_of_games):
        env_state = env.reset()
        env_state = env_state[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(env_state, testing_agent)
            new_state, reward, terminated, truncated, info = env.step(action)
            score += reward
            done = terminated or truncated
            agent.add_to_buffer(env_state, action, reward, new_state, done)
            if not testing_agent:
                agent.learn()
            env_state = new_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not testing_agent:
                agent.save_model()
        print('Episodul', i, 'score %1.f' % score, 'avg score %1.f' % avg_score)

    if plot_data:
        plot_agent_learning(score_history, plot_path)
