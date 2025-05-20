from env import Environment
#from agent import Agents
from greedyagent import GreedyAgents
from agent import RandomAgent, Agents

import numpy as np

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent Reinforcement Learning for Delivery")
    parser.add_argument("--num_agents", type=int, default=5, help="Number of agents")
    parser.add_argument("--n_packages", type=int, default=10, help="Number of packages")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Maximum time steps for the environment")
    parser.add_argument("--map", type=str, default="map.txt", help="Map name")

    args = parser.parse_args()
    np.random.seed(args.seed)
    all_reward = []
    all_delivered = []
    for i in range(10):
        env = Environment(map_file=args.map, max_time_steps=args.max_time_steps,
                        n_robots=args.num_agents, n_packages=args.n_packages,
                        seed = args.seed)
        
        state = env.reset()
        agents = GreedyAgents()
        agents.init_agents(state)
        # print(state)
        done = False
        t = 0
        while not done:
            actions = agents.get_actions(state)
            next_state, reward, done, infos = env.step(actions)
            state = next_state
            t += 1

        print("Episode finished")
        print("Total reward:", infos['total_reward'])
        print("Total time steps:", infos['total_time_steps'])
        delivered_count = sum(1 for pkg in env.packages if pkg.status == 'delivered')
        print(f"Packages Delivered: {delivered_count}/{env.n_packages}")
        all_reward.append(env.total_reward)
        all_delivered.append(delivered_count)
    print("Average reward:", np.mean(all_reward))
    print(all_reward)
    print("Average packages delivered:", np.mean(all_delivered))
    print(all_delivered)
