import argparse
import gymnasium as gym
import ale_py
import torch as th
from torch.utils.tensorboard import SummaryWriter

from simple_dreamer import utils
from simple_dreamer.networks.dreamer_v3 import DreamerV3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="simplenet.yaml", help="your configuration file\
                            that contains hyperparams and layers of for neural network")
    parser.add_argument("--record", nargs='?', type=str, const="default_experiment", default=None, help="record training statistics for tensorboard")
    args = parser.parse_args()
    config = utils.loadConfig(args.config)
    if args.record:
        if args.record == "default_experiment":
            writer = SummaryWriter()
        else:
            writer = SummaryWriter(f"runs/{args.record}")  

    device = "cuda" if th.cuda.is_available() else "cpu"
    env = gym.make("LunarLander-v3", max_episode_steps=1000)
    #env = gym.make("Breakout-v4", obs_type="grayscale")

    encoder_type = "mlp" # cnn if image input else mlp for vecor input
    action_space = env.action_space.n
    obs_shape = env.observation_space.shape
    action_type = env.action_space.dtype
    print(f"action_space: {action_space}, obs_shape: {obs_shape}, action_type: {action_type}")
    
    dreamer = DreamerV3(config, encoder_type, obs_shape, action_space, device)

    for _ in range(config.episodesBeforeStart):
        recurrent_state, latent_state, action = dreamer.init_states()

        obs, _ = env.reset()
        done = False
        while not done:
            with th.no_grad():
                obs = th.as_tensor(obs).to(device)
                action, recurrent_state, latent_state =\
                    dreamer.sample_action(obs, recurrent_state, latent_state, action)

            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().argmax())
            done = terminated or truncated

            dreamer.add(obs.flatten(), th.as_tensor(action), th.as_tensor(reward), done)
            obs = next_obs
        
        env_per_grad_steps = (config.batch_size * config.batch_length\
            * config.action_repeats) // config.replay_ratio
        recurrent_state, latent_state, action = dreamer.init_states()
        n_episodes = 0
        score = 0
        obs, _ = env.reset()
        for grad_step in range(config.grad_steps):
            for i in range(env_per_grad_steps):
                with th.no_grad():
                    obs = th.as_tensor(obs).to(device)
                    action, recurrent_state, latent_state =\
                        dreamer.sample_action(obs, recurrent_state, latent_state, action)

                next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().argmax())
                score+=reward
                done = terminated or truncated
                dreamer.add(obs.flatten(), th.as_tensor(action), th.as_tensor(reward), done)
                
                if done:
                    if args.record:
                        writer.add_scalar('score', score, n_episodes)
                    n_episodes +=1
                    score = 0
                    obs, _ = env.reset()
                else:
                    obs = next_obs
                
            data = dreamer.buffer.sample_batch(config.batch_size, config.batch_length)
            loss_dict = dreamer.learn(data)
            if args.record:
                utils.log_losses(writer, loss_dict, grad_step)

        if args.record:
            writer.close()
        print("ended")
    