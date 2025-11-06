import argparse
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
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
    env = RecordEpisodeStatistics(env)

    encoder_type = "mlp" # cnn if image input else mlp for vecor input
    action_space = env.action_space.n
    obs_shape = env.observation_space.shape
    action_type = env.action_space.dtype
    num_envs = 1
    print(f"action_space: {action_space}, obs_shape: {obs_shape}, action_type: {action_type}")
    
    dreamer = DreamerV3(config, encoder_type, obs_shape, action_space, device)

    env_per_grad_steps = (config.batch_size * config.batch_length\
        * config.action_repeats) // config.replay_ratio

    for i in range(config.episodes_before_start):
        recurrent_state, latent_state, action = dreamer.init_states(num_envs)

        obs, info = env.reset(seed=config.seed)
        done = False
        is_first = True
        while not done:
            with th.no_grad():
                obs = th.as_tensor(obs).to(device)
                recurrent_state, latent_state, action =\
                    dreamer.sample_action(obs, recurrent_state, latent_state, action)

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().argmax())
            done = terminated or truncated
            if done:
                recurrent_state, latent_state, action = dreamer.init_states(num_envs)
                print(f"episode: {i+1}, score: {info['episode']['r']}")


            dreamer.add(obs.flatten(), th.as_tensor(action), th.as_tensor(reward), done, is_first)
            obs = next_obs
            is_first = False
    
    is_first = True
    recurrent_state, latent_state, action = dreamer.init_states(num_envs)
    episode_num = 0
    obs, info = env.reset(seed=config.seed)
    for grad_step in range(config.grad_steps):
        # update target critic parametrs
        if grad_step % config.critic.target_network_update_freq == 0:
            for params, target_params in zip(
                dreamer.critic.parameters(),
                dreamer.target_critic.parameters()
            ):
                target_params.data.copy_(config.critic.tau * params.data\
                    + (1 - config.critic.tau) * target_params.data)
        
        # collect data
        for i in range(env_per_grad_steps):
            with th.no_grad():
                obs = th.as_tensor(obs).to(device)
                recurrent_state, latent_state, action =\
                    dreamer.sample_action(obs, recurrent_state, latent_state, action)

            next_obs, reward, terminated, truncated, info = env.step(action.cpu().numpy().argmax())
            done = terminated or truncated
            dreamer.add(obs.flatten(), th.as_tensor(action), th.as_tensor(reward), done, is_first)
            is_first = False

            if done:
                recurrent_state, latent_state, action = dreamer.init_states(num_envs)
                episode_score = info["episode"]["r"]
                print(f"episode: {episode_num+1}, score: {episode_score}")
                obs, info = env.reset(seed=config.seed)
                if args.record:
                    writer.add_scalar('score', episode_score, episode_num)
                episode_num += 1
                is_first = True
            else:
                obs = next_obs
        
        # sample data and train
        data = dreamer.buffer.sample(config.batch_length, config.batch_size)
        loss_dict = dreamer.learn(data)
        if args.record:
            utils.log_losses(writer, loss_dict, grad_step)

    if args.record:
        writer.close()
    print("ended")
    