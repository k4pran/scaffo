import argparse
import errno
import os
import json
import logging

import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from agent_frame.agent_base import AgentBase

logging.basicConfig(level=logging.INFO, format='[%(levelname)s]\t%(asctime)s\t\t %(message)s')
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

CONFIG_FILE_NAME = 'config.json'
PLOT_FILE_PATH_DEFAULT = "../output/plot.png"

render = False
episodes = 1000
plot_frequency = 10
log_frequency = 1
video_frequency = 100
video_dir = "../output"
plot_title = "Agent Score by Number of Episodes"
plot_x_label = "Episode"
plot_y_label = "Score"


def load_config(**config):
    global render, episodes, plot_frequency, log_frequency, video_frequency, video_dir
    render = config.get('render', render)
    episodes = config.get('episodes', episodes)
    plot_frequency = config.get('plot_frequency', plot_frequency)
    log_frequency = config.get('log_frequency', log_frequency)
    video_frequency = config.get('video_frequency', video_frequency)
    video_dir = config.get('video_dir', video_dir)


if os.path.isfile(CONFIG_FILE_NAME):
    with open('config.json', 'r') as config:
        load_config(**json.load(config))

master_parser = argparse.ArgumentParser().add_subparsers()

game_args_parser = master_parser.add_parser("Game Settings")
game_args_parser.add_argument('--render', '-r', action='store_true',
                              help="Set to always render the environment")
game_args_parser.add_argument('--episodes', '-n', type=int,
                              help="Sets number of episodes to play frequency")
game_args_parser.add_argument('--plot_frequency', '--pf', type=int,
                              help="Sets plotting frequency, set to 0 to disable. Will plot progress every n episodes")
game_args_parser.add_argument('--plot_title', '--pt', type=str,
                              help="Sets title for plots")
game_args_parser.add_argument('--plot_x_label', '--px', type=str,
                              help="Sets label for plot's x-axis")
game_args_parser.add_argument('--plot_y_label', '--py', type=str,
                              help="Sets label for plot's y-axis")
game_args_parser.add_argument('--log_frequency', '--lf', type=int,
                              help="Sets logging frequency, set to 0 to disable. Will log progress every n episodes")
game_args_parser.add_argument('--video_frequency', '--vf', type=int,
                              help="Video recording frequency, records every n episodes")
game_args_parser.add_argument('--video_dir', '--vd', type=str,
                              help="Sets destination directory to save video recordings too")

game_args = game_args_parser.parse_known_args()[0]
overrides = {k: v for k, v in vars(game_args_parser.parse_known_args()[0]).items() if v is not None
             and not (type(v) is bool and not v)}

load_config(**overrides)


def plot(episodes, scores):
    df = pd.DataFrame({
        plot_x_label: episodes,
        plot_y_label: scores
    })
    ax = sns.regplot(x=plot_x_label, y=plot_y_label, data=df)
    ax.set_title("Agent Score by Number of Episodes")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(PLOT_FILE_PATH_DEFAULT)
    plt.close()


def determine_extension(env):
    modes = env.metadata.get('render.modes', [])
    if 'rgb_array' not in modes:
        if 'ansi' in modes:
            return ".json"
        else:
            LOG.warning(
                "Unable to record video due to unsupported mode. Supported modes are 'rgb_array' and 'ansi'".format(
                    env))
            return
    return ".mp4"


def start(env, agent: AgentBase):
    global video_recorder
    scores = []
    total_steps = 0
    video_recorder = None
    video_enabled = True
    video_ext = determine_extension(env)
    if not video_ext:
        video_enabled = False
    for episode in range(1, episodes):

        if (episode % video_frequency) == 0 and video_enabled:
            video_recorder = VideoRecorder(env, video_dir + "/{}{}".format(episode, video_ext), enabled=True)

        score, steps = run_episode(env, agent, video_recorder)

        scores.append(score)
        total_steps += steps

        if (episode + 1) % log_frequency == 0:
            LOG.info(
                "Episode: {} SCORE: {} STEPS: {} TOTAL_STEPS: {}".format(episode, score,
                                                                         steps, total_steps))

        if episode % plot_frequency == 0:
            plot([i for i in range(episode)], scores)

        if video_recorder:
            video_recorder.close()
            video_recorder = None
    env.close()


def run_episode(env, agent, video_recorder=None):
    done = False
    state = env.reset()
    score = 0
    steps = 1
    while not done:
        agent.before()

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.learn(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state

        if video_recorder:
            env.render()
            video_recorder.capture_frame()
        elif render:
            env.render()

        score += reward

        steps += 1
        agent.after()

    return score, steps


def init_output_dir():
    try:
        os.makedirs("../output")
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def resolve_env(gym_env):
    global env
    assert gym_env is not None
    if type(gym_env) == str:
        return gym.make(gym_env)
    return gym_env


def run(gym_env, agent: AgentBase):
    init_output_dir()
    env = resolve_env(gym_env)
    start(env, agent)
    env.close()
