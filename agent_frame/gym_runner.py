import argparse
import errno
import os
import json
import logging
from time import sleep

import gym
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import pandas as pd
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from agent_frame.agent_base import AgentBase

CONFIG_FILE_NAME = 'config.json'
PLOT_FILE_PATH_DEFAULT = "./output/media/plot.png"
EPISODE_SCORE_LOG_FIXED_LENGTH = 7
STEP_SCORE_LOG_FIXED_LENGTH = 6
STEPS_LOG_FIXED_LENGTH = 5

render = False
episodes = 1000
plot_frequency = 10
episode_log_frequency = 1
step_log_frequency = 1
video_frequency = 100
video_dir = "./output/media"
log_dir = "./output/log"
plot_title = "Agent Score by Number of Episodes"
plot_x_label = "Episode"
plot_y_label = "Score"

total_steps = None


def load_config(**config_overrides):
    global render, episodes, plot_frequency, episode_log_frequency, \
        step_log_frequency, log_dir, video_frequency, video_dir
    render = config_overrides.get('render', render)
    episodes = config_overrides.get('episodes', episodes)
    plot_frequency = config_overrides.get('plot_frequency', plot_frequency)
    episode_log_frequency = config_overrides.get('episode_log_frequency', episode_log_frequency)
    step_log_frequency = config_overrides.get('step_log_frequency', step_log_frequency)
    log_dir = config_overrides.get('log_dir', log_dir)
    video_frequency = config_overrides.get('video_frequency', video_frequency)
    video_dir = config_overrides.get('video_dir', video_dir)


if os.path.isfile(CONFIG_FILE_NAME):
    with open(CONFIG_FILE_NAME, 'r') as config:
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
game_args_parser.add_argument('--episode_log_frequency', '--ef', type=int,
                              help="Sets episode logging frequency, set to 0 to disable. Will log progress every n episodes")
game_args_parser.add_argument('--step_log_frequency', '--sf', type=int,
                              help="Sets step logging frequency, set to 0 to disable. Will log progress every n steps")
game_args_parser.add_argument('--log_dir', '--ld', type=str,
                              help="Sets destination directory to save logs to")
game_args_parser.add_argument('--video_frequency', '--vf', type=int,
                              help="Video recording frequency, records every n episodes")
game_args_parser.add_argument('--video_dir', '--vd', type=str,
                              help="Sets destination directory to save video recordings to")

game_args = game_args_parser.parse_known_args()[0]
overrides = {k: v for k, v in vars(game_args_parser.parse_known_args()[0]).items() if v is not None
             and not (type(v) is bool and not v)}

load_config(**overrides)

if not os.path.exists(video_dir):
    os.makedirs(video_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s]\t%(asctime)s\t\t %(message)s',
    handlers=[
        logging.FileHandler(log_dir + "/log.txt"),
        logging.StreamHandler()
    ]
)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


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


def format_as_fixed_length(value, fixed_length):
    value = str(value)
    return value + (fixed_length - len(value)) * " "


def log_episode_summary(episodes, current_episode, score, steps):
    global total_steps

    current_episode = format_as_fixed_length(current_episode, len(str(episodes)))
    score = format_as_fixed_length(score, EPISODE_SCORE_LOG_FIXED_LENGTH)
    steps = format_as_fixed_length(steps, STEPS_LOG_FIXED_LENGTH)

    LOG.info(
        "EPISODE: %s EPISODE STEPS %s SCORE: %s TOTAL_STEPS: %s", current_episode, steps,
        score, total_steps)


def log_step_summary(step, total_score, reward):
    # +3 to account for difference between word lengths of 'EPISODE' and 'STEP'
    step = format_as_fixed_length(step, len(str(episodes)) + 3)
    total_score = format_as_fixed_length(total_score, STEP_SCORE_LOG_FIXED_LENGTH)
    LOG.info("STEP: %s TOTAL_SCORE: %s REWARD: %s", step, total_score, reward)


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

        if episode_log_frequency > 0 and (episode + 1) % episode_log_frequency == 0:
            log_episode_summary(episodes, episode, score, steps)

        if episode % plot_frequency == 0:
            plot([i for i in range(episode)], scores)

        if video_recorder:
            video_recorder.close()
            video_recorder = None
    env.close()


def run_episode(env, agent, video_recorder=None):
    global total_steps, render

    done = False
    state = env.reset()
    total_score = 0
    steps = 0
    while not done:
        agent.before(recording=video_recorder is not None,
                     rendering=render,
                     state=state,
                     total_steps=total_steps,
                     current_steps=steps)

        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.learn(state=state, action=action, reward=reward, next_state=next_state, done=done)

        state = next_state

        if video_recorder:
            sleep(0.08)
            env.render()
            for i in range(6):
                video_recorder.capture_frame()
        elif render:
            sleep(0.08)
            env.render()

        total_score += reward

        steps += 1
        total_steps += 1

        agent.after(recording=video_recorder is not None,
                    rendering=render,
                    state=state,
                    next_state=next_state,
                    reward=reward,
                    total_steps=total_steps,
                    current_steps=steps,
                    done=done,
                    score=total_score,
                    info=info)

        if step_log_frequency > 0 and steps % step_log_frequency == 0:
            log_step_summary(steps, total_score, reward)

    return total_score, steps


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
    global total_steps

    total_steps = 0
    init_output_dir()
    env = resolve_env(gym_env)
    start(env, agent)
    env.close()
