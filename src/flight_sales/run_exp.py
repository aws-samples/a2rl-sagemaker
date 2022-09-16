from __future__ import annotations

import argparse
import math
import os
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Literal

import a2rl as wi
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from nptyping import Float, NDArray
from smallmatter.ds import SimpleMatrixPlotter
from stable_baselines3 import A2C
from typing_extensions import TypeAlias

from .flight_sales_gym import flight_sales_gym, reward_functions

warnings.filterwarnings("ignore")

# Flake8 chokes when using Shape instead of Literal.
# See: https://github.com/ramonhagenaars/nptyping/issues/63
EpisodeRewards: TypeAlias = NDArray[Literal["Timestep"], Float]
ExperimentRewards: TypeAlias = NDArray[Literal["Episode, Timestep"], Float]


def main():
    DEFAULT_MODEL_DIR = os.environ.get("SM_MODEL_DIR", "results")
    DEFAULT_OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR", "results")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        help=f"Where to write model artifacts (default: {DEFAULT_MODEL_DIR}/)",
        default=DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"Where to write additional artifacts (default: {DEFAULT_OUTPUT_DIR}/)",
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Reduce a2rl sampling size (NOTE: cannot be passed as a SageMaker hyperparameter)",
    )
    parser.add_argument(
        "--reward-function",
        default=None,
        nargs="?",
        help="When specified, run just the specific reward function (default: sweep through all)",
    )
    args = parser.parse_args()
    logger.info("CLI args: {}", vars(args))

    run_all(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        f_reward_name=args.reward_function,
        fast_mode=args.fast_mode,
    )


def run_all(
    *,
    model_dir: Path,
    output_dir: Path,
    f_reward_name: None | str = None,
    fast_mode: bool = False,
) -> None:
    # Construct the experiments to sweep through
    kwargs = {"a2rl_sampling_size": 10} if fast_mode else {}
    if f_reward_name is None:
        f_rewards = reward_functions
    else:
        f_rewards = {f_reward_name: reward_functions[f_reward_name]}
    logger.info("Will sweep through {} experiments: {}", len(f_rewards), list(f_rewards))

    for f_reward in f_rewards:
        logger.info("Running experiment with reward function {}()", f_reward)
        run(f_reward, model_dir, output_dir, **kwargs)

    logger.success("All done")


def run(f_reward: str, model_dir: Path, output_dir: Path, a2rl_sampling_size: int = 500) -> None:
    env = flight_sales_gym(f_reward=f_reward)
    previous_model = A2C(
        policy="MlpPolicy",
        env=env,
        verbose=False,
    )  # type: ignore[call-arg,arg-type]
    previous_model.learn(total_timesteps=1000)

    cap_env = wi.TransitionRecorder(env)
    previous_model.set_env(cap_env)
    previous_model.learn(total_timesteps=10000)

    if not f_reward.endswith("_no_jitter"):
        wi_df = wi.WiDataFrame(
            cap_env.df.values,
            columns=["season", "freight_price", "ticket_price", "reward"],
            states=["season", "freight_price"],
            actions=["ticket_price"],
            rewards=["reward"],
        )
    else:
        # no_jitter means constant freight_price, hence won't even pass the tokenizer's check.
        wi_df = wi.WiDataFrame(
            cap_env.df.values,
            columns=["season", "freight_price", "ticket_price", "reward"],
            states=["season"],
            actions=["ticket_price"],
            rewards=["reward"],
        ).trim()

    logger.info("Running random agent...")
    random_agent_rewards = run_random_agent(env)

    logger.info("Running UCB agent...")
    ucb_agent_rewards = run_ucb_agent(env)

    logger.info("Running PPO agent...")
    ppo_agent_rewards = run_ppo_agent(env, previous_model)

    # MBP M1: 5ep ~5min
    logger.info("Running whatif with sample_size={}...", a2rl_sampling_size)
    model_dir = model_dir / f"model-{env.f_reward.__name__}"  # type: ignore[attr-defined]
    tokenizer = wi.AutoTokenizer(wi_df, block_size_row=2)
    model_builder = wi.GPTBuilder(tokenizer, model_dir)
    model_builder.fit()
    whatif_rewards = run_whatif(env, model_builder, sample_size=a2rl_sampling_size, ep=5)

    rewards: dict[str, ExperimentRewards] = {
        "whatif": whatif_rewards,
        "random_agent": random_agent_rewards,
        "ucb_agent": ucb_agent_rewards,
        "ppo_agent": ppo_agent_rewards,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    f_reward_name = env.f_reward.__name__  # type: ignore[attr-defined]
    np.savez(output_dir / f"results-{f_reward_name}.npz", **rewards)
    smp = plot_results(rewards, suptitle=f"Reward function: {f_reward_name}()")
    smp.savefig(output_dir / f"results-{f_reward_name}.png", dpi=150)


def run_whatif(
    env: flight_sales_gym,
    model_builder: wi.simulator.BaseBuilder,
    ep: int = 1,
    sample_size: int = 500,
) -> ExperimentRewards:
    tokenizer = model_builder.tokenizer
    simulator = wi.Simulator(tokenizer, model_builder.model, max_steps=366)

    env.reset()
    env.step(0.5)

    len_ar = len(tokenizer.df.actions) + len(tokenizer.df.rewards)
    field_tokenizer = tokenizer.field_tokenizer
    ctx = np.tile(
        field_tokenizer.transform(env.context(tail=2, fillna=True)).values.ravel()[:-len_ar],
        (ep, 1),
    )

    envs = [deepcopy(env) for _ in range(ep)]
    rewards = np.zeros((ep, 366))

    for __ in range(365):  # 364 days remaining.
        ########################################################################
        # For each trajectory, get a recommended action.
        ########################################################################
        df_recommended = simulator.sample(ctx, max_size=sample_size, as_token=False)
        df_recommended["ticket_price"] = df_recommended["ticket_price"].round(2)
        assert "_trajectory" not in df_recommended.columns
        df_recommended["_trajectory"] = df_recommended.index // sample_size

        # fmt: off
        actions = (
            df_recommended
            .groupby(["_trajectory", "ticket_price"])["reward"]
            .quantile(0.9)
            .groupby("_trajectory")
            .idxmax()
            .apply(lambda x: x[1])
            .sort_index()
        )
        # fmt: on
        ########################################################################

        for i, e in enumerate(envs):
            e.step(actions[i])
            # fmt: off
            ctx[i, :] = (
                field_tokenizer
                .transform(env.context(tail=2, fillna=True))
                .values
                .ravel()[:-len_ar]
            )
            # fmt: on

    rewards[:, :] = [e.reward_history for e in envs]

    return rewards


def run_random_agent(env, ep=100) -> ExperimentRewards:
    random_agent_rewards = []
    for _ in range(ep):  # Number of episodes
        # print("#", end="", flush=True)
        env.reset()
        env.step(0.5)
        done = False
        for __ in range(365):  # Number of days per episode
            action = env.action_space.sample()
            state, reward, done, msg = env.step(action)
        random_agent_rewards.append(env.reward_history)
    # print()
    return np.asarray(random_agent_rewards)


class UpperConfBoundArm:
    def __init__(self, id):
        self.id = id  # set the arm's id
        self.avg_reward = 0.0  # average reward
        self.counts = 0.0  # how many times played
        self.upper_conf_bound = 10000  # Initially big number to try all arms

    def pull(self, reward):  # Calculate the reward and update status when you pull the arm
        self.counts = self.counts + 1  # increment play number
        self.avg_reward = (
            self.avg_reward * (self.counts - 1) + reward
        ) / self.counts  # udate average reward for this arm

    def updateUpperConfidenceBound(self, round_t):
        if self.counts != 0:
            # Remember upper confidence bound gets smaller when we play that arm more often.
            self.upper_conf_bound = math.sqrt(
                2 * math.log(round_t) / self.counts
            )  # update upper confidence bound


def run_ucb_agent(env, ep=100) -> ExperimentRewards:
    ucb_agent_rewards = []
    actions = [np.round(i, 4) for i in np.linspace(0, 1, 41)]
    # print(actions)

    for _ in range(ep):  # Episode counts
        # print("#", end="", flush=True)
        arms = [UpperConfBoundArm(i) for i in range(0, len(actions))]
        env.reset()  # noqa: F841
        count = 1
        for __ in range(366):  # Number of days per episode
            selection_criteria = [arm.avg_reward + arm.upper_conf_bound for arm in arms]
            i = np.argmax(selection_criteria)
            action = actions[i]

            state, reward, done, msg = env.step(action)
            arms[i].pull(reward)
            for arm in arms:
                arm.updateUpperConfidenceBound(count)

            count += 1

        ucb_agent_rewards.append(env.reward_history)
    # print()
    return np.asarray(ucb_agent_rewards)


def run_ppo_agent(env, model, ep=100):
    rl_agent_rewards = []
    for _ in range(ep):  # Episode count
        # print("#", end="", flush=True)
        state = env.reset()
        for __ in range(366):  # Number of days per episode
            action = model.predict(state)
            state, reward, done, msg = env.step(float(action[0]))

        rl_agent_rewards.append(env.reward_history)
    # print()
    return np.asarray(rl_agent_rewards)


def plot_total_rewards(d: dict[str, ExperimentRewards], ax=None) -> None:
    mean_total_rewards = {k: np.mean(np.sum(v, axis=1)) for k, v in d.items()}
    ax = ax if ax else plt.subplots(figsize=(4, 3))[1]

    y_pos = list(range(len(mean_total_rewards)))
    p1 = ax.barh(y_pos, mean_total_rewards.values(), align="edge")

    ax.set_xlabel("Rewards")
    ax.set_yticks(y_pos, d)
    ax.tick_params(axis="both", which="major")
    ax.bar_label(
        p1,
        [f"{reward:,.0f}" for reward in mean_total_rewards.values()],
        label_type="center",
        color="white",
    )


def plot_cumulative_rewards(d: dict[str, EpisodeRewards], ax=None) -> None:
    ax = ax if ax else plt.subplots(figsize=(4, 3))[1]
    for k, v in d.items():
        ax.plot(v, label=k, alpha=0.75)
    ax.legend(d, loc="upper left")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative reward")
    ax.tick_params(axis="both", which="major")


def plot_episode_snapshot(
    ep_wi: EpisodeRewards,
    ep_other: EpisodeRewards,
    other_label: str,
    ax=None,
) -> None:

    ax = ax if ax else plt.subplots(figsize=(4, 3))[1]
    ax.plot(ep_other, "rx", label=other_label, markersize=4, alpha=0.75)
    ax.plot(ep_wi, "g.", label="whatif", markersize=4, alpha=0.75)

    handles, labels = ax.get_legend_handles_labels()
    handles = reversed(handles)
    labels = reversed(labels)
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.125),
        ncol=2,
    )

    ax.set_xlabel("Day")
    ax.set_ylabel("Reward")
    ax.tick_params(axis="both", which="major")


def plot_results(d: dict[str, ExperimentRewards], suptitle: None | str = None):
    smp = SimpleMatrixPlotter(5, init_figcount=len(d) + 1, figsize=(4, 4))
    plot_total_rewards(d, ax=smp.pop())
    plot_cumulative_rewards(
        {k: np.mean(np.cumsum(v, axis=1), axis=0) for k, v in d.items()},
        smp.pop(),
    )

    whatif_rewards = d["whatif"]
    other_rewards = {k: v for k, v in d.items() if k != "whatif"}
    for k, v in other_rewards.items():
        # Pick one of the episodes
        ep_wi = whatif_rewards[-1]
        ep_other = v[-1]
        plot_episode_snapshot(ep_wi=ep_wi, ep_other=ep_other, other_label=k, ax=smp.pop())

    if suptitle:
        smp.fig.suptitle(suptitle)
    smp.trim()
    smp.fig.tight_layout()

    return smp


if __name__ == "__main__":
    main()
