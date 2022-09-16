from __future__ import annotations

from functools import partial, update_wrapper
from math import cos, pi
from random import random
from typing import Protocol, TypedDict

import a2rl as wi
import gym
import numpy as np
from gym import spaces


def fsigmoid(x: float, day: int) -> float:
    """This is the sigmoid function that we use for the propensity to buy.

    Args:
        x: Price
        day: Day of year.

    Returns:
        Conversion probability
    """
    a, b, c = parameters(day)
    return c / (1.0 + np.exp(-a * (b - x)))


def parameters(day: int) -> tuple[float, float, float]:
    """Get parameters for logistic model, which are seasonal.

    Returns:
        (smoothness, mid_price, conversion)
    """
    # fmt: off
    a = 0.3 + (cos(day * (2.0 * pi / 365)) + 2) / 50  # a = Smoothness of transition
    b = 10                                            # b = Transition (i.e., mid price)
    c = 0.2 + (cos(day * (2.0 * pi / 365)) + 2) / 20  # c = Max. propensity
    # fmt: on

    return (a, b, c)


def seasonality(day) -> float:
    """Convert ``day`` to a seasonality coefficient."""
    season = 0.5 * (cos(day * (2.0 * pi / 365)) + 1)
    return season


class Config(TypedDict):
    freight_price: float
    max_fare: int
    daily_quota: int
    visitors: int
    max_weight: int
    max_time: int


config: Config = {
    "freight_price": 0.2,
    "max_fare": 20,
    "daily_quota": 20,
    "visitors": 40,
    "max_weight": 5000,
    "max_time": 1000,
}


def revenue(
    sold: float,
    pax_price: float,
    jitter: bool = True,
    max_weight=config["max_weight"],
    base_freight_price=["freight_price"],
    **kwargs,
) -> tuple[float, float]:
    """Calculate revenue."""
    freight_price = (
        base_freight_price + random()  # nosec: B311 (not for cryptograph)
        if jitter
        else base_freight_price
    )  # nosec: B311 (not for cryptograph)
    return (sold * pax_price + (max_weight - sold * 100) * freight_price), freight_price


def profit(sold: float, pax_price: float, jitter: bool = True, **kwargs) -> tuple[float, float]:
    """Calculate profit."""
    opex = 500
    seat_cost = 1 - 0.35 * random() if jitter else 1 - 0.35  # nosec: B311 (not for cryptography)
    return (sold * pax_price * 20 * seat_cost - opex), seat_cost


def partial_revenue(*, base_freight_price: float, jitter: bool = True):
    """Get a :func:`~revenue()` function with specific ``freight_price`` and `jitter``."""
    p = partial(revenue, jitter=jitter, base_freight_price=base_freight_price)
    update_wrapper(p, revenue)
    p.__name__ += f"_{base_freight_price:.2f}".replace(".", "_")  # type: ignore[attr-defined]
    if not jitter:
        p.__name__ += "_no_jitter"  # type: ignore[attr-defined]
    return p


def get_profit_no_jitter():
    """Get a non-jitter :func:`~profit()` function."""
    p = partial(profit, jitter=False)
    update_wrapper(p, profit)
    p.__name__ += "_no_jitter"  # type: ignore[attr-defined]
    return p


class RewardFunction(Protocol):
    def __call__(
        self,
        sold: float,
        pax_price: float,
        **kwargs,
    ) -> tuple[float, float]:
        ...


# fmt: off
reward_functions: dict[str, RewardFunction] = {
    "revenue_0_20"          : partial_revenue(base_freight_price=0.20),
    "revenue_0_20_no_jitter": partial_revenue(base_freight_price=0.20, jitter=False),
    "revenue_0_10"          : partial_revenue(base_freight_price=0.10),
    "revenue_0_10_no_jitter": partial_revenue(base_freight_price=0.10, jitter=False),
    "revenue_0_05"          : partial_revenue(base_freight_price=0.05),
    "revenue_0_05_no_jitter": partial_revenue(base_freight_price=0.05, jitter=False),
    "revenue_0_02"          : partial_revenue(base_freight_price=0.02),
    "revenue_0_02_no_jitter": partial_revenue(base_freight_price=0.02, jitter=False),
    "revenue_0_01"          : partial_revenue(base_freight_price=0.01),
    "revenue_0_01_no_jitter": partial_revenue(base_freight_price=0.01, jitter=False),
    "profit"                : profit,
    "profit_no_jitter"      : get_profit_no_jitter(),
}
# fmt: on


class flight_sales_gym(gym.Env):
    def __init__(
        self,
        f_reward: str | RewardFunction = "revenue_0_20",
        max_time: int = config["max_time"],
        daily_quota: int = config["daily_quota"],
        visitors: int = config["visitors"],
        max_fare: int = config["max_fare"],
    ):
        super().__init__()
        self.f_reward: RewardFunction = (
            reward_functions[f_reward] if isinstance(f_reward, str) else f_reward
        )
        self.max_time = max_time
        self.daily_quota: int = daily_quota
        self.visitors = visitors
        self.priceScale = max_fare
        self.action_space = spaces.Box(0, 1, (1,), dtype=np.float64)
        self.observation_space = spaces.Box(np.array([1, 0.1]), np.array([30, 5]), dtype=np.float64)
        # self.observation_space = spaces.Box(np.array([0, 0.1]), np.array([1, 5]),dtype=np.float64)

        self.history = wi.WiDataFrame(
            np.zeros((self.max_time, 4)),
            columns=["season", "freight_price", "ticket_price", "reward"],
            states=["season", "freight_price"],
            actions=["ticket_price"],
            rewards=["reward"],
        )

    def render(self):
        pass

    def step(self, action):
        """Gym step function.

        Returns:
            (state, reward, done, msg)
        """
        if self.day > self.max_time:
            raise ValueError("End of step. Please reset().")

        if isinstance(action, np.ndarray):
            action = action[0]

        # Calculate reward
        avail_seats = self.daily_quota
        actual_price = max(action * self.priceScale, 1e-5)
        for _ in range(self.visitors):
            if avail_seats > 0:
                if random() < fsigmoid(actual_price, self.day):  # nosec: B311 (not for cryptograph)
                    avail_seats -= 1
        sold = self.daily_quota - avail_seats
        reward, random_component = self.f_reward(sold=sold, pax_price=actual_price)

        # Fill-up this timestep's record
        self.reward_history.append(reward)
        self.history.loc[self.day - 1, ["ticket_price", "reward"]] = (action, reward)

        # To the next day.
        self.day += 1
        season = seasonality(self.day)
        done = self.day > self.max_time
        if not done:
            self.history.iloc[self.day - 1] = (season, random_component, np.nan, np.nan)

        return np.array([season, random_component]), reward, done, {}

    def context(self, tail: int = -1, fillna: bool = False):
        start = 0 if tail < 0 else max(self.day - tail, 0)
        ctx = self.history.iloc[start : self.day].copy()
        if fillna:
            ctx.fillna(method="ffill", inplace=True)
        return ctx

    def reset(self):
        """Gym reset function.

        Returns:
            state
        """
        self.day: int = 1
        self.history.iloc[1:] = 0.0
        self.reward_history = []

        season = seasonality(self.day)
        random_component = self.f_reward(0, 0)[-1]  # "Lag" feature.
        self.history.iloc[0] = (season, random_component, np.nan, np.nan)

        return np.array([season, random_component])
