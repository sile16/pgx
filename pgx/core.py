# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import abc
from typing import Literal, Optional, Tuple, get_args

import jax
import jax.numpy as jnp

from pgx._src.struct import dataclass
from pgx._src.types import Array, PRNGKey

TRUE = jnp.bool_(True)
FALSE = jnp.bool_(False)


# Pgx environments are versioned like OpenAI Gym or Brax.
# OpenAI Gym forces user to specify version (e.g., `MountainCar-v0`); while Brax does not (e.g., `ant`)
# We follow the way of Brax. One can check the environment version by `Env.version`.
# We do not explicitly include version in EnvId for three reasons:
# (1) In game domain, performance measure is not the score in environment but
#     the comparison to other agents (i.e., environment version is less important),
# (2) we do not provide older versions (as with OpenAI Gym), and
# (3) it is tedious to remember and write version numbers.
#
# Naming convention:
# Hyphen - is used to represent that there is a different original game source, and
# Underscore - is used for the other cases.
EnvId = Literal[
    "2048",
    "animal_shogi",
    "backgammon",
    "bridge_bidding",
    "chess",
    "connect_four",
    "gardner_chess",
    "go_9x9",
    "go_19x19",
    "hex",
    "kuhn_poker",
    "leduc_holdem",
    # "mahjong",
    "minatar-asterix",
    "minatar-breakout",
    "minatar-freeway",
    "minatar-seaquest",
    "minatar-space_invaders",
    "othello",
    "pig",
    "shogi",
    "shut_the_box",
    "sparrow_mahjong",
    "tic_tac_toe",
]


@dataclass
class State(abc.ABC):
    """Base state class of all Pgx game environments. Basically an immutable (frozen) dataclass.
    A basic usage is generating via `Env.init`:

        state = env.init(jax.random.PRNGKey(0))

    and `Env.step` receives and returns this state class:

        state = env.step(state, action)

    Serialization via `flax.struct.serialization` is supported.
    There are 6 common attributes over all games:

    Attributes:
        current_player (Array): id of agent to play.
            Note that this does NOT represent the turn (e.g., black/white in Go).
            This ID is consistent over the parallel vmapped states.
        observation (Array): observation for the current state.
            `Env.observe` is called to compute.
        rewards (Array): the `i`-th element indicates the intermediate reward for
            the agent with player-id `i`. If `Env.step` is called for a terminal state,
            the following `state.rewards` is zero for all players.
        terminated (Array): denotes that the state is terminal state. Note that
            some environments (e.g., Go) have an `max_termination_steps` parameter inside
            and will terminate within a limited number of states (following AlphaGo).
        truncated (Array): indicates that the episode ends with the reason other than termination.
            Note that current Pgx environments do not invoke truncation but users can use `TimeLimit` wrapper
            to truncate the environment. In Pgx environments, some MinAtar games may not terminate within a finite timestep.
            However, the other environments are supposed to terminate within a finite timestep with probability one.
        legal_action_mask (Array): Boolean array of legal actions. If illegal action is taken,
            the game will terminate immediately with the penalty to the palyer.
    """

    current_player: Array
    observation: Array
    rewards: Array
    terminated: Array
    truncated: Array
    legal_action_mask: Array
    _step_count: Array

    @property
    @abc.abstractmethod
    def env_id(self) -> EnvId:
        """Environment id (e.g. "go_19x19")"""
        ...

    def _repr_html_(self) -> str:
        return self.to_svg()

    def to_svg(
        self,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> str:
        """Return SVG string. Useful for visualization in notebook.

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            str: SVG string
        """
        from pgx._src.visualizer import Visualizer

        v = Visualizer(color_theme=color_theme, scale=scale)
        return v.get_dwg(states=self).tostring()

    def save_svg(
        self,
        filename,
        *,
        color_theme: Optional[Literal["light", "dark"]] = None,
        scale: Optional[float] = None,
    ) -> None:
        """Save the entire state (not observation) to a file.
        The filename must end with `.svg`

        Args:
            color_theme (Optional[Literal["light", "dark"]]): xxx see also global config.
            scale (Optional[float]): change image size. Default(None) is 1.0

        Returns:
            None
        """
        from pgx._src.visualizer import save_svg

        save_svg(self, filename, color_theme=color_theme, scale=scale)


class Env(abc.ABC):
    """Environment class API.

    !!! example "Example usage"

        ```py
        env: Env = pgx.make("tic_tac_toe")
        state = env.init(jax.random.PRNGKey(0))
        action = jax.random.int32(4)
        state = env.step(state, action)
        ```

    """

    def __init__(self): ...

    def init(self, key: PRNGKey) -> State:
        """Return the initial state. Note that no internal state of
        environment changes.

        Args:
            key: pseudo-random generator key in JAX. Consumed in this function.

        Returns:
            State: initial state of environment

        """
        state = self._init(key)
        observation = self.observe(state)
        return state.replace(observation=observation)  # type: ignore

    def step(
        self,
        state: State,
        action: Array,
        key: Optional[Array] = None,
    ) -> State:
        """Step function."""
        is_illegal = ~state.legal_action_mask[action]
        current_player = state.current_player

        # If the state is already terminated or truncated, environment does not take usual step,
        # but return the same state with zero-rewards for all players
        state = jax.lax.cond(
            (state.terminated | state.truncated),
            lambda: state.replace(rewards=jnp.zeros_like(state.rewards)),  # type: ignore
            lambda: self._step(state.replace(_step_count=state._step_count + 1), action, key),  # type: ignore
        )

        # Taking illegal action leads to immediate game terminal with negative reward
        state = jax.lax.cond(
            is_illegal,
            lambda: self._step_with_illegal_action(state, current_player),
            lambda: state,
        )

        # All legal_action_mask elements are **TRUE** at terminal state
        # This is to avoid zero-division error when normalizing action probability
        # Taking any action at terminal state does not give any effect to the state
        state = jax.lax.cond(
            state.terminated,
            lambda: state.replace(legal_action_mask=jnp.ones_like(state.legal_action_mask)),  # type: ignore
            lambda: state,
        )

        observation = self.observe(state)
        state = state.replace(observation=observation)  # type: ignore

        return state

    def observe(self, state: State, player_id: Optional[Array] = None) -> Array:
        """Observation function."""
        if player_id is None:
            player_id = state.current_player
        else:
            warnings.warn("[Pgx] `player_id` in `observe` is deprecated. This argument will be removed in the future.", DeprecationWarning)
        obs = self._observe(state, player_id)
        return jax.lax.stop_gradient(obs)

    @abc.abstractmethod
    def _init(self, key: PRNGKey) -> State:
        """Implement game-specific init function here."""
        ...

    @abc.abstractmethod
    def _step(self, state, action, key) -> State:
        """Implement game-specific step function here."""
        ...

    @abc.abstractmethod
    def _observe(self, state: State, player_id: Array) -> Array:
        """Implement game-specific observe function here."""
        ...

    @property
    @abc.abstractmethod
    def id(self) -> EnvId:
        """Environment id."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Environment version. Updated when behavior, parameter, or API is changed.
        Refactoring or speeding up without any expected behavior changes will NOT update the version number.
        """
        ...

    @property
    @abc.abstractmethod
    def num_players(self) -> int:
        """Number of players (e.g., 2 in Tic-tac-toe)"""
        ...

    @property
    def num_actions(self) -> int:
        """Return the size of action space (e.g., 9 in Tic-tac-toe)"""
        state = self.init(jax.random.PRNGKey(0))
        return int(state.legal_action_mask.shape[0])

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Return the matrix shape of observation"""
        state = self.init(jax.random.PRNGKey(0))
        obs = self._observe(state, state.current_player)
        return obs.shape

    @property
    def _illegal_action_penalty(self) -> float:
        """Negative reward given when illegal action is selected."""
        return -1.0

    def _step_with_illegal_action(self, state: State, loser: Array) -> State:
        penalty = self._illegal_action_penalty
        reward = jnp.ones_like(state.rewards) * (-1 * penalty) * (self.num_players - 1)
        reward = reward.at[loser].set(penalty)
        return state.replace(rewards=reward, terminated=TRUE)  # type: ignore


class StochasticEnv(Env):
    """
    Interface for games separating Player Agency (Deterministic) from
    Nature Agency (Stochastic).
    """

    @abc.abstractmethod
    def _step_deterministic(self, state: State, action: Array) -> State:
        """
        PHASE 1: Player Decision (implementation).
        Applies the player's action.

        Returns:
            afterstate (State): A special intermediate state.
                - MUST set internal flag `state._is_stochastic = True`.
                - MUST zero out stochastic features in observation.
        """
        ...

    @abc.abstractmethod
    def _step_stochastic(self, state: State, action: Array) -> State:
        """
        PHASE 2: Nature Resolution (implementation).
        Applies a specific random outcome.

        Args:
            state: The 'afterstate' from step_deterministic.
            action: The specific random event index or value.

        Returns:
            next_state (State): The resulting state.
                - MUST set `state._is_stochastic = False`.
                - MUST populate stochastic features.
        """
        ...

    def step_deterministic(self, state: State, action: Array) -> State:
        """
        PHASE 1: Player Decision.
        Applies the player's action and updates observation.
        """
        state = self._step_deterministic(state, action)
        observation = self.observe(state)
        return state.replace(observation=observation)  # type: ignore

    def step_stochastic(self, state: State, action: Array) -> State:
        """
        PHASE 2: Nature Resolution.
        Applies a specific random outcome and updates observation.
        """
        state = self._step_stochastic(state, action)
        observation = self.observe(state)
        return state.replace(observation=observation)  # type: ignore

    @abc.abstractmethod
    def chance_outcomes(self, state: State) -> Tuple[Array, Array]:
        """
        Returns all possible random outcomes and their probabilities for the current afterstate.
        
        Returns:
            outcomes (Array): Indices/Values of possible outcomes.
            probs (Array): Probability of each outcome.
        """
        ...

    def step_stochastic_random(self, state: State, key: PRNGKey) -> State:
        """Helper to sample a random outcome and apply it."""
        outcomes, probs = self.chance_outcomes(state)
        outcome = jax.random.choice(key, outcomes, p=probs)
        return self.step_stochastic(state, outcome)

    def _step_stochastic_random(self, state: State, key: PRNGKey) -> State:
        """Internal helper without observation update (used by _step)."""
        outcomes, probs = self.chance_outcomes(state)
        outcome = jax.random.choice(key, outcomes, p=probs)
        return self._step_stochastic(state, outcome)

    def _step(self, state: State, action: Array, key: PRNGKey) -> State:
        """
        Combines Deterministic -> Stochastic phases for standard evaluation.
        Note: observation is updated by the parent step() method.
        """
        # Standard flow: State(P) -> action -> State(C) -> random -> State(P')

        # 1. Deterministic Step (Player Action)
        state = self._step_deterministic(state, action)

        # 2. Stochastic Step (Nature), if applicable
        # Some actions might lead to immediate termination or next player without chance
        # e.g., "Hold" in Pig
        state = jax.lax.cond(
            state._is_stochastic,
            lambda: self._step_stochastic_random(state, key),
            lambda: state
        )

        return state


def available_envs() -> Tuple[EnvId, ...]:
    """List up all environment id available in `pgx.make` function.

    !!! example "Example usage"

        ```py
        pgx.available_envs()
        ('2048', 'animal_shogi', 'backgammon', 'chess', 'connect_four', 'go_9x9', 'go_19x19', 'hex', 'kuhn_poker', 'leduc_holdem', 'minatar-asterix', 'minatar-breakout', 'minatar-freeway', 'minatar-seaquest', 'minatar-space_invaders', 'othello', 'shogi', 'sparrow_mahjong', 'tic_tac_toe')
        ```


    !!! note "`BridgeBidding` environment"

        `BridgeBidding` environment requires the domain knowledge of bridge game.
        So we forbid users to load the bridge environment by `make("bridge_bidding")`.
        Use `BridgeBidding` class directly by `from pgx.bridge_bidding import BridgeBidding`.

    """
    games = get_args(EnvId)
    games = tuple(filter(lambda x: x != "bridge_bidding", games))
    return games


def make(env_id: EnvId):  # noqa: C901
    """Load the specified environment.

    !!! example "Example usage"

        ```py
        env = pgx.make("tic_tac_toe")
        ```

    !!! note "`BridgeBidding` environment"

        `BridgeBidding` environment requires the domain knowledge of bridge game.
        So we forbid users to load the bridge environment by `make("bridge_bidding")`.
        Use `BridgeBidding` class directly by `from pgx.bridge_bidding import BridgeBidding`.

    """
    # NOTE: BridgeBidding environment requires the domain knowledge of bridge
    # So we forbid users to load the bridge environment by `make("bridge_bidding")`.
    if env_id == "2048":
        # Use optimized variant (1.5x faster with no-rotation optimization)
        from pgx.play2048_v2_all import Play2048V2All

        return Play2048V2All()
    elif env_id == "animal_shogi":
        from pgx.animal_shogi import AnimalShogi

        return AnimalShogi()
    elif env_id == "backgammon":
        # Use optimized variant (2x faster with branchless + fast obs)
        from pgx.backgammon_v2_all import BackgammonV2All

        return BackgammonV2All()
    elif env_id == "chess":
        from pgx.chess import Chess

        return Chess()
    elif env_id == "connect_four":
        # Use optimized variant (1.8x faster with height tracking)
        from pgx.connect_four_v2_bitboard import ConnectFourV2Bitboard

        return ConnectFourV2Bitboard()
    elif env_id == "gardner_chess":
        from pgx.gardner_chess import GardnerChess

        return GardnerChess()
    elif env_id == "go_9x9":
        from pgx.go import Go

        return Go(size=9, komi=7.5)
    elif env_id == "go_19x19":
        from pgx.go import Go

        return Go(size=19, komi=7.5)
    elif env_id == "hex":
        from pgx.hex import Hex

        return Hex()
    elif env_id == "kuhn_poker":
        from pgx.kuhn_poker import KuhnPoker

        return KuhnPoker()
    elif env_id == "leduc_holdem":
        from pgx.leduc_holdem import LeducHoldem

        return LeducHoldem()
    # elif env_id == "mahjong":
    #     from pgx.mahjong import Mahjong

    #     return Mahjong()
    elif env_id == "minatar-asterix":
        from pgx.minatar.asterix import MinAtarAsterix  # type: ignore

        return MinAtarAsterix()
    elif env_id == "minatar-breakout":
        from pgx.minatar.breakout import MinAtarBreakout  # type: ignore

        return MinAtarBreakout()
    elif env_id == "minatar-freeway":
        from pgx.minatar.freeway import MinAtarFreeway  # type: ignore

        return MinAtarFreeway()
    elif env_id == "minatar-seaquest":
        from pgx.minatar.seaquest import MinAtarSeaquest  # type: ignore

        return MinAtarSeaquest()
    elif env_id == "minatar-space_invaders":
        from pgx.minatar.space_invaders import MinAtarSpaceInvaders  # type: ignore

        return MinAtarSpaceInvaders()
    elif env_id == "othello":
        from pgx.othello import Othello

        return Othello()
    elif env_id == "pig":
        from pgx.pig import Pig

        return Pig()
    elif env_id == "shogi":
        from pgx.shogi import Shogi

        return Shogi()
    elif env_id == "shut_the_box":
        from pgx.shut_the_box import ShutTheBox

        return ShutTheBox()
    elif env_id == "sparrow_mahjong":
        from pgx.sparrow_mahjong import SparrowMahjong

        return SparrowMahjong()
    elif env_id == "tic_tac_toe":
        from pgx.tic_tac_toe import TicTacToe

        return TicTacToe()
    else:
        envs = "\n".join(available_envs())
        raise ValueError(f"Wrong env_id '{env_id}' is passed. Available ids are: \n{envs}")
