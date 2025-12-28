"""
Tests to verify that optimized variants produce identical results to originals.

These tests ensure that performance optimizations don't change game behavior.
"""

import jax
import jax.numpy as jnp
import pytest


# =============================================================================
# 2048 Variant Tests
# =============================================================================

class Test2048Variants:
    """Test that all 2048 variants produce identical results."""

    @pytest.fixture
    def envs_2048(self):
        """Load all 2048 environment variants."""
        from pgx.play2048 import Play2048
        from pgx.play2048_v2_branchless import Play2048V2Branchless
        from pgx.play2048_v2_no_rotate import Play2048V2NoRotate
        from pgx.play2048_v2_all import Play2048V2All

        return {
            "original": Play2048(),
            "branchless": Play2048V2Branchless(),
            "no_rotate": Play2048V2NoRotate(),
            "all": Play2048V2All(),
        }

    def test_init_identical(self, envs_2048):
        """Test that initialization produces identical boards."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed)
            states = {name: env.init(key) for name, env in envs_2048.items()}

            baseline = states["original"]._board
            for name, state in states.items():
                assert jnp.allclose(baseline, state._board), \
                    f"2048 {name} init differs from original at seed {seed}"

    def test_step_deterministic_identical(self, envs_2048):
        """Test that deterministic steps produce identical boards."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed)
            states = {name: env.init(key) for name, env in envs_2048.items()}

            # Take a few deterministic steps
            for _ in range(3):
                if states["original"].terminated:
                    break

                # Get a legal action from original
                legal = states["original"].legal_action_mask
                action = jnp.argmax(legal)

                # Apply to all variants
                new_states = {}
                for name, env in envs_2048.items():
                    new_states[name] = env.step_deterministic(states[name], action)

                baseline = new_states["original"]._board
                for name, state in new_states.items():
                    assert jnp.allclose(baseline, state._board), \
                        f"2048 {name} step_deterministic differs from original"

                states = new_states

    def test_legal_action_mask_identical(self, envs_2048):
        """Test that legal action masks are identical."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed)
            states = {name: env.init(key) for name, env in envs_2048.items()}

            baseline = states["original"].legal_action_mask
            for name, state in states.items():
                assert jnp.allclose(baseline, state.legal_action_mask), \
                    f"2048 {name} legal_action_mask differs from original at seed {seed}"

    def test_slide_and_merge_branchless(self):
        """Test branchless slide_and_merge against original."""
        from pgx.play2048 import _slide_and_merge as original
        from pgx.play2048_v2_branchless import _slide_and_merge_branchless as branchless

        test_cases = [
            jnp.int32([0, 2, 0, 2]),  # merge
            jnp.int32([2, 2, 2, 2]),  # double merge
            jnp.int32([0, 0, 0, 2]),  # slide only
            jnp.int32([4, 0, 2, 0]),  # slide no merge
            jnp.int32([1, 2, 3, 4]),  # no change
            jnp.int32([0, 0, 0, 0]),  # empty
        ]

        for tc in test_cases:
            orig_result, orig_reward = original(tc)
            branch_result, branch_reward = branchless(tc)
            assert jnp.allclose(orig_result, branch_result), \
                f"Branchless slide_and_merge differs for {tc.tolist()}"
            assert jnp.allclose(orig_reward, branch_reward), \
                f"Branchless reward differs for {tc.tolist()}"


# =============================================================================
# Backgammon Variant Tests
# =============================================================================

class TestBackgammonVariants:
    """Test that all Backgammon variants produce identical results."""

    @pytest.fixture
    def envs_bg(self):
        """Load all Backgammon environment variants."""
        from pgx.backgammon import Backgammon
        from pgx.backgammon_v2_fast_obs import BackgammonV2FastObs
        from pgx.backgammon_v2_branchless import BackgammonV2Branchless
        from pgx.backgammon_v2_all import BackgammonV2All

        return {
            "original": Backgammon(short_game=True),
            "fast_obs": BackgammonV2FastObs(short_game=True),
            "branchless": BackgammonV2Branchless(short_game=True),
            "all": BackgammonV2All(short_game=True),
        }

    def test_init_identical(self, envs_bg):
        """Test that initialization produces identical boards."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed + 100)
            states = {name: env.init(key) for name, env in envs_bg.items()}

            baseline = states["original"]._board
            for name, state in states.items():
                assert jnp.allclose(baseline, state._board), \
                    f"Backgammon {name} init differs from original at seed {seed}"

    def test_step_stochastic_identical(self, envs_bg):
        """Test that stochastic steps (dice rolls) produce identical boards."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed + 100)
            states = {name: env.init(key) for name, env in envs_bg.items()}

            # Apply same dice roll
            dice_action = jnp.int32(10)  # Some dice combination
            new_states = {}
            for name, env in envs_bg.items():
                if states[name]._is_stochastic:
                    new_states[name] = env.step_stochastic(states[name], dice_action)
                else:
                    new_states[name] = states[name]

            baseline = new_states["original"]._board
            for name, state in new_states.items():
                assert jnp.allclose(baseline, state._board), \
                    f"Backgammon {name} step_stochastic differs from original"

    def test_legal_action_mask_identical(self, envs_bg):
        """Test that legal action masks are identical after dice roll."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed + 100)
            states = {name: env.init(key) for name, env in envs_bg.items()}

            # Apply dice roll to get to decision state
            dice_action = jnp.int32(7)  # Non-doubles
            for name, env in envs_bg.items():
                if states[name]._is_stochastic:
                    states[name] = env.step_stochastic(states[name], dice_action)

            baseline = states["original"].legal_action_mask
            for name, state in states.items():
                assert jnp.allclose(baseline, state.legal_action_mask), \
                    f"Backgammon {name} legal_action_mask differs from original"

    def test_step_deterministic_identical(self, envs_bg):
        """Test that deterministic steps produce identical boards."""
        for seed in range(5):
            key = jax.random.PRNGKey(seed + 100)
            states = {name: env.init(key) for name, env in envs_bg.items()}

            # Apply dice roll
            dice_action = jnp.int32(7)
            for name, env in envs_bg.items():
                if states[name]._is_stochastic:
                    states[name] = env.step_stochastic(states[name], dice_action)

            # Get a legal action
            legal = states["original"].legal_action_mask
            if not legal.any():
                continue

            action = jnp.int32(jnp.argmax(legal))

            # Apply deterministic step
            new_states = {}
            for name, env in envs_bg.items():
                new_states[name] = env.step_deterministic(states[name], action)

            baseline = new_states["original"]._board
            for name, state in new_states.items():
                assert jnp.allclose(baseline, state._board), \
                    f"Backgammon {name} step_deterministic differs from original"

    def test_branchless_is_action_legal(self):
        """Test branchless action legality check against original."""
        from pgx.backgammon import Backgammon, _is_action_legal as original_check
        from pgx.backgammon_v2_branchless import _is_action_legal_branchless

        env = Backgammon(short_game=True)

        for seed in range(5):
            key = jax.random.PRNGKey(seed + 200)
            state = env.init(key)

            # Apply dice roll
            if state._is_stochastic:
                state = env.step_stochastic(state, jnp.int32(10))

            board = state._board

            # Check all actions
            for action in range(26 * 6):
                orig = original_check(board, jnp.int32(action))
                branchless = _is_action_legal_branchless(board, jnp.int32(action))
                assert bool(orig) == bool(branchless), \
                    f"Branchless legality check differs for action {action}"


# =============================================================================
# Full Game Simulation Tests
# =============================================================================

class TestFullGameSimulation:
    """Test that full games produce identical final states."""

    def test_2048_full_game(self):
        """Run a full 2048 game and verify all variants match."""
        from pgx.play2048 import Play2048
        from pgx.play2048_v2_all import Play2048V2All

        env_orig = Play2048()
        env_opt = Play2048V2All()

        key = jax.random.PRNGKey(42)
        state_orig = env_orig.init(key)
        state_opt = env_opt.init(key)

        max_steps = 50
        for step in range(max_steps):
            if state_orig.terminated:
                assert state_opt.terminated, "Termination mismatch"
                break

            assert jnp.allclose(state_orig._board, state_opt._board), \
                f"Board mismatch at step {step}"

            key, subkey = jax.random.split(key)
            logits = jnp.where(state_orig.legal_action_mask, 0.0, -1e9)
            action = jax.random.categorical(subkey, logits=logits)

            key, k1 = jax.random.split(key)
            state_orig = env_orig.step(state_orig, action, k1)
            state_opt = env_opt.step(state_opt, action, k1)

    def test_backgammon_full_game(self):
        """Run a full backgammon game and verify all variants match."""
        from pgx.backgammon import Backgammon
        from pgx.backgammon_v2_all import BackgammonV2All

        env_orig = Backgammon(short_game=True)
        env_opt = BackgammonV2All(short_game=True)

        key = jax.random.PRNGKey(42)
        state_orig = env_orig.init(key)
        state_opt = env_opt.init(key)

        max_steps = 50
        for step in range(max_steps):
            if state_orig.terminated:
                assert state_opt.terminated, "Termination mismatch"
                break

            assert jnp.allclose(state_orig._board, state_opt._board), \
                f"Board mismatch at step {step}"

            if state_orig._is_stochastic:
                key, subkey = jax.random.split(key)
                dice_action = jax.random.randint(subkey, shape=(), minval=0, maxval=21)
                state_orig = env_orig.step_stochastic(state_orig, dice_action)
                state_opt = env_opt.step_stochastic(state_opt, dice_action)
            else:
                key, subkey = jax.random.split(key)
                logits = jnp.where(state_orig.legal_action_mask, 0.0, -1e9)
                action = jax.random.categorical(subkey, logits=logits)

                key, k1 = jax.random.split(key)
                state_orig = env_orig.step(state_orig, action, k1)
                state_opt = env_opt.step(state_opt, action, k1)


# =============================================================================
# Backgammon Win Score Tests (Single, Gammon, Backgammon)
# =============================================================================

class TestBackgammonWinScores:
    """Test that all Backgammon variants compute identical win scores."""

    @pytest.fixture
    def envs_bg(self):
        """Load all Backgammon environment variants."""
        from pgx.backgammon import Backgammon
        from pgx.backgammon_v2_fast_obs import BackgammonV2FastObs
        from pgx.backgammon_v2_branchless import BackgammonV2Branchless
        from pgx.backgammon_v2_all import BackgammonV2All

        return {
            "original": Backgammon(short_game=True),
            "fast_obs": BackgammonV2FastObs(short_game=True),
            "branchless": BackgammonV2Branchless(short_game=True),
            "all": BackgammonV2All(short_game=True),
        }

    def test_calc_win_score_identical(self):
        """Test that _calc_win_score is identical across variants."""
        from pgx.backgammon import _calc_win_score as original_calc
        from pgx.backgammon_v2_all import _calc_win_score as opt_calc

        # Backgammon win (3 points): opponent has pieces on bar or winner's home
        backgammon_board = jnp.zeros(28, dtype=jnp.int32)
        backgammon_board = backgammon_board.at[26].set(15)  # Black won (all off)
        backgammon_board = backgammon_board.at[23].set(-15)  # White in Black's home

        orig_score = original_calc(backgammon_board)
        opt_score = opt_calc(backgammon_board)
        assert orig_score == 3, f"Original should be 3 (backgammon), got {orig_score}"
        assert opt_score == orig_score, f"Optimized score {opt_score} != original {orig_score}"

        # Gammon win (2 points): opponent bore off 0 pieces, not on bar/home
        gammon_board = jnp.zeros(28, dtype=jnp.int32)
        gammon_board = gammon_board.at[26].set(15)  # Black won
        gammon_board = gammon_board.at[7].set(-15)  # White in mid-board

        orig_score = original_calc(gammon_board)
        opt_score = opt_calc(gammon_board)
        assert orig_score == 2, f"Original should be 2 (gammon), got {orig_score}"
        assert opt_score == orig_score, f"Optimized score {opt_score} != original {orig_score}"

        # Single win (1 point): opponent bore off at least 1 piece
        single_board = jnp.zeros(28, dtype=jnp.int32)
        single_board = single_board.at[26].set(15)  # Black won
        single_board = single_board.at[27].set(-3)  # White bore off 3
        single_board = single_board.at[3].set(-12)  # Rest still on board

        orig_score = original_calc(single_board)
        opt_score = opt_calc(single_board)
        assert orig_score == 1, f"Original should be 1 (single), got {orig_score}"
        assert opt_score == orig_score, f"Optimized score {opt_score} != original {orig_score}"

    def test_rewards_identical_short_game(self, envs_bg):
        """Test that rewards are identical during short game play (50 steps)."""
        # Play 50 steps and compare intermediate rewards
        key = jax.random.PRNGKey(42)
        states = {name: env.init(key) for name, env in envs_bg.items()}

        for step in range(50):
            if states["original"].terminated:
                break

            if states["original"]._is_stochastic:
                key, subkey = jax.random.split(key)
                dice_action = jax.random.randint(subkey, shape=(), minval=0, maxval=21)
                states = {name: env.step_stochastic(state, dice_action)
                          for (name, env), state in zip(envs_bg.items(), states.values())}
            else:
                key, subkey = jax.random.split(key)
                legal = states["original"].legal_action_mask
                logits = jnp.where(legal, 0.0, -1e9)
                action = jax.random.categorical(subkey, logits)

                key, k1 = jax.random.split(key)
                states = {name: env.step(state, action, k1)
                          for (name, env), state in zip(envs_bg.items(), states.values())}

            # Check rewards match at each step
            baseline = states["original"]
            for name, state in states.items():
                assert jnp.allclose(baseline.rewards, state.rewards), \
                    f"Step {step}: {name} rewards {state.rewards} != original {baseline.rewards}"

    def test_board_state_identical_during_play(self, envs_bg):
        """Test that board states remain identical during gameplay."""
        key = jax.random.PRNGKey(123)
        states = {name: env.init(key) for name, env in envs_bg.items()}

        for step in range(30):
            if states["original"].terminated:
                break

            # Check boards match
            baseline_board = states["original"]._board
            for name, state in states.items():
                assert jnp.allclose(baseline_board, state._board), \
                    f"Step {step}: {name} board differs from original"

            if states["original"]._is_stochastic:
                key, subkey = jax.random.split(key)
                dice_action = jax.random.randint(subkey, shape=(), minval=0, maxval=21)
                states = {name: env.step_stochastic(state, dice_action)
                          for (name, env), state in zip(envs_bg.items(), states.values())}
            else:
                key, subkey = jax.random.split(key)
                legal = states["original"].legal_action_mask
                logits = jnp.where(legal, 0.0, -1e9)
                action = jax.random.categorical(subkey, logits)

                key, k1 = jax.random.split(key)
                states = {name: env.step(state, action, k1)
                          for (name, env), state in zip(envs_bg.items(), states.values())}
