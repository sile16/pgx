
import jax
import jax.numpy as jnp

from pgx.backgammon2p import Backgammon2P, _is_gammon, _calc_win_score, _is_all_off

def _make_state_with_board(env, board, dice):
    state = env.init(jax.random.PRNGKey(0))
    state = state.replace(_board=board)
    return env.set_dice(state, dice)

def test_force_play_both():
    """
    Test that if both dice can be played, the agent must play both.
    Setup:
    - Checker at 24 (Bar).
    - Dice 1, 6.
    - Entry for 1 (Index 0) is blocked.
    - Entry for 6 (Index 5) is open.
    - From 5, move using 1 (to Index 6) is open.
    - So 24->0 is illegal. 24->5 is legal.
    - Sequence 24->5->6 is legal.
    - Action should be valid only if it encodes both moves.
    """
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[24].set(1) # Bar
    board = board.at[0].set(-2) # Block entry at 0 (Die 1)
    
    dice = jnp.array([0, 5], dtype=jnp.int32) # 1 and 6
    state = _make_state_with_board(env, board, dice)
    
    # Correct action:
    # Move 6 first (24->5). Then 1 (5->6).
    # src(die=6) = 24 (Bar, idx 1).
    # src(die=1) = 5 (Point 5, idx 7).
    # indices: 0->-2, 1->24, 2->0, 3->1 ... 7->5.
    
    action_correct = 7 * 26 + 1 # src1(1)=5, src2(6)=24
    
    # Incorrect actions:
    # Only playing 6? (src1=Pass, src2=24) -> 0 * 26 + 1
    action_only_6 = 0 * 26 + 1
    
    mask = state.legal_action_mask
    
    assert mask[action_correct], "Chain move 24->5->6 should be legal"
    assert not mask[action_only_6], "Playing only 6 should be illegal (must play both)"
    
def test_prefer_higher_die():
    """
    Test that if we can play either die but not both, we must play the higher one.
    """
    env = Backgammon2P()
    
    # Setup for "can play either but not both":
    # Checker at 24 (Bar).
    # Dice 1, 6.
    # Entry 1 (0) is Open.
    # Entry 6 (5) is Open.
    # Common 2nd step destination: 6.
    # 0->6 (using 6) is BLOCKED.
    # 5->6 (using 1) is BLOCKED.
    # Block index 6.
    
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[24].set(1) # Bar
    board = board.at[6].set(-2) # Block 6
    
    dice = jnp.array([0, 5], dtype=jnp.int32) # 1 and 6
    state = _make_state_with_board(env, board, dice)
    
    # Actions:
    # Play 1 (24->0): src(1)=24, src(6)=Pass.
    # Play 6 (24->5): src(1)=Pass, src(6)=24.
    
    action_play_1 = 1 * 26 + 0 # src1=24, src2=Pass
    action_play_6 = 0 * 26 + 1 # src1=Pass, src2=24
    
    mask = state.legal_action_mask
    
    assert mask[action_play_6], "Must be able to play higher die (6)"
    assert not mask[action_play_1], "Cannot play lower die (1) if higher is possible and both impossible"

def test_bearing_off_exact():
    """
    Test bearing off exact count.
    Checker at 1. Roll 1. Move 1->Off.
    """
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    # Checker at Point 1 (index 0). 
    # Wait, backgammon board indices:
    # White moves 23 -> 0. Home board is 0..5.
    # Black moves 0 -> 23. Home board is 18..23.
    # The code uses indices 0..23.
    # _make_init_board: 
    # [2, 0, 0, 0, 0, -5, ...]
    # index 0 has +2 (White). index 5 has -5 (Black).
    # White moves > 0 ? No, standard PGX backgammon usually aligns so current player moves towards higher indices?
    # Let's check `_apply_move`:
    # tgt = src + die.
    # So moving +die means increasing index.
    # So home board must be high indices (18..23).
    # Bearing off is > 23 (26).
    # Bar is 24.
    
    # So "Point 1" for bearing off means Point 23?
    # No, Point 1 usually means 1 pip away from off.
    # If off is >23, then 23 is 1 pip away.
    # 23 + 1 = 24 (Bar?). No.
    # _tgt_from_src_die:
    # tgt_from_board = src + die.
    # If src=23, die=1. tgt=24.
    # _tgt_from_src_die handles 24 as bar?
    # "tgt_normal = jnp.where((tgt_from_board >= 0) & (tgt_from_board <= 23), tgt_from_board, 26)"
    # So 24 becomes 26 (Off).
    # So src=23 is 1 pip away.
    
    # So for "Checker at 1 pip away", we put it at 23.
    board = board.at[23].set(1)
    board = board.at[26].set(14)
    
    dice = jnp.array([0, 1], dtype=jnp.int32) # 1 and 2
    
    state = _make_state_with_board(env, board, dice)
    
    # Use 1 to bear off: src=23 (idx 25).
    # Use 2 to bear off: src=23 (idx 25).
    
    # Die 1 is idx 0. Die 2 is idx 1.
    # Action: src(die1), src(die2).
    
    action_use_2 = 0 * 26 + 25 # src1=Pass, src2=23
    action_use_1 = 25 * 26 + 0 # src1=23, src2=Pass
    
    mask = state.legal_action_mask
    
    assert mask[action_use_2], "Should use higher die to bear off"
    assert not mask[action_use_1], "Should not use lower die if higher is available?"
    
    # Wait, logic check:
    # If I use 1 (die=1), 23+1=24->Off.
    # If I use 2 (die=2), 23+2=25->Off.
    # Both bear off.
    # Remaining die:
    # If use 1, remaining is 2. Can I use 2? No checkers left.
    # If use 2, remaining is 1. Can I use 1? No checkers left.
    # State outcome is identical (15 off).
    # Does rule "prefer higher die" apply when both result in "max possible moves" (1 move)?
    # Yes, "If you can play one number but not both, you must play the higher one."
    # Here I can play 1 (and then stop).
    # Or I can play 2 (and then stop).
    # I cannot play both.
    # So I must play 2.

def test_bearing_off_chain():
    """
    Move internal to home board then bear off.
    Checker at Point 6 (18). Dice 5, 1.
    Move 18->23 (5). Then 23->Off (1).
    """
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[18].set(1) # Point 6 away (18)
    board = board.at[26].set(14) # Others off
    
    dice = jnp.array([0, 4], dtype=jnp.int32) # 1, 5
    
    state = _make_state_with_board(env, board, dice)
    
    # src(5) = 18 (idx 20). Move to 23.
    # src(1) = 23 (idx 25). Move to Off.
    
    # Action: src(die1=1)=23, src(die2=5)=18.
    action = 25 * 26 + 20 
    
    assert state.legal_action_mask[action]

def test_doubles_action_structure():
    """
    Doubles allow 4 moves.
    In backgammon2p, this is split into 2 steps of 2 moves.
    """
    env = Backgammon2P()
    board = jnp.zeros(28, dtype=jnp.int32)
    board = board.at[24].set(4) # 4 checkers at Bar
    
    dice = jnp.array([5, 5], dtype=jnp.int32) # 6, 6
    
    state = _make_state_with_board(env, board, dice)
    
    # First step: Move 2 checkers 24->18.
    # src1=24, src2=24.
    action = 1 * 26 + 1
    
    assert state.legal_action_mask[action]
    
    next_state = env.step(state, jnp.int32(action), jax.random.PRNGKey(0))
    
    # Check intermediate state
    assert not next_state.terminated
    assert next_state._remaining_actions == 1 # 1 "pair" of actions left
    assert not next_state._is_stochastic
    assert next_state.current_player == state.current_player
    
    # Second step: Move remaining 2 checkers 24->18.
    assert next_state.legal_action_mask[action]
    
    final_state = env.step(next_state, jnp.int32(action), jax.random.PRNGKey(0))
    
    # Turn should end
    assert final_state.current_player != state.current_player
