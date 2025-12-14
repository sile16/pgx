from pgx.pig import State as PigState
from pgx._src.visualizer import Config


def _make_pig_dwg(dwg, state: PigState, config: Config):
    GRID_SIZE = config["GRID_SIZE"]
    
    # Create a group for the visualization
    g = dwg.g()

    # Draw scores
    g.add(dwg.text(f"P1 Score: {state._scores[0]}", insert=(10, 30), fill="black", font_size="20px"))
    g.add(dwg.text(f"P2 Score: {state._scores[1]}", insert=(10, 60), fill="black", font_size="20px"))
    
    # Draw turn total
    g.add(dwg.text(f"Turn Total: {state._turn_total}", insert=(10, 90), fill="blue", font_size="20px"))
    
    # Draw current player indicator
    p_text = "P1's Turn" if state.current_player == 0 else "P2's Turn"
    g.add(dwg.text(p_text, insert=(10, 120), fill="red", font_size="20px", font_weight="bold"))

    return g
