import os
import sys
import jax
import pgx
from pgx.experimental import act_randomly, auto_reset

os.makedirs("tmp", exist_ok=True)

env_id: pgx.EnvId = sys.argv[1]
color_theme = sys.argv[2]
env = pgx.make(env_id)
init = jax.jit(env.init)
step = jax.jit(auto_reset(env.step, env.init))

rng = jax.random.PRNGKey(9999)

states = []
rng, subkey = jax.random.split(rng)
state = init(subkey)
# while not state.terminated.all():
for i in range(50):
    state.save_svg(f"tmp/{env_id}_{i:03d}.svg", color_theme=color_theme)
    rng, subkey = jax.random.split(rng)
    action = act_randomly(subkey, state.legal_action_mask[None, :])[0]
    rng, subkey = jax.random.split(rng)
    state = step(state, action, subkey)
