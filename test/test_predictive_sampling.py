import jax
import jax.numpy as jnp

from playground.architectures import MLP
from playground.envs.pendulum.pendulum_env import PendulumSwingupEnv
from playground.predictive_sampling import (
    PredictiveSampling,
    PredictiveSamplingOptions,
)


def make_optimizer():
    """Make a simple PredictiveSampling instance."""
    env = PendulumSwingupEnv()
    options = PredictiveSamplingOptions(
        episode_length=500,
        planning_horizon=20,
        num_envs=32,
        num_samples=16,
        noise_std=0.5,
        learning_rate=1e-3,
        batch_size=100,
        epochs_per_iteration=100,
        iterations=10,
    )
    policy = MLP(
        layer_sizes=(16, 16, options.planning_horizon * env.action_size)
    )
    return PredictiveSampling(env, policy, options)


def test_training_state():
    """Test creating a training state."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    rng, init_rng = jax.random.split(rng)
    training_state = ps.make_training_state(init_rng)

    U = ps.policy.apply(
        training_state.params, jnp.zeros(ps.env.observation_size)
    )
    assert U.shape == (ps.env.action_size * ps.options.planning_horizon,)

    assert training_state.opt_state is not None


def test_rollout():
    """Test rolling out an action sequence."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    jit_reset = jax.jit(ps.env.reset)
    jit_step = jax.jit(ps.env.step)

    rng, reset_rng, action_rng = jax.random.split(rng, 3)
    start_state = jit_reset(reset_rng)
    action_sequence = jax.random.normal(
        action_rng,
        (ps.options.planning_horizon, ps.env.action_size),
    )

    # Manually apply the action sequence
    manual_reward = 0.0
    state = start_state
    for action in action_sequence:
        state = jit_step(state, action)
        manual_reward += state.reward

    # Use the rollout method
    reward = ps.rollout(start_state, action_sequence)
    assert reward == manual_reward


def test_choose_action_sequence():
    """Test choosing an action sequence."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    rng, init_rng = jax.random.split(rng)
    training_state = ps.make_training_state(init_rng)
    policy_params = training_state.params

    jit_reset = jax.jit(ps.env.reset)

    rng, reset_rng, act_rng, sample_rng = jax.random.split(rng, 4)
    start_state = jit_reset(reset_rng)
    last_action_sequence = jax.random.normal(
        act_rng,
        (ps.options.planning_horizon, ps.env.action_size),
    )

    best_action_sequence = ps.choose_action_sequence(
        start_state, last_action_sequence, policy_params, sample_rng
    )
    assert best_action_sequence.shape == (
        ps.options.planning_horizon,
        ps.env.action_size,
    )
    best_reward = ps.rollout(start_state, best_action_sequence)
    other_reward = ps.rollout(start_state, jnp.zeros_like(best_action_sequence))
    assert best_reward > other_reward


def test_episode():
    """Test running an episode from a single initial state."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    training_state = ps.make_training_state(rng)
    policy_params = training_state.params

    rng, episode_rng = jax.random.split(rng)
    obs, actions = ps.episode(policy_params, episode_rng)

    assert obs.shape == (ps.options.episode_length, ps.env.observation_size)
    assert actions.shape == (
        ps.options.episode_length,
        ps.options.planning_horizon,
        ps.env.action_size,
    )
    assert jnp.allclose(obs[-1], jnp.array([-1.0, 0.0, 0.0]), atol=1e-1)


def test_regression():
    """Test doing regression on recorded policy data."""
    rng = jax.random.PRNGKey(0)
    ps = make_optimizer()
    training_state = ps.make_training_state(rng)

    # Gather some data
    print("Gathering data")
    rng, episode_rng = jax.random.split(rng)
    episode_rngs = jax.random.split(episode_rng, ps.options.num_envs)
    obs, actions = jax.vmap(ps.episode, in_axes=(None, 0))(training_state.params, episode_rngs)
    obs = obs.reshape((-1, ps.env.observation_size))
    actions = actions.reshape((-1, ps.options.planning_horizon, ps.env.action_size)) 

    # See how well the old policy fits the data
    act_pred = ps.policy.apply(training_state.params, obs).reshape(
        (
            ps.options.episode_length * ps.options.num_envs,
            ps.options.planning_horizon,
            ps.env.action_size,
        )
    )
    old_error = jnp.mean(jnp.square(act_pred - actions))

    # Fit the policy to the data
    print("Fitting the policy to the data")
    rng, regress_rng = jax.random.split(rng)
    new_training_state = ps.regress_policy(
        training_state, obs, actions, regress_rng
    )

    # Make sure the new policy fits the data better
    act_pred = ps.policy.apply(new_training_state.params, obs).reshape(
        (
            ps.options.episode_length * ps.options.num_envs,
            ps.options.planning_horizon,
            ps.env.action_size,
        )
    )
    new_error = jnp.mean(jnp.square(act_pred - actions))
    assert new_error < old_error

    # Roll out the fitted policy
    print("Rolling out the fitted policy")
    jit_reset = jax.jit(ps.env.reset)
    jit_step = jax.jit(ps.env.step)
    jit_policy = jax.jit(lambda obs: ps.policy.apply(new_training_state.params, obs).reshape(ps.options.planning_horizon, ps.env.action_size)[0])

    rng, rollout_rng = jax.random.split(rng)
    state = jit_reset(rollout_rng)
    state = state.tree_replace({"pipeline_state.qpos": jax.numpy.array([0.0]),
                                "pipeline_state.qvel": jax.numpy.array([0.0])})
    observations = [state.obs]
    for t in range(ps.options.episode_length):
        action = jit_policy(state.obs)
        state = jit_step(state, action)
        observations.append(state.obs)
    observations = jnp.stack(observations)
    theta = jnp.arctan2(observations[:, 1], observations[:, 0])
    theta_dot = observations[:, 2]

    print(observations[-1])

    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    plt.plot(theta)
    plt.subplot(2,1,2)
    plt.plot(theta_dot)
    plt.show()




if __name__ == "__main__":
    # test_training_state()
    # test_rollout()
    # test_choose_action_sequence()
    # test_episode()
    test_regression()
