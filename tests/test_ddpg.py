# %%
"""
Tests for the DDPG module
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import ddpg


def run_ddpg_gridworld(n_iters=1002, show=False):
    """
    run an experiment of the DDPG agent in a grid world env
    """
    # define critic and actor networks
    # example critic
    state_in = keras.Input(shape=2)
    action_in = keras.Input(shape=2)
    goal_in = keras.Input(shape=2)
    f_x = keras.layers.Concatenate(axis=-1)([
        state_in,
        action_in,
        goal_in
    ])
    f_x = keras.layers.Dense(50, activation='relu')(f_x)
    f_x = keras.layers.Dense(50, activation='relu')(f_x)
    f_x = keras.layers.Dense(50, activation='relu')(f_x)
    f_x = keras.layers.Dense(50, activation='relu')(f_x)
    f_x = keras.layers.Dense(50, activation='relu')(f_x)
    f_x = keras.layers.Dense(50, activation='relu')(f_x)
    f_x = keras.layers.Dense(30, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(10, activation='relu')(f_x)
    f_x = keras.layers.Dense(1, activation='linear')(f_x)
    critic = keras.Model(
        inputs=[state_in, action_in, goal_in],
        outputs=f_x
    )
    critic.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.01),
        loss='mse'
    )

    # example actor
    state_in = keras.Input(shape=2)
    goal_in = keras.Input(shape=2)
    f_x = keras.layers.Concatenate(axis=-1)([
        state_in,
        goal_in
    ])
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(20, activation='relu')(f_x)
    f_x = keras.layers.Dense(10, activation='relu')(f_x)
    f_x = keras.layers.Dense(2, activation='linear')(f_x)
    f_x = tf.keras.backend.l2_normalize(
        f_x, axis=-1
    )
    actor = keras.Model(
        inputs=[state_in, goal_in],
        outputs=f_x
    )
    actor.compile(
        tf.keras.optimizers.Adam(lr=0.002),
        loss='mse'
    )

    agent = ddpg.ddpg.DDPGBase(
        actor,
        critic,
        critic_epochs=1,
        actor_epochs=1,
        gamma=0.9,
        tau=0.4
    )

    # define gridworld env
    goal = np.array([0.5, 0.5])

    def calculate_reward(next_states, goal):
        next_states = next_states.reshape(-1, 2)

        return (np.linalg.norm(
            goal[None, :] - next_states,
            axis=-1
        ).reshape(-1) < 0.2).astype(np.float)

    def calculate_next_state(state, action):
        return np.clip(
            state + action/40,
            0, 1
        ).reshape(-1)

    # training loop
    states = []
    actions = []
    goals = []
    rewards = []
    next_states = []
    performance_trajectory = []

    n_state_grid = 50
    state_grid = np.stack(np.meshgrid(
        np.linspace(0, 1, n_state_grid),
        np.linspace(0, 1, n_state_grid)
    ))[[1, 0]].reshape(2, n_state_grid*n_state_grid).T

    for iteration in range(n_iters):
        if len(states) > 1000:
            states = states[-1000:]
            actions = actions[-1000:]
            goals = goals[-1000:]
            rewards = rewards[-1000:]
            next_states = next_states[-1000:]

        if iteration % 50 == 1:
            print('Iteration {}/{}'.format(iteration, n_iters))

        state = np.random.rand(2)

        action = 2*np.random.rand(2)-1
        action = action / np.linalg.norm(action)

        next_state = calculate_next_state(
            state, action
        )

        reward = calculate_reward(
            next_state,
            goal
        )[0]

        states.append(state)
        actions.append(action)
        goals.append(goal)
        rewards.append(reward)
        next_states.append(next_state)

        if iteration % 100 == 1:
            for _ in range(50):
                agent.train(
                    np.array(states),
                    np.array(actions),
                    np.array(goals),
                    np.array(rewards),
                    np.array(next_states)
                )

        if iteration % 200 == 1:
            returns = []
            if show:
                plt.figure(figsize=(7, 7))
            for _ in range(20):
                test_state = np.random.rand(2)
                test_rewards = []

                traj = []
                for __ in range(50):
                    traj.append(test_state.copy())
                    action = np.array(agent.actor([
                        test_state.reshape((1, 2)),
                        goal.reshape((1, 2))
                    ]))

                    test_state = calculate_next_state(
                        test_state, action.reshape(-1)
                    )

                    test_rewards.append(
                        calculate_reward(
                            test_state,
                            goal
                        )[0]
                    )

                returns.append(
                    np.sum(
                        np.array(test_rewards) * (
                            agent.gamma ** np.arange(len(test_rewards))
                        )
                    )
                )
                if show:
                    plt.plot(
                        np.array(traj)[:, 0],
                        np.array(traj)[:, 1]
                    )
                    plt.scatter(
                        [traj[0][0]],
                        [traj[0][1]]
                    )
            if show:
                plt.scatter(
                    [goal[0]], [goal[1]], marker='*'
                )
                plt.show()

            performance_trajectory.append(
                np.mean(returns)
            )

            if show:
                goals_to_plot = np.array([
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1]
                ], dtype=float)
                q_vals = np.array([
                    np.array(agent.critic([
                        state_grid,
                        np.repeat(
                            goal[None, :],
                            len(state_grid),
                            axis=0
                        ),
                        np.repeat(
                            goal[None, :],
                            len(state_grid),
                            axis=0
                        )
                    ]))
                    for goal in goals_to_plot
                ])

                plt.imshow(
                    q_vals[0].reshape(
                        n_state_grid,
                        n_state_grid
                    )
                )
                plt.colorbar()
                plt.show()

                winner = np.argmax(
                    q_vals,
                    axis=0
                )
                winning_dirs = np.squeeze(goals_to_plot[winner])
                # breakpoint()

                plt.figure(figsize=(7, 7))
                plt.quiver(
                    state_grid[:, 0],
                    state_grid[:, 1],
                    winning_dirs[:, 0],
                    winning_dirs[:, 1]
                )
                plt.show()

                plt.imshow(
                    calculate_reward(
                        np.array(state_grid),
                        goal
                    ).reshape(
                        n_state_grid,
                        n_state_grid
                    )
                )
                plt.colorbar()
                plt.show()

                plt.plot(performance_trajectory)
                plt.show()
    return performance_trajectory


def test_ddpg_gridworld():
    """
    Test the DDPG's performance in a grid world environment
    """
    performances = [
        run_ddpg_gridworld()[-1]
        for _ in range(20)
    ]
    # assert that at least half of the agents are successful
    # (i.e. return > 4)after 1000 iterations
    assert np.mean(np.array(performances) > 4) > 0.4


def test_tape_gradient(show=False):
    """
    Test sign of apply_gradients
    """
    last_mean_vals = []
    for _ in range(20):
        model = keras.models.Sequential([
            keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            keras.layers.Dense(10, activation='relu'),
            keras.layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(lr=0.0001),
            loss='mse'
        )

        mean_vals = []
        for _ in range(1000):
            with tf.GradientTape() as tape:
                y_vals = model(
                    np.random.rand(1000, 5)
                )
                loss = -tf.math.reduce_mean(y_vals)

                grad = tape.gradient(
                    loss, model.trainable_variables
                )

                model.optimizer.apply_gradients(
                    zip(grad, model.trainable_variables)
                )
            mean_vals.append(
                np.mean(
                    model(
                        np.random.rand(100, 5)
                    )
                )
            )
        if show:
            plt.plot(mean_vals)
        last_mean_vals.append(
            mean_vals[-1]
        )
    assert np.mean(last_mean_vals) > 2

# %%
