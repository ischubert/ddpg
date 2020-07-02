# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras
import ddpg
import matplotlib.pyplot as plt

# %%
# example critic
state_in = keras.Input(shape=2)
action_in = keras.Input(shape=2)
goal_in = keras.Input(shape=2)

fx = keras.layers.Concatenate(axis=-1)([
    state_in,
    action_in,
    goal_in
])

fx = keras.layers.Dense(50, activation='relu')(fx)
fx = keras.layers.Dense(50, activation='relu')(fx)
fx = keras.layers.Dense(50, activation='relu')(fx)
fx = keras.layers.Dense(50, activation='relu')(fx)
fx = keras.layers.Dense(50, activation='relu')(fx)
fx = keras.layers.Dense(50, activation='relu')(fx)
fx = keras.layers.Dense(30, activation='relu')(fx)
fx = keras.layers.Dense(20, activation='relu')(fx)
fx = keras.layers.Dense(10, activation='relu')(fx)
fx = keras.layers.Dense(1, activation='linear')(fx)

critic = keras.Model(
    inputs=[state_in, action_in, goal_in],
    outputs=fx
)
critic.compile(
    optimizer='Adam',
    loss='mse'
)

# example actor
state_in = keras.Input(shape=2)
goal_in = keras.Input(shape=2)

fx = keras.layers.Concatenate(axis=-1)([
    state_in,
    goal_in
])

fx = keras.layers.Dense(20, activation='relu')(fx)
fx = keras.layers.Dense(20, activation='relu')(fx)
fx = keras.layers.Dense(20, activation='relu')(fx)
fx = keras.layers.Dense(20, activation='relu')(fx)
fx = keras.layers.Dense(20, activation='relu')(fx)
fx = keras.layers.Dense(10, activation='relu')(fx)
fx = keras.layers.Dense(2, activation='linear')(fx)
fx = tf.keras.backend.l2_normalize(
    fx, axis=-1
)

actor = keras.Model(
    inputs=[state_in, goal_in],
    outputs=fx
)
# we don't compile the model, but only assign it an optimizer
actor.optimizer = tf.keras.optimizers.Adam(lr=0.01)

# %%
agent = ddpg.ddpg.DDPGBase(
    actor,
    critic,
    gamma=0,
    actor_epochs=100
)

# %%
goal = np.random.rand(2)

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

for iteration in range(1000):

    state = np.random.rand(2)

    action = 2*np.random.rand(2)-1
    action = action / np.linalg.norm(action)

    next_state = state + action/20

    reward = 2- np.linalg.norm(
        goal - next_state
    )

    states.append(state)
    actions.append(action)
    goals.append(goal)
    rewards.append(reward)
    next_states.append(next_state)

    if iteration % 10 == 1:
        agent.train(
            np.array(states),
            np.array(actions),
            np.array(goals),
            np.array(rewards),
            np.array(next_states)
        )

    if iteration % 10 == 1:

        returns = []

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
                # actor only outputs normalized actions
                test_state += action.reshape(-1)/20

                test_rewards.append(
                    2- np.linalg.norm(
                        goal - test_state
                    )
                )

            returns.append(
                np.sum(
                    np.array(test_rewards) * (
                        agent.gamma ** np.arange(len(test_rewards))
                    )
                )
            )
            plt.plot(
                np.array(traj)[:, 0],
                np.array(traj)[:, 1]
            )
            plt.scatter(
                [traj[0][0]],
                [traj[0][1]]
            )
        plt.scatter(
            [goal[0]], [goal[1]], marker='*'
        )
        plt.show()

        performance_trajectory.append(
            np.mean(returns)
        )

        q_vals = np.array(agent.critic([
            state_grid,
            np.repeat(
                np.array([1.,0.])[None,:],
                len(state_grid),
                axis=0
            ),
            np.repeat(
                goal[None,:],
                len(state_grid),
                axis=0
            )
        ]))
        plt.imshow(q_vals.reshape(
            n_state_grid,
            n_state_grid
        ))
        plt.colorbar()
        plt.show()

        plt.imshow(
            2-np.linalg.norm(
                goal[None,:] - state_grid,
                axis=-1
            ).reshape(
                n_state_grid,
                n_state_grid
            )
        )
        plt.colorbar()
        plt.show()

        plt.plot(performance_trajectory)
        plt.show()

# %%
