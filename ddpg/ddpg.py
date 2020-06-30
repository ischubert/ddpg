"""
Class for RL with DDPG
"""
import tensorflow as tf
import numpy as np


class DDPGBase():
    """
    Base class for RL with DDPG
    """

    def __init__(
            self,
            actor,
            critic,
            gamma=0.9,
            tau=0.9,
            critic_epochs=10,
            actor_epochs=10,
            critic_batch_size=32,
            actor_batch_size=32,
    ):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.tau = tau
        self.critic_epochs = critic_epochs
        self.actor_epochs = actor_epochs
        self.critic_batch_size = critic_batch_size
        self.actor_batch_size = actor_batch_size

    def train(
            self,
            states,
            actions,
            goals,
            rewards,
            next_states
    ):
        """
        Train both actor and critic based on data from the buffer
        """
        # TODO: Finish implementation, take special care of the target network stuff which I ignored so far
        critic_targets = rewards + self.gamma*self.critic([
            next_states,
            # next actions
            self.actor([
                next_states,
                goals
            ]),
            goals
        ])

        # update critic
        self.critic.fit(
            # X
            [
                states,
                actions,
                goals
            ],
            # Y
            critic_targets,
            epochs=self.critic_epochs,
            batch_size=self.critic_batch_size
        )

        # update actor
        for _ in range(self.actor_epochs):

            indices = np.random.permutation(len(states))
            batches = indices[
                :self.actor_batch_size*(len(states)//self.actor_batch_size)
            ].reshape(-1, self.actor_batch_size)

            for batch in batches:

                with tf.GradientTape() as tape:
                    actor_actions = self.actor([
                        states[batch],
                        goals[batch]
                    ])
                    critic_value = self.critic([
                        states[batch],
                        actor_actions,
                        goals[batch]
                    ])
                    actor_loss = -tf.math.reduce_mean(critic_value)

                actor_grad = tape.gradient(
                    actor_loss, self.actor.trainable_variables)
                self.actor.optimizer.apply_gradients(
                    zip(actor_grad, self.actor.trainable_variables)
                )
