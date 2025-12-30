import numpy as np
import tensorflow as tf
import random, os
from copy import deepcopy
import threading

# Global thread lock (ensures only one thread executes sess.run at a time)
sess_lock = threading.Lock()


def fc(x, scope, nh, act=tf.nn.relu):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        # Initialize with normal distribution of smaller standard deviation
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        w = tf.get_variable("w", [nin, nh], initializer=initializer)
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
        z = tf.matmul(x, w) + b
        h = act(z)
        return h

def huber_loss(y_true, y_pred, delta=0.5):
    error = tf.abs(y_true - y_pred)
    quadratic_term = tf.minimum(error, delta)
    linear_term = error - quadratic_term
    return tf.reduce_mean(0.5 * tf.square(quadratic_term) + delta * linear_term)





class Estimator:
    def __init__(self,sess,action_dim,state_dim,n_valid_node,learning_rate,global_step,scope="estimator",summaries_dir=None, gamma=0.99, gae_lambda=0.95):
        self.sess = sess
        self.n_valid_node = n_valid_node
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.scope = scope
        self.learning_rate = learning_rate
        self.global_step = global_step
        self.T = 144
        self.clip_ratio = 0.2  # PPO clip parameter
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.summary_writer = None
        with tf.variable_scope(scope):
            # Main network
            with tf.variable_scope("value"):
                value_loss = self._build_value_model()
            # Target network
            with tf.variable_scope("target_network"):
                self.target_value_output = self._build_target_value_model()

            with tf.variable_scope("policy"):
                actor_loss, entropy = self._build_policy()
            with tf.variable_scope("target_network_policy"):
                self.target_policy_output = self._build_target_policy_model()
            # Synchronization operations
            self.update_target_ops = self._build_update_target_ops()
            self.loss = actor_loss + .5 * value_loss - 10 * entropy

        #  Summaries
        self.summaries = tf.summary.merge([
            tf.summary.scalar("value_loss", self.value_loss),
            tf.summary.scalar("value_output", tf.reduce_mean(self.value_output)),
        ])

        self.policy_summaries = tf.summary.merge([
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("adv", tf.reduce_mean(self.tfadv)),
            tf.summary.scalar("entropy", self.entropy),
        ])

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)
            self.summary_writer = tf.summary.FileWriter(summary_dir)

        self.neighbors_list = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]

    def compute_gae(self, rewards, values, next_values, gamma=None, gae_lambda=None):
        if gamma is None:
            gamma = self.gamma
        if gae_lambda is None:
            gae_lambda = self.gae_lambda

        min_len = min(len(rewards), len(values))
        rewards = np.array(rewards)[:min_len]
        values = np.array(values)[:min_len]

        if len(next_values) > 0:
            next_values = np.array(next_values)[:min_len]
            next_value = next_values[-1] if len(next_values) > 0 else 0
        else:
            next_value = 0

        advantages = []
        returns = []
        next_advantage = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * next_value - values[i]
            advantage = delta + gamma * gae_lambda * next_advantage
            return_r = advantage + values[i]

            advantages.insert(0, advantage)
            returns.insert(0, return_r)

            next_advantage = advantage
            if i > 0:
                next_value = values[i]

        return np.array(advantages), np.array(returns)
    # Build value model network
    def _build_value_model(self):
        self.state = X = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="X")
        # Define TD target value
        self.y_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="y")
        self.value_l1 = fc(X, "l1", 256, act=tf.nn.relu)
        self.value_l2 = fc(self.value_l1, "l2", 128, act=tf.nn.relu)
        self.value_l3 = fc(self.value_l2, "l3", 64, act=tf.nn.relu)
        self.value_l4 = fc(self.value_l3, "l4", 32, act=tf.nn.relu)
        # Output layer, for predicting state value
        # self.value_output = fc(l4, "value_output", 1, act=tf.nn.relu)
        self.value_output = fc(self.value_l4, "value_output", 1, act=lambda x: x)
        self.value_loss = huber_loss(self.y_pl, self.value_output, delta=0.5)
        value_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.scope + "/value"
        )
        # Gradient clipping
        value_grads = tf.gradients(self.value_loss, value_params)
        clipped_value_grads, _ = tf.clip_by_global_norm(value_grads, 1)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate * 0.5)
        # Apply gradients
        self.value_train_op = optimizer.apply_gradients(
            zip(clipped_value_grads, value_params)
        )
        return self.value_loss
    def _build_target_value_model(self):
        # Input placeholder for target network
        self.target_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="target_X")
        self.target_l1 = fc(self.target_state, "l1", 256, act=tf.nn.relu)
        self.target_l2 = fc(self.target_l1, "l2", 128, act=tf.nn.relu)
        self.target_l3 = fc(self.target_l2, "l3", 64, act=tf.nn.relu)
        self.target_l4 = fc(self.target_l3, "l4", 32, act=tf.nn.relu)
        target_value_output = fc(self.target_l4, "value_output", 1, act=lambda x: x)
        return target_value_output
    def _build_policy(self):
        self.old_probs = tf.placeholder(shape=[None, 1], dtype=tf.float32, name="old_probs")  # shape=[batch_size, 1]
        self.policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="P")
        self.ACTION = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="action")
        self.tfadv = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='advantage')
        self.neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")
        l1 = fc(self.policy_state, "l1", 256, act=tf.nn.relu)
        l2 = fc(l1, "l2", 128, act=tf.nn.relu)
        l3 = fc(l2, "l3", 32, act=tf.nn.relu)
        # Avoid valid_logits being all zeros
        # self.logits = logits = fc(l3, "logits", self.action_dim, act=tf.nn.relu)
        self.logits = logits = fc(l3, "logits", self.action_dim, act=lambda x: x)
        self.valid_logits = logits * self.neighbor_mask
        self.logsoftmaxprob = tf.nn.log_softmax(self.valid_logits)
        self.softmaxprob = tf.exp(self.logsoftmaxprob)
        # Calculate log probability of selected action
        logp = tf.reduce_sum(self.logsoftmaxprob * self.ACTION, axis=1, keepdims=True)  # shape=[batch_size, 1]
        # Calculate importance sampling ratio
        ratio = tf.exp(logp - tf.log(self.old_probs + 1e-10))  # shape=[batch_size, 1]
        # Calculate clipped loss
        surr1 = ratio * self.tfadv  # shape=[batch_size, 1]
        surr2 = tf.clip_by_value(ratio, 1.0 - 0.2, 1.0 + 0.2) * self.tfadv
        self.actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        #  # Calculate entropy and total loss
        self.entropy = -tf.reduce_mean(self.softmaxprob * self.logsoftmaxprob)
        self.policy_loss = self.actor_loss - 0.01 * self.entropy
        # policy network parameters
        policy_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,scope=self.scope + "/policy"
        )
        # Compute gradients
        gradients = tf.gradients(self.policy_loss, policy_params)
        # Gradient clipping
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # Apply gradients
        self.policy_train_op = optimizer.apply_gradients(
            zip(clipped_gradients, policy_params),
            global_step=self.global_step
        )
        return self.actor_loss, self.entropy

    def _build_target_policy_model(self):
        self.target_policy_state = tf.placeholder(shape=[None, self.state_dim], dtype=tf.float32, name="target_P")
        self.target_neighbor_mask = tf.placeholder(shape=[None, self.action_dim], dtype=tf.float32, name="neighbormask")
        target_l1 = fc(self.target_policy_state, "l1", 256, act=tf.nn.relu)
        target_l2 = fc(target_l1, "l2", 128, act=tf.nn.relu)
        target_l3 = fc(target_l2, "l3", 32, act=tf.nn.relu)
        self.target_logits = target_logits = fc(target_l3, "logits", self.action_dim, act=lambda x: x)
        self.target_valid_logits = target_logits * self.target_neighbor_mask
        self.target_logsoftmaxprob = tf.nn.log_softmax(self.target_logits)
        self.target_softmaxprob = tf.exp(self.target_logsoftmaxprob)
    def _build_update_target_ops(self):
        main_value_vars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope + "/value")]
        target_value_vars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope + "/target_network")]
        main_policy_vars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope + "/policy" )]
        target_policy_vars = [v for v in tf.trainable_variables() if v.name.startswith(self.scope + "/target_network_policy")]

        update_ops = []
        for m_var, t_var in zip(main_value_vars+main_policy_vars, target_value_vars+target_policy_vars):
            op = t_var.assign(m_var)
            update_ops.append(op)
        return update_ops


    def action(self, s, ava_node, context, epsilon):

        old_probs_list = []
        # Calculate value output of current state
        with sess_lock:
            value_output = self.sess.run(self.target_value_output, {self.target_state: s}).flatten()
        action_tuple = []# Store node identifiers corresponding to final selected actions
        valid_prob = []
        # Used for training policy gradient.
        action_choosen_mat = []
        policy_state = []
        curr_state_value = []
        next_state_ids = []
        grid_ids = [x for x in range(self.n_valid_node)]
        # Initialize valid action mask
        self.valid_action_mask = np.zeros((self.n_valid_node, self.action_dim))
        for i in range(len(ava_node)):
            for j in ava_node[i]:
                self.valid_action_mask[i][j] = 1
        curr_neighbor_mask = deepcopy(self.valid_action_mask)
        self.valid_neighbor_node_id = [[i for i in range(self.action_dim)], [i for i in range(self.action_dim)]]
        # Calculate policy probabilities
        with sess_lock:
            action_probs = self.sess.run(self.target_softmaxprob, {self.target_policy_state: s,self.target_neighbor_mask: curr_neighbor_mask})
        curr_neighbor_mask_policy = []
        # Sample actions and collect relevant information
        for idx, grid_valid_idx in enumerate(grid_ids):
            action_prob = action_probs[idx]
            # Probability used by state value function
            valid_prob.append(action_prob)
            if int(context[idx]) == 0:
                continue
            # Get valid action indices for current state
            valid_actions = np.where(curr_neighbor_mask[idx] == 1)[0]  # valid action indices
            if len(valid_actions) == 0:
                continue
            # Select action based on epsilon
            if np.random.rand() < epsilon:
                curr_action_indices_temp = np.random.choice(
                    valid_actions,
                    size=int(context[idx]),
                    replace=True
                )
            else:
                # Only use probability values corresponding to valid actions
                valid_action_prob = action_prob[valid_actions].copy()
                sum_prob = np.sum(valid_action_prob)
                if sum_prob < 1e-6:
                    valid_action_prob = np.ones_like(valid_action_prob) / len(valid_actions)
                else:
                    valid_action_prob = valid_action_prob / sum_prob
                valid_action_prob = valid_action_prob / np.sum(valid_action_prob)
                curr_action_indices_temp = np.random.choice(
                    valid_actions,
                    size=int(context[idx]),
                    p=valid_action_prob
                )
            curr_action_indices = [0] * self.action_dim
            for kk in curr_action_indices_temp:
                curr_action_indices[kk] += 1
            self.valid_neighbor_grid_id = self.valid_neighbor_node_id
            for curr_action_idx, num_driver in enumerate(curr_action_indices):
                if num_driver > 0:
                    end_node_id = int(self.valid_neighbor_node_id[grid_valid_idx][curr_action_idx])
                    action_tuple.append(end_node_id)
                    temp_a = np.zeros(self.action_dim)
                    temp_a[curr_action_idx] = 1
                    action_choosen_mat.append(temp_a)
                    policy_state.append(s[idx])
                    curr_state_value.append(value_output[idx])
                    next_state_ids.append(self.valid_neighbor_grid_id[grid_valid_idx][curr_action_idx])
                    curr_neighbor_mask_policy.append(curr_neighbor_mask[idx])
                    old_probs_list.append(action_prob[curr_action_idx])
        policy_state = np.stack(policy_state)
        return action_tuple, np.stack(valid_prob), \
            policy_state, np.stack(action_choosen_mat), curr_state_value, \
            np.stack(curr_neighbor_mask_policy), next_state_ids, \
            np.array(old_probs_list).reshape(-1, 1)

    def compute_advantage(self, curr_state_value, next_state_ids, next_state, node_reward, gamma=None):
        # Calculate advantage function
        advantage = []
        node_reward = node_reward.flatten()

        with sess_lock:
            next_state_value = self.sess.run(self.target_value_output, {self.target_state: next_state}).flatten()
        min_len = min(len(node_reward), len(curr_state_value))
        node_reward = node_reward[:min_len]
        curr_state_value = curr_state_value[:min_len]
        advantages, _ = self.compute_gae(node_reward, curr_state_value, next_state_value, gamma)
        return advantage,next_state_value

    def compute_targets(self, curr_state, node_reward, gamma=None):
        # Calculate target values
        node_reward = node_reward.flatten()
        with sess_lock:
            curr_state_value = self.sess.run(self.target_value_output, {self.target_state: curr_state}).flatten()
            next_state_value = np.roll(curr_state_value, -1)
            if len(next_state_value) > 0:
                next_state_value[-1] = 0
            else:
                next_state_value = np.array([0])


        _, returns = self.compute_gae(node_reward, curr_state_value, next_state_value, gamma)
        return returns.reshape([-1, 1]), curr_state_value




    def update_policy(self, policy_replay,global_step):
        sess = self.sess
        # Sample batch data
        (batch_s, batch_a, batch_r, batch_mask,
         batch_old_probs, batch_old_indices,
         batch_s_curr, batch_a_value, batch_targets, batch_s_next) = policy_replay.sample()

        feed_dict = {
            self.state: batch_s_curr,
            self.y_pl: batch_targets,
            self.policy_state: batch_s,
            self.tfadv: batch_r.reshape([-1, 1]),
            self.ACTION: batch_a,
            self.neighbor_mask: batch_mask,
            self.old_probs: batch_old_probs
        }
        ops_to_run = [
            self.policy_summaries,
            self.policy_train_op,
            self.value_train_op,
            self.policy_loss,
            self.value_loss,
            self.old_probs,
            self.softmaxprob,
            self.logits,
        ]
        ppo_epochs = 4
        for _ in range(ppo_epochs):
            with sess_lock:
                summaries, _,_, policy_loss,value_loss, old_probs_val, new_probs_val, logits_val = sess.run(
                    ops_to_run,
                    feed_dict=feed_dict
                )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()
        return policy_loss,value_loss

    def verify_target_network_sync(self):
        main_value_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope + "/value"
        )
        target_value_vars = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.scope + "/target_network"
        )
        # Perform synchronization operations
        with sess_lock:
            self.sess.run(self.update_target_ops)
        for main_var, target_var in zip(main_value_vars, target_value_vars):
            main_val = self.sess.run(main_var)
            target_val = self.sess.run(target_var)
            if not np.allclose(main_val, target_val):
                print(f"Synchronization failed: {main_var.name} and {target_var.name}")
                return
        print("Target network parameters synchronized successfully!")

class DoubleBuffer:
    def __init__(self, memory_size, batch_size, max_episodes=3):
        # Initialize two independent experience buffers
        self.front_buffer = policyReplayMemory(memory_size, batch_size)
        self.back_buffer = policyReplayMemory(memory_size, batch_size)
        self.current_ptr = 0  # 0: front is write buffer, back is training buffer
        self.front_buffer.max_episodes = max_episodes
        self.back_buffer.max_episodes = max_episodes
    def swap_buffers(self):
        # Atomic swap of front and back buffer pointer
        self.current_ptr = 1 - self.current_ptr
    def get_write_buffer(self):
        # Get current buffer for writing (used by main thread)
        return self.front_buffer if self.current_ptr == 0 else self.back_buffer
    def get_train_buffer(self):
        # Get current buffer for training (used by training thread)
        return self.back_buffer if self.current_ptr == 0 else self.front_buffer
class policyReplayMemory:
    def __init__(self, memory_size, batch_size):
        self.states = []
        self.neighbor_mask = []
        self.actions = []
        self.rewards = []
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.current = 0
        self.curr_lens = 0
        self.old_probs = []
        self.old_action_indices = []
        self.state_mats = []
        self.action_mats = []
        self.targets_batches = []
        self.s_grids = []
        self.episode_boundaries = []  # Store the index at the end of each episode
        self.max_episodes = 3  # Limit the number of episodes retained

    def clear_old_episodes(self):
        # Clear old experience exceeding the max_episodes limit
        if len(self.episode_boundaries) <= self.max_episodes :
            return
        # Calculate the starting index to be retained
        start_idx = self.episode_boundaries[-self.max_episodes - 1]
        # Retain the experience from the most recent max_episodes episodes
        self.states = self.states[start_idx:]
        self.actions = self.actions[start_idx:]
        self.rewards = self.rewards[start_idx:]
        self.neighbor_mask = self.neighbor_mask[start_idx:]
        self.old_probs = self.old_probs[start_idx:]
        self.old_action_indices = self.old_action_indices[start_idx:]
        self.state_mats = self.state_mats[start_idx:]
        self.action_mats = self.action_mats[start_idx:]
        self.targets_batches = self.targets_batches[start_idx:]
        self.s_grids = self.s_grids[start_idx:]
        self.curr_lens = self.states.shape[0]
        # Recalculate episode boundaries
        new_boundaries = []
        for boundary in self.episode_boundaries:
            if boundary > start_idx:
                new_boundaries.append(boundary - start_idx)
        self.episode_boundaries = new_boundaries
    def add_episode_boundary(self):
        # Record the end position of the current episode
        if self.curr_lens > 0:
            self.episode_boundaries.append(self.curr_lens)
            # Only retain the boundaries of the most recent max_episodes episodes
            if len(self.episode_boundaries) > self.max_episodes:
                self.clear_old_episodes()
                # self.episode_boundaries.pop(0)

    def get_recent_episodes_indices(self):
        # Retrieve the index range for the most recent max_episodes episodes
        if len(self.episode_boundaries) == 0:
            return np.arange(self.curr_lens)
        # Calculate the starting index (retaining the most recent max_episodes episodes)
        start_idx = 0
        if len(self.episode_boundaries) > self.max_episodes:
            start_idx = self.episode_boundaries[-self.max_episodes - 1]
        else:
            start_idx = 0

        end_idx = self.episode_boundaries[-1]
        return np.arange(start_idx, end_idx)
        # Add data to policy replay memory
    def add(self, s, a, r, mask,old_probs,action_indices,state_mat, action_mat, targets, s_grid):
        if self.curr_lens == 0:
            self.states = s
            self.actions = a
            self.rewards = r
            self.neighbor_mask = mask
            self.curr_lens = self.states.shape[0]
            self.old_probs = old_probs
            self.old_action_indices = action_indices
            self.state_mats = state_mat
            self.action_mats = action_mat
            self.targets_batches = targets
            self.s_grids = s_grid
        elif self.curr_lens <= self.memory_size:
            self.states = np.concatenate((self.states, s), axis=0)
            self.neighbor_mask = np.concatenate((self.neighbor_mask, mask), axis=0)
            self.actions = np.concatenate((self.actions, a), axis=0)
            self.rewards = np.concatenate((self.rewards, r), axis=0)
            self.curr_lens = self.states.shape[0]
            self.old_probs = np.concatenate((self.old_probs, old_probs), axis=0)
            self.old_action_indices = np.concatenate((self.old_action_indices, action_indices), axis=0)
            self.state_mats = np.concatenate((self.state_mats, state_mat), axis=0)
            self.action_mats = np.concatenate((self.action_mats, action_mat), axis=0)
            self.targets_batches = np.concatenate((self.targets_batches, targets), axis=0)
            self.s_grids = np.concatenate((self.s_grids, s_grid), axis=0)
        else:
            new_sample_lens = s.shape[0]
            index = random.randint(0, self.curr_lens - new_sample_lens)
            self.states[index:(index + new_sample_lens)] = s
            self.actions[index:(index + new_sample_lens)] = a
            self.rewards[index:(index + new_sample_lens)] = r
            self.neighbor_mask[index:(index + new_sample_lens)] = mask
            self.old_probs[index:(index + new_sample_lens)] = old_probs
            self.old_action_indices[index:(index + new_sample_lens)] = action_indices
            self.state_mats[index:(index + new_sample_lens)] = state_mat
            self.action_mats[index:(index + new_sample_lens)] = action_mat
            self.targets_batches[index:(index + new_sample_lens)] = targets
            self.s_grids[index:(index + new_sample_lens)] = s_grid
    # Sample a batch
    def sample(self):
        recent_indices = self.get_recent_episodes_indices()
        # Sampled from the latest episode
        indices = np.random.choice(
            recent_indices,
            size=min(self.batch_size, len(recent_indices)),
            replace=False
        )
        return [
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.neighbor_mask[indices],
            self.old_probs[indices],
            self.old_action_indices[indices],
            self.state_mats[indices],
            self.action_mats[indices],
            self.targets_batches[indices],
            self.s_grids[indices]
        ]

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.neighbor_mask = []
        self.curr_lens = 0
        self.old_probs = []
        self.old_action_indices = []
        self.state_mats = []
        self.action_mats = []
        self.targets_batches = []
        self.s_grids = []






