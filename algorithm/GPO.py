

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tl
from algorithm.gcn import GraphCNN
from algorithm.gsn import GraphSNN


def invoke_model(orchestrate_agent, obs, exp):

    # Invoke the model for decision making and obtain outputs
    node_act, cluster_act, selected_node_probs, selected_cluster_probs, node_inputs, cluster_inputs,node_probs, cluster_probs = \
        orchestrate_agent.invoke_model(obs)
    # Value prediction
    node_values, service_values = orchestrate_agent.sess.run(
        [orchestrate_agent.node_value, orchestrate_agent.service_value],
        feed_dict={
            orchestrate_agent.node_inputs: node_inputs,
            orchestrate_agent.cluster_inputs: cluster_inputs
        }
    )
    # Process node action selection
    node_choice = [x for x in [node_act[0][0][0]]]
    server_choice = []
    # Process cluster action selection, adjust server indices
    selected_action = cluster_act[0][0][0]
    if selected_action >= 12:
        server_choice.append(selected_action - 11)
    else:
        server_choice.append(-(selected_action + 1))
    # Store experience
    time_step_sample = {
        'node_inputs': node_inputs,
        'cluster_inputs': cluster_inputs,
        'old_node_probs': selected_node_probs[0],
        'old_cluster_probs': selected_cluster_probs[0],
        'node_action_idx': node_choice,
        'cluster_action_idx': selected_action,
        'full_old_node_probs': node_probs[0],
        'full_old_cluster_probs': cluster_probs[0],
        'node_values': node_values.flatten()[0],
        'service_values': service_values.flatten()[0],
        'done': False
    }
    exp['samples'].append(time_step_sample)
    return node_choice, server_choice, exp


def act_offload_agent(orchestrate_agent, exp, done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state):
    obs = [done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state]
    # Invoke learning model to obtain execution nodes and updated experience dat
    node, use_exec, exp = invoke_model(orchestrate_agent, obs, exp)
    return node, use_exec, exp


def decrease_var(var, min_var, decay_rate):
    if var - decay_rate >= min_var:
        var -= decay_rate
    else:
        var = min_var
    return var



def train_orchestrate_agent(orchestrate_agent, exp, entropy_weight, entropy_weight_min, entropy_weight_decay):
    samples = exp['samples']
    batch_size = 32
    gamma = 0.995
    lam = 0.95

    all_batches = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]
        if len(batch) < batch_size:
            continue

        batch_node_inputs = np.array([s['node_inputs'] for s in batch])
        batch_node_inputs = np.squeeze(batch_node_inputs, axis=1)
        batch_cluster_inputs = np.array([s['cluster_inputs'] for s in batch])
        batch_cluster_inputs = np.squeeze(batch_cluster_inputs, axis=1)
        batch_old_node_probs = np.array([s['old_node_probs'] for s in batch])
        batch_old_cluster_probs = np.array([s['old_cluster_probs'] for s in batch])
        batch_full_old_node = np.stack([s['full_old_node_probs'] for s in batch]).reshape((-1, 12, 1))
        batch_full_old_cluster = np.stack([s['full_old_cluster_probs'] for s in batch]).reshape((-1, 24, 1))
        batch_node_rewards = np.array([s['node_reward'] for s in batch])
        batch_service_rewards = np.array([s['service_reward'] for s in batch])
        batch_node_values = np.array([s['node_values'] for s in batch])
        batch_service_values = np.array([s['service_values'] for s in batch])
        node_act_idxs = np.array([s['node_action_idx'] for s in batch]).reshape(-1, 1)
        cluster_act_idxs = np.array([s['cluster_action_idx'] for s in batch]).reshape(-1, 1)
        all_batches.append((
            batch_node_inputs, batch_cluster_inputs,
            batch_old_node_probs, batch_old_cluster_probs,
            batch_node_values,batch_service_values, batch_node_rewards,batch_service_rewards
        ))

        def calculate_gae(rewards, values, gamma=0.99, lam=0.95):
            advantages, returns = [], []
            next_advantage = 0
            next_value = 0
            for i in reversed(range(len(rewards))):
                delta = rewards[i] + gamma * next_value - values[i]
                advantage = delta + gamma * lam * next_advantage
                return_r = advantage + values[i]
                returns.insert(0, return_r)
                advantages.insert(0, advantage)
                next_advantage = advantage
                next_value = values[i]
            return np.array(advantages), np.array(returns)
        # Node advantage calculation
        node_advantages, node_returns = calculate_gae(batch_node_rewards, batch_node_values, gamma, lam)
        # Service advantage calculation
        service_advantages, service_returns = calculate_gae(batch_service_rewards, batch_service_values, gamma, lam)
    # Iterate through each batch for training
    total_loss = 0
    for epoch in range(3):
        np.random.shuffle(all_batches)
        for batchs in all_batches:
            node_inputs, cluster_inputs, old_node_probs, old_cluster_probs, batch_node_values,batch_service_values,batch_node_rewards, batch_service_rewards = batchs
            feed_dict = {
                orchestrate_agent.node_inputs: node_inputs,
                orchestrate_agent.cluster_inputs: cluster_inputs,
                orchestrate_agent.old_node_probs: old_node_probs,
                orchestrate_agent.old_cluster_probs: old_cluster_probs,
                orchestrate_agent.adv_node: node_advantages.reshape(-1, 1),
                orchestrate_agent.adv_service: service_advantages.reshape(-1, 1),
                orchestrate_agent.node_value_target: node_returns.reshape(-1, 1),
                orchestrate_agent.service_value_target: service_returns.reshape(-1, 1),
                orchestrate_agent.real_node_acts: node_act_idxs,
                orchestrate_agent.real_cluster_acts: cluster_act_idxs,
                orchestrate_agent.full_old_node_probs: batch_full_old_node,
                orchestrate_agent.full_old_cluster_probs: batch_full_old_cluster,
            }

            # Execute PPO update
            ops_to_run = [
                orchestrate_agent.actor_train_op,
                orchestrate_agent.critic_train_op,
                orchestrate_agent.grad_weights,
                orchestrate_agent.grad_bias,
                orchestrate_agent.total_loss,
                orchestrate_agent.value_loss,
                orchestrate_agent.actor_loss,
                orchestrate_agent.node_actor_loss,
                orchestrate_agent.service_actor_loss,
                orchestrate_agent.node_value_loss,
                orchestrate_agent.service_value_loss,
                orchestrate_agent.new_node_probs_selected,
                orchestrate_agent.new_cluster_probs_selected,
                orchestrate_agent.real_node_acts,
                orchestrate_agent.real_cluster_acts,
                orchestrate_agent.service_surr1,
                orchestrate_agent.service_surr2,
                orchestrate_agent.service_log_ratio,
                orchestrate_agent.service_ratio
            ]
            #  Run all operations in a single session cal
            (_, _, grad_w, grad_b,
             loss_val, value_loss, actor_loss,
             node_actor_loss, service_actor_loss,
             node_value_loss, service_value_loss,
             new_node_probs, new_cluster_probs,
             node_act_idxs, cluster_act_idxs,
             service_surr1_val, service_surr2_val,
             service_log_ratio_val, service_ratio_val) = orchestrate_agent.sess.run(
                ops_to_run,
                feed_dict=feed_dict
            )
            total_loss += loss_val

    exp['samples'] = []
    entropy_weight = decrease_var(entropy_weight, entropy_weight_min, entropy_weight_decay)
    avg_loss = total_loss / (len(all_batches) * batch_size) if all_batches else 0
    return entropy_weight, avg_loss,node_actor_loss,service_actor_loss,node_value_loss,service_value_loss


class Agent(object):
    def __init__(self):
        pass





def leaky_relu(features, alpha=0.3, name=None):
    return tf.nn.leaky_relu(features, alpha=alpha, name=name)



class NodeActor:
    def __init__(self, node_inputs, gcn_outputs, act_fn):
        with tf.variable_scope("node_actor"):
            # Use original input features and GCN outputs
            self.gcn_reshaped = tf.slice(
                tf.reshape(gcn_outputs, tf.shape(node_inputs)),
                [0, 0, 0], [-1, -1, 24]
            )
            self.merged = tf.concat([node_inputs, self.gcn_reshaped], axis=2)
            self.hid1 = tl.fully_connected(self.merged, 128, activation_fn=act_fn)
            self.hid2 = tl.fully_connected(self.hid1, 64, activation_fn=act_fn)
            self.hid3 = tl.fully_connected(self.hid2, 32, activation_fn=act_fn)
            self.logits = tl.fully_connected(self.hid3, 1, activation_fn=None)
            self.probs = tf.nn.softmax(self.logits, axis=1)

class ServiceActor:
    def __init__(self, inputs, executor_levels, act_fn):
        with tf.variable_scope("service_actor"):
            batch_size = tf.shape(inputs)[0]
            self.merged = inputs
            self.hid1 = tl.fully_connected(self.merged, 64, activation_fn=act_fn, scope="layer1")
            self.hid1 = tf.contrib.layers.layer_norm(self.hid1)
            self.hid2 = tl.fully_connected(self.hid1, 32, activation_fn=act_fn, scope="layer2")
            self.hid2 = tf.contrib.layers.layer_norm(self.hid2)
            self.hid3 = tl.fully_connected(self.hid2, 16, activation_fn=act_fn, scope="layer3")
            self.logits = tl.fully_connected(self.hid3, 24, activation_fn=None,
                                             weights_initializer=tf.orthogonal_initializer(gain=1.0),
                                             biases_initializer=tf.constant_initializer(0.05),
                                             scope="output_layer"
                                             )
            logits_reshaped = tf.reshape(self.logits, [batch_size, -1])
            self.probs = tf.nn.softmax(logits_reshaped, axis=1)
            self.probs = tf.expand_dims(self.probs, axis=-1)

def huber_loss(y_true, y_pred, delta=0.5):
    error = tf.abs(y_true - y_pred)
    quadratic_term = tf.minimum(error, delta)
    linear_term = error - quadratic_term
    return tf.reduce_mean(0.5 * tf.square(quadratic_term) + delta * linear_term)

class OrchestrateAgent(Agent):
    def __init__(self, sess, node_input_dim, cluster_input_dim, hid_dims, output_dim,
                 max_depth, executor_levels, eps=1e-6, act_fn=leaky_relu,
                 optimizer=lambda lr: tf.compat.v1.train.AdamOptimizer(learning_rate=lr,epsilon=1e-5, clipnorm=1.0 ), clip_ratio=0.2, scope='orchestrate_agent'):
        Agent.__init__(self)
        self.node_logits = None
        self.sess = sess
        self.clip_ratio = clip_ratio  # PPO clipping parameter
        self.gae_lambda = 0.95  # GAE lambda parameter
        self.target_kl = 0.1  # KL divergence threshold
        self.node_input_dim = node_input_dim
        self.cluster_input_dim = cluster_input_dim
        self.hid_dims = hid_dims
        self.output_dim = output_dim
        self.max_depth = max_depth
        self.executor_levels = executor_levels
        self.eps = eps
        self.act_fn = act_fn
        self.optimizer = optimizer
        self.scope = scope
        self.node_inputs = tf.placeholder(tf.float32, [None, None, self.node_input_dim])
        self.cluster_inputs = tf.placeholder(tf.float32,  [None, None, self.cluster_input_dim])
        self.gcn = GraphCNN(
            self.node_inputs, self.node_input_dim, self.hid_dims,
            output_dim, self.max_depth, self.act_fn, self.scope)
        self.gsn = GraphSNN(
            tf.concat([self.node_inputs, self.gcn.outputs], axis=2),
            self.node_input_dim + self.output_dim, self.hid_dims,
            output_dim, self.act_fn, self.scope)
        self.graph_global = tf.reduce_mean(self.gsn.summaries[0], axis=1, keepdims=True)
        enhanced_cluster_inputs = tf.concat(
            [self.cluster_inputs, self.graph_global],
            axis=2
        )
        # Learning rate scheduling
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        initial_lr = 1e-3
        decay_steps = 1000
        decay_rate = 0.96
        self.lr_rate = tf.compat.v1.train.exponential_decay(
            initial_lr,
            self.global_step,
            decay_steps,
            decay_rate,
            staircase=True
        )

        self.node_actor = NodeActor(
            self.node_inputs,
            self.gcn.outputs,
            self.act_fn
        )
        self.service_actor = ServiceActor(
            enhanced_cluster_inputs,
            self.executor_levels,
            self.act_fn
        )
        self.node_act_probs = self.node_actor.probs
        self.cluster_act_probs = self.service_actor.probs
        self.force_cluster_act_counter = 0
        # Selected action
        self.node_act_vec = tf.placeholder(tf.float32, [None, None])
        self.cluster_act_vec = tf.placeholder(tf.float32, [None, None, None])
        # Advantage
        self.adv_node = tf.placeholder(tf.float32, [None, 1], name="adv_node")
        self.adv_service = tf.placeholder(tf.float32, [None, 1], name="adv_service")

        self.entropy_weight = tf.placeholder(tf.float32, ())

        # Node action sampling
        self.node_acts = tf.random.categorical(
            tf.squeeze(tf.log(self.node_act_probs + 1e-8), axis=-1),
            1
        )
        self.node_acts = tf.expand_dims(self.node_acts, axis=-1)
        # Service action sampling
        self.cluster_acts = tf.random.categorical(
            tf.squeeze(tf.log(self.cluster_act_probs + 1e-8), axis=-1),
            1
        )
        self.cluster_acts = tf.expand_dims(self.cluster_acts, axis=-1)
        # Entropy calculation
        self.node_entropy = tf.reduce_sum(tf.multiply(
            self.node_act_probs, tf.log(self.node_act_probs + self.eps)))
        self.cluster_entropy = tf.reduce_sum(tf.multiply(
           self.cluster_act_probs,tf.log(self.cluster_act_probs + self.eps)))
        # Critic network
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            critic_input = tf.stop_gradient(self.gsn.summaries[0])
            self.node_value = self._build_node_critic(critic_input, self.node_act_probs)

            enhanced_critic_input = tf.concat(
                [self.cluster_inputs, self.graph_global],
                axis=2
            )
            self.service_value = self._build_service_critic(enhanced_critic_input, self.cluster_act_probs)
        # Actor parameters
        self.actor_params = (
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="node_actor") +
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="service_actor")
        )
        # Critic parameters
        self.critic_params = (
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + "/node_critic") +
                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope + "/service_critic")
        )
        # Network paramter saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)

        self.full_old_node_probs = tf.placeholder(tf.float32, [None, None, 1])
        self.full_old_cluster_probs = tf.placeholder(tf.float32, [None, None, 1])

        # PPO loss function
        self.old_node_probs = tf.placeholder(tf.float32, [None, 1], name="old_node_probs")
        self.old_cluster_probs = tf.placeholder(tf.float32, [None, 1], name="old_cluster_probs")
        self.real_node_acts = tf.placeholder(tf.int32, [None, 1])
        self.real_cluster_acts = tf.placeholder(tf.int32, [None, 1])

        # Extract new probabilities for selected actions from model output
        selected_node, selected_cluster = self.get_selected_probs(
            self.real_node_acts,
            self.real_cluster_acts
        )
        self.new_node_probs_selected = tf.squeeze(selected_node, axis=-1)
        self.new_cluster_probs_selected = tf.squeeze(selected_cluster, axis=-1)
        # Node action log probabilit
        old_node_logprob = tf.log(self.old_node_probs + 1e-8)
        new_node_logprob = tf.log(self.new_node_probs_selected + 1e-8)
        node_log_ratio = new_node_logprob - old_node_logprob
        node_ratio = tf.exp(node_log_ratio)
        # Node clipping loss
        node_surr1 = node_ratio * self.adv_node
        node_surr2 = tf.clip_by_value(node_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * self.adv_node
        self.node_actor_loss = -tf.reduce_mean(tf.minimum(node_surr1, node_surr2))
        # Service action log probability
        old_cluster_logprob = tf.log(self.old_cluster_probs + 1e-8)
        new_cluster_logprob = tf.log(self.new_cluster_probs_selected + 1e-8)
        # Service action log ratio
        service_log_ratio = new_cluster_logprob - old_cluster_logprob
        self.service_log_ratio = service_log_ratio
        service_ratio = tf.exp(service_log_ratio)
        self.service_ratio = service_ratio
        # Service clipping loss
        self.service_surr1 = service_ratio * self.adv_service
        self.service_surr2 = tf.clip_by_value(service_ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * self.adv_service
        self.service_actor_loss = -tf.reduce_mean(tf.minimum(self.service_surr1, self.service_surr2))
        self.entropy = tf.reduce_mean(self.node_entropy + self.cluster_entropy)
        self.actor_loss = self.node_actor_loss + self.service_actor_loss - 0.1 * self.entropy
        with tf.variable_scope("service_actor", reuse=True):
            # Get policy network output layer parameters
            output_weights = tf.get_variable("output_layer/weights")
            output_bias = tf.get_variable("output_layer/biases")
            # Calculate gradients
            self.grad_weights, self.grad_bias = tf.gradients(
                self.service_actor_loss,
                [output_weights, output_bias]
            )
        # Value loss (Critic)
        self.node_value_target = tf.placeholder(tf.float32, [None, 1], name="node_value_target")
        self.service_value_target = tf.placeholder(tf.float32, [None, 1], name="service_value_target")
        self.node_value_loss = huber_loss(self.node_value_target, self.node_value, delta=0.5)
        self.service_value_loss = huber_loss(self.service_value_target, self.service_value, delta=0.5)
        self.value_loss = self.node_value_loss + self.service_value_loss

        self.total_loss = self.actor_loss + 0.5 * self.value_loss - 0.1 * self.entropy
        # Orchestrate gradients
        # Actor gradients
        actor_gradients = tf.gradients(self.actor_loss, self.actor_params)
        # Critic gradients
        critic_gradients = tf.gradients(self.value_loss, self.critic_params)
        # Gradient clipping
        self.clipped_actor_grads, _ = tf.clip_by_global_norm(actor_gradients, 1.0)
        self.clipped_critic_grads, _ = tf.clip_by_global_norm(critic_gradients, 1.0)
        # Optimizers
        self.actor_optimizer = self.optimizer(self.lr_rate)
        self.critic_optimizer = self.optimizer(self.lr_rate * 0.5)
        self.actor_train_op = self.actor_optimizer.apply_gradients(
            zip(self.clipped_actor_grads, self.actor_params),
            global_step=self.global_step
        )
        self.critic_train_op = self.critic_optimizer.apply_gradients(
            zip(self.clipped_critic_grads, self.critic_params),

        )
        with tf.variable_scope(scope):
            # KL divergence monitoring
            # Calculate KL divergence between old and new policies for nodes and clusters
            node_kl = tf.reduce_sum(
                self.full_old_node_probs *
                (tf.log(self.full_old_node_probs + 1e-8) -
                 tf.log(self.node_act_probs + 1e-8)),
                axis=1
            )
            cluster_kl = tf.reduce_sum(
                self.full_old_cluster_probs *
                (tf.log(self.full_old_cluster_probs + 1e-8) -
                 tf.log(self.cluster_act_probs + 1e-8)),
                axis=1
            )
            self.approx_kl = tf.reduce_mean(node_kl + cluster_kl)

            self.sess.run(tf.global_variables_initializer())
            uninit_vars = [v for v in tf.global_variables() if v not in self.actor_params + self.critic_params]
            if uninit_vars:
                self.sess.run(tf.variables_initializer(uninit_vars))


    def get_selected_probs(self, node_idxs, cluster_idxs):

        selected_node = tf.gather(
            self.node_act_probs,
            node_idxs,
            axis=1,
            batch_dims=1
        )

        selected_cluster = tf.gather(
            self.cluster_act_probs,
            cluster_idxs,
            axis=1,
            batch_dims=1
        )
        return selected_node, selected_cluster

    def _build_node_critic(self, gsn_summary, node_act_probs):
        # Node-Critic Network
        with tf.variable_scope("node_critic", reuse=tf.AUTO_REUSE):
            node_global = tf.reduce_mean(gsn_summary, axis=1)
            node_action_feat = tl.fully_connected(
                tf.reduce_mean(node_act_probs, axis=1),
                num_outputs=24,
                activation_fn=self.act_fn
            )
            combined = tf.concat([node_global, node_action_feat], axis=1)
            x = tl.fully_connected(combined, 128, activation_fn=self.act_fn)
            x = tl.layer_norm(x)
            x = tl.fully_connected(x, 64, activation_fn=self.act_fn)
            x = tl.fully_connected(x, 32, activation_fn=self.act_fn)
            node_value = tl.fully_connected(x, 1, activation_fn=None)
        return node_value

    def _build_service_critic(self, gsn_summary, cluster_act_probs):
        # Service-Critic Network
        with tf.variable_scope("service_critic", reuse=tf.AUTO_REUSE):
            service_global = tf.reduce_max(gsn_summary, axis=1)
            service_action_feat = tl.fully_connected(
                tf.reduce_max(cluster_act_probs, axis=1),
                num_outputs=24,
                activation_fn=self.act_fn
            )
            combined = tf.concat([service_global, service_action_feat], axis=1)
            x = tl.fully_connected(combined, 256, activation_fn=self.act_fn)
            x = tl.layer_norm(x)
            x = tl.fully_connected(x, 128, activation_fn=self.act_fn)
            x = tl.fully_connected(x, 64, activation_fn=self.act_fn)
            x = tl.fully_connected(x, 32, activation_fn=self.act_fn)
            service_value = tl.fully_connected(x, 1, activation_fn=None)
        return service_value

    def predict(self, node_inputs, cluster_inputs):
        # Get raw probability distributions
        node_probs, cluster_probs, node_acts, cluster_acts = self.sess.run([
            self.node_act_probs,
            self.cluster_act_probs,
            self.node_acts,
            self.cluster_acts,
        ], feed_dict={
            self.node_inputs: node_inputs,
            self.cluster_inputs: cluster_inputs
        })
        self.force_cluster_act_counter += 1
        batch_size = node_probs.shape[0]
        selected_node_probs = np.array([
            node_probs[i][node_acts[i][0][0]]
            for i in range(batch_size)
        ])
        selected_cluster_probs = np.array([
            cluster_probs[i][cluster_acts[i][0][0]]
            for i in range(batch_size)
        ])

        return selected_node_probs, selected_cluster_probs, node_acts, cluster_acts,node_probs,cluster_probs
    def translate_state(self, obs):
        done_tasks, undone_tasks, curr_tasks_in_queue, deploy_state = obs
        done_tasks = np.array(done_tasks)
        undone_tasks = np.array(undone_tasks)
        curr_tasks_in_queue = np.array(curr_tasks_in_queue)
        deploy_state = np.array(deploy_state)
        total_num_nodes = len(curr_tasks_in_queue)
        # Inputs to feed
        node_inputs = np.zeros([total_num_nodes, self.node_input_dim])
        cluster_inputs = np.zeros([1, self.cluster_input_dim])
        for i in range(len(node_inputs)):
            node_inputs[i, :12] = curr_tasks_in_queue[i, :12]
            node_inputs[i, 12:24] = deploy_state[i, :12]
        cluster_inputs[0, :12] = done_tasks[:12]
        cluster_inputs[0, 12:] = undone_tasks[:12]
        return (np.expand_dims(node_inputs, axis=0),
                np.expand_dims(cluster_inputs, axis=0))



    def invoke_model(self, obs):
        # Invoke learning model
        node_inputs, cluster_inputs = self.translate_state(obs)
        selected_node_probs, selected_cluster_probs, node_acts, cluster_acts, node_probs, cluster_probs = self.predict(node_inputs, cluster_inputs)
        return node_acts, cluster_acts, selected_node_probs, selected_cluster_probs, node_inputs, cluster_inputs, node_probs, cluster_probs


