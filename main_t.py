
from collections import defaultdict
from algorithm.MAPPO import *
from algorithm.GPO import *
from env.platform import *
from env.run import *
from gRPC.generated import cluster_pb2_grpc
from gRPC.generated import cluster_pb2
from concurrent import futures
import grpc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time


def subtract_lists(list1, list2):
    return [a - b for a, b in zip(list1, list2)]
def flatten(list):
    return [y for x in list for y in x]

def get_task_backlog(master1, master2,MAX_TASK_TYPE):
    backlog = []
    all_nodes = master1.node_list + master2.node_list
    for node in all_nodes:
        # Initialize task type counter
        task_counts = [0] * MAX_TASK_TYPE
        # Count task types in the node's task queue
        for task in node.task_queue:
            task_counts[task[0]] += 1
        backlog.append(task_counts)
    return backlog

def calculate_task_match_reward(task_backlog_before,change_service, deploy_state_before, deploy_state_after, done_tasks, undone_tasks):

    max_task_type = len(task_backlog_before[0])
    task_type_weights = defaultdict(float)
    historical_weight = 0.6
    current_weight = 0.4
    #  Calculate historical task volume
    historical_total = [0] * max_task_type
    for node_done_tasks, node_undone_tasks in zip(done_tasks, undone_tasks):
        for task_type in range(max_task_type):
            historical_total[task_type] += node_done_tasks[task_type] + node_undone_tasks[task_type]
    for task_type in range(max_task_type):
        task_type_weights[task_type] = historical_total[task_type] * historical_weight
    # Calculate current queue backlog
    current_backlog = [0] * max_task_type
    for node_backlog in task_backlog_before:
        for task_type, count in enumerate(node_backlog):
            current_backlog[task_type] += count
    for task_type in range(max_task_type):
        task_type_weights[task_type] += current_backlog[task_type] * current_weight

    #  Determine high-demand task type
    max_demand_type = max(task_type_weights, key=lambda k: task_type_weights[k], default=-1)
    if max_demand_type == -1:
        return 0.0

     # Expand services for high-demand task types
    target_increase = 0
    for node_idx in range(len(deploy_state_before)):
        delta = deploy_state_after[node_idx][max_demand_type] - deploy_state_before[node_idx][max_demand_type]
        target_increase += max(delta, 0)
    target_reward = target_increase * task_type_weights[max_demand_type] * 2.0

    # Expand services for non-high-demand task types
    non_target_penalty = 0
    for task_type in range(max_task_type):
        if task_type == max_demand_type:
            continue
        for node_idx in range(len(deploy_state_before)):
            delta = deploy_state_after[node_idx][task_type] - deploy_state_before[node_idx][task_type]
            non_target_penalty += max(delta, 0) * task_type_weights[task_type]

    total_reward = target_reward - non_target_penalty * 0.3
    return total_reward

def calculate_reward(master1, master2, cur_done, cur_undone):
    weights = {
        'throughput': 1,
        'fail_penalty': -0.15,
        'resource_penalty': -0.05,
        'throughput_bonus': 0.3
    }

    # Core throughput reward
    total_tasks = [
        cur_done[0] + cur_undone[0],
        cur_done[1] + cur_undone[1]
    ]
    throughput_rate = [
        cur_done[0] / (total_tasks[0] + 1e-6),
        cur_done[1] / (total_tasks[1] + 1e-6)
    ]
    # Failure penalty, only penalized when failure rate exceeds 10%
    fail_penalty = [
        cur_undone[0] / (total_tasks[0] + 1e-6),
        cur_undone[1] / (total_tasks[1] + 1e-6)
    ]
    # Resource penalty (only for overload)
    def calculate_overload(master):
        overload_scores = []
        for node in master.node_list:
            # Only penalize when CPU or memory exceeds 85%
            cpu_overload = max(node.cpu / node.cpu_max - 0.85, 0)
            mem_overload = max(node.mem / node.mem_max - 0.85, 0)
            overload_scores.append(cpu_overload + mem_overload)
        return np.mean(overload_scores)

    resource_penalty = [
        calculate_overload(master1),
        calculate_overload(master2)
    ]

    # Non-linear throughput incentive
    throughput_bonus = [
        np.exp(2 * throughput_rate[0]) - 1,
        np.exp(2 * throughput_rate[1]) - 1
    ]
    rewards = []
    for i in range(2):
        reward = (
                weights['throughput'] * throughput_rate[i] +
                weights['fail_penalty'] * fail_penalty[i] +
                weights['resource_penalty'] * resource_penalty[i] +
                weights['throughput_bonus'] * throughput_bonus[i]
        )
        rewards.append(reward)
    return rewards

def to_grid_rewards(node_reward):
    return np.array(node_reward).reshape([-1, 1])

def serve():
    # Configure server parameters
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', 100 * 1024 * 1024),
            ('grpc.max_receive_message_length', 100 * 1024 * 1024)
        ]
    )
    # Registration Services
    cluster_pb2_grpc.add_ClusterServiceServicer_to_server(
        ClusterService(), server
    )

    # Start the service
    server.add_insecure_port('[::]:50052')
    server.start()
    print("The gRPC server has started and is listening on port 50052.")
    server.wait_for_termination()

global_q_estimator = None
global_epsilon = 0.5
global_s_grid = None
global_ava_node = None
global_context = [1, 1]
global_master1 = None
global_master2 = None
global_cloud = None

def start_servers():
    # Create and configure threads
    cluster1_thread = threading.Thread(
        target=serve,
        name="Cluster2-Server",
        daemon=True
    )
    # Start thread
    cluster1_thread.start()
#
class ClusterService(cluster_pb2_grpc.ClusterServiceServicer):
    def GetAvailableNodes(self, request, context):
        raw_tasks = []
        for value in request.tasks:
            field_name = value.WhichOneof("value")
            if field_name == "int_value":
                raw_tasks.append(value.int_value)
            elif field_name == "float_value":
                raw_tasks.append(value.float_value)

        with threading.Lock():
            q_estimator = global_q_estimator
            epsilon = global_epsilon
            s_grid = global_s_grid
            ava_node = global_ava_node
            master1 = global_master1
            master2 = global_master2
            cloud = global_cloud
            # Execute action decision
            act, _, _, _, _, _, _, _ = q_estimator.action(
                s_grid, ava_node, global_context, epsilon
            )
        node_index = act[0]
        if 0 <= node_index < 6:
            master1.node_list[node_index].task_queue.insert(0, raw_tasks)
        elif 6 <= node_index < 12:
            master2.node_list[node_index - 6].task_queue.insert(0, raw_tasks)
        else:
            cloud.task_queue.insert(0, raw_tasks)
        # Construct node response
        single_node = cluster_pb2.NodeInfo(
            node_id=str(node_index),
            ip="localhost",  # Needs to be replaced with actual IP
            port=50052,
            labels={"Completed"}
        )
        return cluster_pb2.NodeResponse(node=single_node)



train_counter = 0
exit_flag = False
global_step2 = 0
def execution(RUN_TIMES, BREAK_POINT, TRAIN_TIMES, CHO_CYCLE):
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    experience_counter = 0
    global global_q_estimator, global_epsilon, global_s_grid, global_ava_node, global_context
    global global_master1, global_master2, global_cloud
    train_condition = threading.Condition()
    global train_counter
    global exit_flag
    global global_step2
    train_counter = 0
    exit_flag = False
    global_step2 = 0
    lambda_factor = 0.8 # Discount factor for Estimated time for task types
    task_type_estimates = defaultdict(float)  # Estimated time for task types
    task_type_actuals = defaultdict(list)  # tore actual processing time for calculation
    loss_tmp = []
    node_actor_loss_t = []
    service_actor_loss_t = []
    node_value_loss_t = []
    service_value_loss_t = []
    policy_losses = []
    value_losses = []

    vaild_node = 12  # Number of available edge nodes
    SLOT_TIME = 0.25  # Length of a time slot
    MAX_TESK_TYPE = 12  # Task types
    POD_CPU = 15.0  # CPU resources required by POD
    POD_MEM = 1.0  # Memory resources required by POD
    # Resource requirement coefficients for different service types
    service_coefficient = [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 1.4, 1.4]
    # DRL-related parameters
    epsilon_start = 0.5  # Initial exploration rate
    epsilon_end = 0.01  # Minimum exploration rate
    epsilon_decay = 0.995
    epsilon = epsilon_start
    gamma = 0.9
    starter_learning_rate = 5e-4
    global_step = tf.Variable(0, trainable=False, name='global_step')
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)

    action_dim = 13
    state_dim = 175
    node_input_dim = 24

    cluster_input_dim = 24
    hid_dims = [64, 32]
    output_dim = 24
    max_depth = 3
    entropy_weight_init = 1

    exec_cap = 24
    entropy_weight_min = 0.0001
    entropy_weight_decay = 1e-3

    # GPU-related parameters
    worker_num_gpu = 2
    worker_gpu_fraction = 0.5

    record = []
    throughput_list = []
    sum_rewards = []
    achieve_num = []
    achieve_num_sum = []
    fail_num = []
    deploy_reward = []
    current_time = str(time.time())
    log_dir = "./log/{}/".format(current_time)
    all_rewards = []

    episode_rewards = []

    sess = tf.Session()
    tf.set_random_seed(1)
    q_estimator = Estimator(sess, action_dim, state_dim, 2, learning_rate=learning_rate, global_step=global_step,
                            scope="q_estimator", summaries_dir=log_dir)
    global_q_estimator = q_estimator
    global_epsilon = epsilon
    sess.run(tf.global_variables_initializer())

    double_buffer = DoubleBuffer(memory_size=1e+6, batch_size=int(3200), max_episodes=3)
    saver = tf.compat.v1.train.Saver()

    all_task1 = get_all_task('./data/Task_1_test.csv')
    all_task2 = get_all_task('./data/Task_2_test.csv')

    config = tf.ConfigProto(device_count={'GPU': worker_num_gpu},
                            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=worker_gpu_fraction))
    sess = tf.Session(config=config)
    orchestrate_agent = OrchestrateAgent(sess, node_input_dim, cluster_input_dim, hid_dims, output_dim, max_depth,
                                         range(1, exec_cap + 1),
                                         optimizer=lambda lr: tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3))
    def training_worker():
        global global_step2, exit_flag,train_counter
        while True:
            with train_condition:
                # Wait for main thread notification or exit signal
                while train_counter == 0 and not exit_flag:
                    train_condition.wait()
                if exit_flag:
                    break

                train_counter = 0
            # Acquire training buffer
            train_buffer = double_buffer.get_train_buffer()
            # Execute asynchronous training
            for _ in range(TRAIN_TIMES):
                policy_loss, value_loss = q_estimator.update_policy(train_buffer, global_step2)
                global_step2 += 1
                # Record losses
            with train_condition:
                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                # Synchronous Network
            if global_step2 % 100 == 0:
                q_estimator.verify_target_network_sync()


    train_thread = threading.Thread(target=training_worker, daemon=True)
    train_thread.start()


    exp = {
        'samples': [],  # Structured experience pool
        'wall_time': []
    }
    start_servers()  # Start gRPC service
    for n_iter in np.arange(RUN_TIMES):
        # Initialize settings and repeat experiments multiple times

        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        batch_reward = []

        oreward = []
        cur_time = 0
        entropy_weight = entropy_weight_init
        orchestration_reward = 0.0
        pre_done = [0, 0]
        pre_undone = [0, 0]

        pre_done_kind = [[0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE]
        pre_undone_kind = [[0] * MAX_TESK_TYPE, [0] * MAX_TESK_TYPE]
        context = [1, 1]
        global_context = context

        ############ Set up according to your own needs  ###########
        # The parameters here are set only to support the operation of the program, and may not be consistent with the actual system

        deploy_state = [
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Node 0
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # Node 1
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],  # Node 2
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],  # Node 3
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 4
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],  # Node 5
            [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # Node 6
            [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1],  # Node 7
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],  # Node 8
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],  # Node 9
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Node 10
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1]  # Node 11
        ]

        node_list1 = [Node(i, 195, 1750, 2000, 308000, [], []) for i in range(6)]

        node_list2 = [Node(i + 6, 195, 1750, 2000, 308000, [], []) for i in range(6)]

        global_master1 = Master(0, 4000, 308000, node_list1, [], all_task1, 0, 0, 0, [0] * MAX_TESK_TYPE,
                                [0] * MAX_TESK_TYPE)
        global_master2 = Master(1, 4000, 308000, node_list2, [], all_task2, 0, 0, 0, [0] * MAX_TESK_TYPE,
                                [0] * MAX_TESK_TYPE)


        global_cloud = Cloud(12, [], [], 4000, 10000, 16000, 1024000)  # (..., cpu, mem)
        master1 = global_master1
        master2 = global_master2
        cloud = global_cloud
        for i in range(MAX_TESK_TYPE):
            docker = Docker(POD_MEM * service_coefficient[i], POD_CPU * service_coefficient[i], cur_time, cur_time,i, [-1])
            cloud.service_list.append(docker)

        # Create Docker containers based on deploy_state
        for i in range(vaild_node):
            for ii in range(MAX_TESK_TYPE):
                dicision = deploy_state[i][ii]
                if i < 6 and dicision == 1:
                    j = i
                    if master1.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,cur_time,
                                        ii, [-1])
                        master1.node_list[j].mem = master1.node_list[j].mem + POD_MEM * service_coefficient[ii]
                        master1.node_list[j].service_list.append(docker)

                if i >= 6 and dicision == 1:
                    j = i - 6
                    if master2.node_list[j].mem >= POD_MEM * service_coefficient[ii]:
                        docker = Docker(POD_MEM * service_coefficient[ii], POD_CPU * service_coefficient[ii], cur_time,cur_time,
                                        ii, [-1])
                        master2.node_list[j].mem = master2.node_list[j].mem + POD_MEM * service_coefficient[ii]
                        master2.node_list[j].service_list.append(docker)

        #  slot
        for slot in range(BREAK_POINT):

            cur_time = cur_time + SLOT_TIME
            # frame
            if slot % CHO_CYCLE == 0 and slot != 0:
                done_tasks = []
                undone_tasks = []
                curr_tasks_in_queue = []
                cur_done_kind = [
                    subtract_lists(master1.done_kind, pre_done_kind[0]),
                    subtract_lists(master2.done_kind, pre_done_kind[1])]
                cur_undone_kind = [
                    subtract_lists(master1.undone_kind, pre_undone_kind[0]),
                    subtract_lists(master2.undone_kind, pre_undone_kind[1])]
                # Data on completed and incomplete task types prior to the update
                pre_done_kind = [master1.done_kind.copy(), master2.done_kind.copy()]
                pre_undone_kind = [master1.undone_kind.copy(), master2.undone_kind.copy()]
                # Retrieve task status, including success, failure, and unresolved
                for i in range(MAX_TESK_TYPE):
                    done_tasks.append(float(master1.done_kind[i] + master2.done_kind[i]))
                    undone_tasks.append(float(master1.undone_kind[i] + master2.undone_kind[i]))
                for i in range(6):
                    tmp = [0.0] * MAX_TESK_TYPE
                    for j in range(len(master1.node_list[i].task_queue)):
                        tmp[master1.node_list[i].task_queue[j][0]] = tmp[master1.node_list[i].task_queue[j][0]] + 1.0
                    curr_tasks_in_queue.append(tmp)
                for i in range(6):
                    tmp = [0.0] * MAX_TESK_TYPE
                    for k in range(len(master2.node_list[i].task_queue)):
                        tmp[master2.node_list[i].task_queue[k][0]] = tmp[master2.node_list[i].task_queue[k][0]] + 1
                    curr_tasks_in_queue.append(tmp)
                deploy_state_float = []
                for i in range(len(deploy_state)):
                    tmp = []
                    for j in range(len(deploy_state[0])):
                        tmp.append(float(deploy_state[i][j]))
                    deploy_state_float.append(tmp)
                    # Service orchestration decision
                change_node, change_service, exp = act_offload_agent(orchestrate_agent, exp, done_tasks, undone_tasks,
                                                                     curr_tasks_in_queue, deploy_state_float)

                # Execute service orchestration decisions
                for i in range(len(change_node)):
                    if change_service[i] < 0:
                        # Delete docker and release memory
                        service_index = -1 * change_service[i] - 1
                        if change_node[i] < 6:
                            docker_idx = 0
                            while docker_idx < len(master1.node_list[change_node[i]].service_list):
                                if docker_idx >= len(master1.node_list[change_node[i]].service_list):
                                    break
                                if master1.node_list[change_node[i]].service_list[docker_idx].kind == service_index:
                                    master1.node_list[change_node[i]].mem = master1.node_list[change_node[i]].mem + \
                                                                            master1.node_list[
                                                                                change_node[i]].service_list[
                                                                                docker_idx].mem
                                    del master1.node_list[change_node[i]].service_list[docker_idx]
                                    deploy_state[change_node[i]][service_index] = max(0, deploy_state[change_node[i]][
                                        service_index] - 1.0)
                                    break
                                else:
                                    docker_idx = docker_idx + 1
                        else:
                            node_index = change_node[i] - 6
                            docker_idx = 0
                            while docker_idx < len(master2.node_list[node_index].service_list):
                                if docker_idx >= len(master2.node_list[node_index].service_list):
                                    break
                                if master2.node_list[node_index].service_list[docker_idx].kind == service_index:
                                    master2.node_list[node_index].mem = master2.node_list[node_index].mem + \
                                                                        master2.node_list[node_index].service_list[
                                                                            docker_idx].mem
                                    del master2.node_list[node_index].service_list[docker_idx]
                                    deploy_state[change_node[i]][service_index] = max(0, deploy_state[change_node[i]][
                                        service_index] - 1.0)
                                else:
                                    docker_idx = docker_idx + 1
                    else:
                        # Add docker
                        service_index = change_service[i] - 1
                        if change_node[i] < 6:
                            if master1.node_list[change_node[i]].mem >= POD_MEM * service_coefficient[service_index] :
                                docker = Docker(POD_MEM * service_coefficient[service_index],
                                                POD_CPU * service_coefficient[service_index],
                                                cur_time,cur_time, service_index, [-1])
                                master1.node_list[change_node[i]].mem = master1.node_list[
                                                                            change_node[i]].mem - POD_MEM * \
                                                                        service_coefficient[service_index]
                                master1.node_list[change_node[i]].service_list.append(docker)

                                deploy_state[change_node[i]][service_index] = deploy_state[change_node[i]][service_index] + 1

                        else:
                            node_index = change_node[i] - 6
                            if master2.node_list[node_index].mem >= POD_MEM * service_coefficient[service_index] :
                                docker = Docker(POD_MEM * service_coefficient[service_index],
                                                POD_CPU * service_coefficient[service_index],
                                                cur_time,cur_time, service_index, [-1])
                                master2.node_list[node_index].mem = master2.node_list[node_index].mem - POD_MEM * \
                                                                    service_coefficient[service_index]
                                master2.node_list[node_index].service_list.append(docker)

                                deploy_state[change_node[i]][service_index] =deploy_state[change_node[i]][service_index] + 1
                task_backlog_before = get_task_backlog(master1, master2, MAX_TESK_TYPE)
                orchestration_reward = calculate_task_match_reward(task_backlog_before,change_service, deploy_state_float,deploy_state, cur_done_kind, cur_undone_kind)

                exp['samples'][-1]['service_reward'] = orchestration_reward
                exp['samples'][-1]['node_reward'] = sum(immediate_reward)
                exp['wall_time'].append(cur_time)

                # Service Orchestration Training
                if slot % (32 * CHO_CYCLE) == 0:
                    entropy_weight, loss,node_actor_loss,service_actor_loss,node_value_loss,service_value_loss = train_orchestrate_agent(orchestrate_agent, exp, entropy_weight,entropy_weight_min, entropy_weight_decay)
                    loss_tmp.append(loss)
                    node_actor_loss_t.append(node_actor_loss)
                    service_actor_loss_t.append(service_actor_loss)
                    node_value_loss_t.append(node_value_loss)
                    service_value_loss_t.append(service_value_loss)

            master1 = update_task_queue(master1, cur_time, master1.id, deploy_state)
            master2 = update_task_queue(master2, cur_time, master2.id, deploy_state)
            task1 = [-1]
            task2 = [-1]
            if len(master1.task_queue) != 0:
                task1 = master1.task_queue[0]
                del master1.task_queue[0]
            if len(master2.task_queue) != 0:
                task2 = master2.task_queue[0]
                del master2.task_queue[0]
            curr_task = [task1, task2]
            if curr_task != [[-1], [-1]]:
                ava_node = []
                for i in range(len(curr_task)):
                    if curr_task[i][0] == -1:
                        continue
                    tmp_list = [12]  # cloud
                    for ii in range(len(deploy_state)):
                        if deploy_state[ii][curr_task[i][0]] >= 1:
                            tmp_list.append(ii)
                    ava_node.append(tmp_list)

                # Current Status of CPU and Memory
                cpu_list1 = []
                mem_list1 = []
                cpu_list2 = []
                mem_list2 = []
                task_num1 = [len(master1.task_queue)]
                task_num2 = [len(master2.task_queue)]
                for i in range(6):
                    cpu_list1.append([master1.node_list[i].cpu, master1.node_list[i].cpu_max])
                    mem_list1.append([master1.node_list[i].mem, master1.node_list[i].mem_max])
                    task_num1.append(len(master1.node_list[i].task_queue))
                for i in range(6):
                    cpu_list2.append([master2.node_list[i].cpu, master2.node_list[i].cpu_max])
                    mem_list2.append([master2.node_list[i].mem, master2.node_list[i].mem_max])
                    task_num2.append(len(master2.node_list[i].task_queue))
                s_grid = np.array([flatten(flatten([deploy_state, [task_num1], cpu_list1, mem_list1])),
                                   flatten(flatten([deploy_state, [task_num2], cpu_list2, mem_list2]))])
                global_s_grid = s_grid
                global_ava_node = ava_node
                # Request for Scheduling Decision
                act, valid_action_prob_mat, policy_state, action_choosen_mat, \
                    curr_state_value, curr_neighbor_mask, next_state_ids,old_probs_list = q_estimator.action(s_grid, ava_node, context,epsilon)
                # Place the current task into the queue based on scheduling decisions.
                for i in range(len(act)):
                    if curr_task[i][0] == -1:
                        continue
                    if act[i] == 12:
                        cloud.task_queue.append(curr_task[i])
                        continue
                    if 0 <= act[i] < 6:
                        master1.node_list[act[i]].task_queue.append(curr_task[i])
                        continue
                    if 6 <= act[i] < 12:
                        master2.node_list[act[i] - 6].task_queue.append(curr_task[i])
                        continue
                    else:
                        pass
            # Update task status
            for i in range(6):
                master1.node_list[i].task_queue, undone, undone_kind = check_queue(master1.node_list[i].task_queue,cur_time)
                for j in undone_kind:
                    master1.undone_kind[j] = master1.undone_kind[j] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]
                master2.node_list[i].task_queue, undone, undone_kind = check_queue(master2.node_list[i].task_queue,cur_time)
                for j in undone_kind:
                    master2.undone_kind[j] = master2.undone_kind[j] + 1
                master1.undone = master1.undone + undone[0]
                master2.undone = master2.undone + undone[1]

            cloud.task_queue, undone, undone_kind = check_queue(cloud.task_queue, cur_time)
            if undone[0] > 0:
                for j in undone_kind:
                    master1.undone_kind[j] = master1.undone_kind[j] + 1
            else:
                for j in undone_kind:
                    master2.undone_kind[j] = master2.undone_kind[j] + 1
            master1.undone = master1.undone + undone[0]
            master2.undone = master2.undone + undone[1]

            master1, master2, cloud = update_all_docker_states(master1, master2, cloud, cur_time,
                                                               service_coefficient, POD_CPU,deploy_state,
                                                               task_type_estimates,task_type_actuals,lambda_factor)
            if slot % 2 == 0 and slot != 0:
                # Task update is about to expire
                master1 = check_timeout_queue_master(cloud, master1, cur_time, deploy_state,task_type_estimates)
                master2 = check_timeout_queue_master(cloud, master2, cur_time, deploy_state,task_type_estimates)
                master1 = check_timeout_queue_node(cloud, master1, cur_time, deploy_state,task_type_estimates,task_type_actuals)
                master2 = check_timeout_queue_node(cloud, master2, cur_time, deploy_state,task_type_estimates,task_type_actuals)

                master1, master2, cloud = update_all_docker_states(master1, master2, cloud, cur_time,
                                                                   service_coefficient, POD_CPU,deploy_state,
                                                                   task_type_estimates,task_type_actuals,lambda_factor)



            cur_done = [master1.done - pre_done[0], master2.done - pre_done[1]]
            cur_undone = [master1.undone - pre_undone[0], master2.undone - pre_undone[1]]
            pre_done = [master1.done, master2.done]
            pre_undone = [master1.undone, master2.undone]
            achieve_num.append(sum(cur_done))
            fail_num.append(sum(cur_undone))
            immediate_reward = calculate_reward(master1, master2, cur_done, cur_undone)
            oreward.append(orchestration_reward)
            record.append([master1, master2, cur_done, cur_undone, immediate_reward, orchestration_reward])
            deploy_reward.append(orchestration_reward)

            if slot != 0:
                r_grid = to_grid_rewards(immediate_reward)
                targets_batch,qvalue_next = q_estimator.compute_targets(action_mat_prev, s_grid, r_grid, gamma)
                # Calculate advantage function for training policy network
                advantage,qvalue_next_adv = q_estimator.compute_advantage(curr_state_value_prev, next_state_ids_prev,
                                                          s_grid, r_grid, gamma)
                if curr_task[0][0] != -1 and curr_task[1][0] != -1:
                    policy_replay = double_buffer.get_write_buffer()

                    policy_replay.add(policy_state_prev, action_choosen_mat_prev, np.array(advantage).reshape(-1, 1), curr_neighbor_mask_prev,
                                      old_probs_list,np.array(act).reshape(-1, 1),state_mat_prev, action_mat_prev, targets_batch, s_grid)

                    experience_counter += 1
                    if experience_counter >= 3300:
                        policy_replay.add_episode_boundary()
                        double_buffer.swap_buffers()
                        experience_counter = 0
                        with train_condition:
                            train_counter += 1
                            train_condition.notify()  # Wake up training thread

            state_mat_prev = s_grid
            action_mat_prev = valid_action_prob_mat
            action_choosen_mat_prev = action_choosen_mat
            curr_neighbor_mask_prev = curr_neighbor_mask
            policy_state_prev = policy_state


            curr_state_value_prev = curr_state_value
            next_state_ids_prev = next_state_ids
            all_rewards.append(sum(immediate_reward))
            batch_reward.append(orchestration_reward)

        sum_rewards.append(float(sum(all_rewards)) / float(len(all_rewards)))
        all_rewards = []
        all_number = sum(achieve_num) + sum(fail_num)
        throughput_list.append(sum(achieve_num) / float(all_number))
        print('throughput_list_all =', throughput_list, '\ncurrent_achieve_number =', sum(achieve_num),
              ', current_fail_number =', sum(fail_num))


        achieve_num = []
        fail_num = []
        episode_reward = np.sum(batch_reward[1:])
        episode_rewards.append(episode_reward)
        saver.save(sess, "./model/model.ckpt")
    with train_condition:
        exit_flag = True
        train_condition.notify_all()
    train_thread.join()  # Wait for the training thread to exit
    saver.save(sess, "./model/model_before_testing.ckpt")
    tf.reset_default_graph()
    time_str = str(time.time())
    with open("./result/" + time_str + ".json", "w") as f:
        json.dump(record, f, cls=CustomEncoder)
    return throughput_list


if __name__ == "__main__":
    ############ Set up according to your own needs  ###########
    # The parameters are set to support the operation of the program, and may not be consistent with the actual system
    RUN_TIMES = 100
    TASK_NUM = 3300
    TRAIN_TIMES = 50
    CHO_CYCLE = 100
    execution(RUN_TIMES, TASK_NUM, TRAIN_TIMES, CHO_CYCLE)
