import csv
import grpc

from gRPC.generated import cluster_pb2_grpc
from gRPC.generated import cluster_pb2

def get_all_task(path):
    type_list = []
    start_time = []
    end_time = []
    cpu_list = []
    mem_list = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            type_list.append(row[3])
            start_time.append(row[5])
            end_time.append(row[6])
            cpu_list.append(row[7])
            mem_list.append(row[8])

    init_time = int(start_time[0])
    for i in range(len(start_time)):
        type_list[i] = int(type_list[i]) - 1
        start_time[i] = int(start_time[i]) - init_time
        end_time[i] = int(end_time[i]) - init_time
        cpu_list[i] = int(cpu_list[i]) / 100.0
        mem_list[i] = float(mem_list[i])
    all_task = [type_list, start_time, end_time, cpu_list, mem_list]
    return all_task


def put_task(task_queue, task):
    for i in range(len(task_queue) - 1):
        j = len(task_queue) - i - 1
        task_queue[j] = task_queue[j - 1]
    task_queue[0] = task
    return task_queue


def check_timeout_queue_master(cloud, master, cur_time, deploy_state,task_type_estimates):

    while len(master.task_queue) > 0 and cur_time >= master.task_queue[0][2] - task_type_estimates.get(master.task_queue[0][0],2.5):
        task = master.task_queue[0]
        ava_node = []
        if len(task) > 1:
            offset = 0 if master.id == 0 else 6
            for node_type in range(6):
                if deploy_state[node_type + offset][master.task_queue[0][0]] >= 1:
                    ava_node.append(node_type + offset)
        for node in master.node_list:
            for service in node.service_list:

                if service.kind == task[0] and len(service.doing_task) > 1:
                    if node.id in ava_node:
                        ava_node.remove(node.id)
                        break
        if len(ava_node) == 0:
            ava_node.append(12)

        if len(ava_node) >= 1 and ava_node[0] != 12:
            offset = 0 if master.id == 0 else 6
            selected_node_idx = max(ava_node,
                                    key=lambda node_idx: (
                                            0.6 * ((task[2] - cur_time) / max(task[2] - task[1], 1e-6)) +
                                            0.2 * ((master.node_list[node_idx - offset].cpu_max - master.node_list[node_idx - offset].cpu) / (
                                                    master.node_list[node_idx - offset].cpu_max + 1e-6)) +
                                            0.2 * ((master.node_list[node_idx - offset].mem_max - master.node_list[node_idx - offset].mem) / (
                                                    master.node_list[node_idx - offset].mem_max + 1e-6))
                                    ))
            master.node_list[selected_node_idx - offset].task_queue.append(task)
            del master.task_queue[0]
        else:
            cloud.task_queue.append(task)
            del master.task_queue[0]
    return master


def check_timeout_queue_node(cloud, master, cur_time, deploy_state,task_type_estimates,task_type_actuals):

    num = 0
    for n in master.node_list:
        i = 0
        while len(n.task_queue) > 0 and i < len(n.task_queue):
            if cur_time >= n.task_queue[i][2] - task_type_estimates.get(n.task_queue[i][0],2.5):
                task = [-1]
                for s in n.service_list:
                    if s.kind == n.task_queue[i][0]:
                        num = num + 1
                        if s.doing_task[0] != -1:
                            num = num - 1
                            continue

                if num == 0:
                    task = n.task_queue[0]
                ava_node = []
                if len(task) > 1:
                    offset = 0 if master.id == 0 else 6
                    for node_type in range(6):
                        if deploy_state[node_type + offset][n.task_queue[i][0]] >= 1:
                            ava_node.append(node_type + offset)
                else:
                    break
                for node in master.node_list:
                    for service in node.service_list:

                        if service.kind == task[0] and len(service.doing_task) > 1:
                            if node.id in ava_node:
                                ava_node.remove(node.id)
                                break
                if len(ava_node) == 0:
                    ava_node.append(12)

                if len(ava_node) >= 1 and ava_node[0] != 12:
                    offset = 0 if master.id == 0 else 6
                    selected_node_idx = max(ava_node,
                                            key=lambda node_idx: (
                                                    0.6 * ((task[2] - cur_time) / max(task[2] - task[1], 1e4-6)) +
                                                    0.2 * ((master.node_list[node_idx - offset].cpu_max -
                                                            master.node_list[node_idx - offset].cpu) / (master.node_list[node_idx - offset].cpu_max + 1e-6)) +
                                                    0.2 * ((master.node_list[node_idx - offset].mem_max -
                                                            master.node_list[node_idx - offset].mem) / (master.node_list[node_idx - offset].mem_max + 1e-6))
                                            ))
                    master.node_list[selected_node_idx - offset].task_queue.append(task)
                    del n.task_queue[i]
                else:
                    cloud.task_queue.append(task)
                    del n.task_queue[i]
            else:
                i += 1
    return master


def get_available_nodes(task):
    channel = grpc.insecure_channel('localhost:50051')
    stub = cluster_pb2_grpc.ClusterServiceStub(channel)
    try:
        request = cluster_pb2.NodeRequest(
            source_cluster_id="cluster-1",
            tasks=[
                cluster_pb2.Value(int_value=x) if isinstance(x, int)
                else cluster_pb2.Value(float_value=x)
                for x in task
            ]
        )
        response = stub.GetAvailableNodes(request, timeout=5)
        if response.node:
            return {
                'node_id': response.node.node_id,
                'ip': response.node.ip,
                'port': response.node.port,
                'labels': dict(response.node.labels)
            }
        return None
    except grpc.RpcError as e:
        print(f"RPC error: {e.code()}: {e.details()}")
        return None
def check_timeout_queue_task(task, cloud, master1, master2, cur_time, deploy_state, POD_CPU, service_coefficient,task_type_estimates,task_type_actuals,lambda_factor):
    flag = 0
    done = [0, 0]
    done_kind = []
    ava_node = []
    for node in master1.node_list:
        if deploy_state[node.id][task[0]] >= 1:
            ava_node.append(node.id)
    for node in master1.node_list:
        if node.id in ava_node:

            available_services = sum(
                1 for service in node.service_list
                if service.kind == task[0] and len(service.doing_task) <= 1
            )
            if available_services == 0:
                ava_node.remove(node.id)
    if len(ava_node) == 0:
        node = get_available_nodes(task)
        if node.node_id >= 0:
            ava_node.append(node.node_id)
            flag = 1
        done[task[5]] = done[task[5]] + 1
        done_kind.append(task[0])
        return 1, done, done_kind

    if len(ava_node) == 0:
        ava_node.append(12)
        flag = 1

    if len(ava_node) >= 1 and ava_node[0] != 12:
        def calculate_score(node_idx):

            if node_idx < 6:
                current_master = master1
                offset = 0
            else:
                current_master = master2
                offset = 6
            return 0.6 * ((task[2] - cur_time) / max(task[2] - task[1], 1e-6)) + \
                    0.2 * ((current_master.node_list[node_idx - offset].cpu_max - current_master.node_list[node_idx - offset].cpu) / (
                            current_master.node_list[node_idx - offset].cpu_max + 1e-6)) + \
                    0.2 * ((current_master.node_list[node_idx - offset].mem_max - current_master.node_list[node_idx - offset].mem) / (
                            current_master.node_list[node_idx - offset].mem_max + 1e-6))

        selected_node_idx = max(ava_node, key=calculate_score)


        if selected_node_idx < 6:
            target_master = master1
            target_offset = 0
        else:
            target_master = master2
            target_offset = 6
        node = target_master.node_list[selected_node_idx - target_offset]
        node.task_queue.insert(0, task)

        for i in range(len(node.service_list)):
            if node.service_list[i].available_time <= cur_time and len(node.service_list[i].doing_task) > 1:
                task_type = node.service_list[i].doing_task[0]
                actual_time = cur_time - node.service_list[i].start_time
                if task_type in task_type_estimates:
                    task_type_estimates[task_type] = lambda_factor * task_type_estimates[task_type] + (1 - lambda_factor) * actual_time
                else:
                    task_type_estimates[task_type] = actual_time
                done[node.service_list[i].doing_task[5]] = done[node.service_list[i].doing_task[5]] + 1
                done_kind.append(task_type)
                node.service_list[i].doing_task = [-1]
                node.service_list[i].available_time = cur_time
        i = 0
        if len(node.task_queue) > 0:
            for j in range(len(node.service_list)):
                if node.task_queue[i][0] == node.service_list[j].kind:
                    if node.service_list[j].available_time > cur_time:
                        continue
                    if node.service_list[j].available_time <= cur_time:
                        to_do = (node.task_queue[i][3]) / node.service_list[j].cpu
                        if cur_time + to_do <= node.task_queue[i][2] and (node.cpu_max - node.cpu) >= POD_CPU * \
                                service_coefficient[node.task_queue[i][0]]:
                            node.cpu = node.cpu + POD_CPU * service_coefficient[node.task_queue[i][0]]
                            node.service_list[j].start_time = cur_time
                            node.service_list[j].available_time = cur_time + to_do
                            node.service_list[j].doing_task = node.task_queue[i]
                            del node.task_queue[i]
                            return 1 ,done, done_kind

    if flag == 1:
        cloud.task_queue.insert(0, task)
        for i in range(len(cloud.service_list)):
            if cloud.service_list[i].available_time <= cur_time and len(cloud.service_list[i].doing_task) > 1:
                task_type = cloud.service_list[i].doing_task[0]
                actual_time = cur_time - cloud.service_list[i].start_time
                if task_type in task_type_estimates:
                    task_type_estimates[task_type] = lambda_factor * task_type_estimates[task_type] + (1 - lambda_factor) * actual_time
                else:
                    task_type_estimates[task_type] = actual_time
                done[cloud.service_list[i].doing_task[5]] = done[cloud.service_list[i].doing_task[5]] + 1
                done_kind.append(task_type)
                cloud.cpu = cloud.cpu - POD_CPU * service_coefficient[cloud.service_list[i].doing_task[0]]
                cloud.service_list[i].doing_task = [-1]
                cloud.service_list[i].available_time = cur_time
        i = 0
        if len(cloud.task_queue) > 0:
            for j in range(len(cloud.service_list)):
                if cloud.task_queue[i][0] == cloud.service_list[j].kind:
                    if cloud.service_list[j].available_time > cur_time:
                        del cloud.task_queue[i]

                        return 0, done, done_kind
                    if cloud.service_list[j].available_time <= cur_time:

                        to_do = (cloud.task_queue[i][3]) / cloud.service_list[j].cpu
                        if cur_time + to_do <= cloud.task_queue[i][2] and (cloud.cpu_max - cloud.cpu) >= POD_CPU * \
                                service_coefficient[cloud.task_queue[i][0]]:
                            cloud.cpu = cloud.cpu + POD_CPU * service_coefficient[cloud.task_queue[i][0]]
                            cloud.service_list[j].start_time = cur_time
                            cloud.service_list[j].available_time = cur_time + to_do
                            cloud.service_list[j].doing_task = cloud.task_queue[i]
                            del cloud.task_queue[i]
                            return 1, done, done_kind
    return 0,done, done_kind


def update_task_queue(master, cur_time, master_id, deploy_state):
    k = 0
    i = 0
    while len(master.task_queue) > i:
        if master.task_queue[i][0] == -1:
            i = i + 1
            continue
        if cur_time >= master.task_queue[i][2]:
            master.undone = master.undone + 1
            master.undone_kind[master.task_queue[i][0]] = master.undone_kind[master.task_queue[i][0]] + 1
            del master.task_queue[i]
        else:
            i = i + 1

    while master.all_task[1][master.all_task_index] < cur_time:
        k += 1
        task = [master.all_task[0][master.all_task_index], master.all_task[1][master.all_task_index],
                master.all_task[2][master.all_task_index], master.all_task[3][master.all_task_index],
                master.all_task[4][master.all_task_index], master_id]
        master.task_queue.append(task)
        master.all_task_index = master.all_task_index + 1

    tmp_task_list = []
    for i in range(len(master.task_queue)):
        if master.task_queue[i][0] != -1:
            tmp_task_list.append(master.task_queue[i])
    tmp_task_list = sorted(tmp_task_list, key=lambda x: (x[2] - x[1], x[2], x[1]))
    master.task_queue = tmp_task_list

    return master


def check_queue(task_queue, cur_time):
    task_queue = sorted(task_queue, key=lambda x: (x[2] - x[1], x[2], x[1]))
    undone = [0, 0]
    undone_kind = []
    i = 0
    while len(task_queue) != i:
        flag = 0
        if cur_time >= task_queue[i][2]:
            undone[task_queue[i][5]] = undone[task_queue[i][5]] + 1
            undone_kind.append(task_queue[i][0])
            del task_queue[i]
            flag = 1
        if flag == 1:
            flag = 0
        else:
            i = i + 1
    return task_queue, undone, undone_kind


def update_docker(cloud, master1, master2, node, cur_time, service_coefficient, POD_CPU, deploy_state,task_type_estimates,task_type_actuals,lambda_factor):
    done = [0, 0]
    undone = [0, 0]
    done_kind = []
    undone_kind = []


    for i in range(len(node.service_list)):
        if node.service_list[i].available_time <= cur_time and node.service_list[i].doing_task[0] != -1:
            task_type = node.service_list[i].doing_task[0]
            actual_time = cur_time - node.service_list[i].start_time
            task_type_actuals[task_type].append(actual_time)
            window_size = 10
            if len(task_type_actuals[task_type]) > window_size:
                task_type_actuals[task_type].pop(0)
            current_avg = sum(task_type_actuals[task_type]) / len(task_type_actuals[task_type])
            if task_type in task_type_estimates:
                task_type_estimates[task_type] = lambda_factor * task_type_estimates[task_type] + (1 - lambda_factor) * current_avg
            else:
                task_type_estimates[task_type] = actual_time
            done[node.service_list[i].doing_task[5]] = done[node.service_list[i].doing_task[5]] + 1
            done_kind.append(task_type)
            node.cpu = node.cpu - POD_CPU * service_coefficient[node.service_list[i].doing_task[0]]
            node.service_list[i].doing_task = [-1]
            node.service_list[i].available_time = cur_time


    i = 0
    while i != len(node.task_queue):
        flag = 0
        for j in range(len(node.service_list)):
            if i == len(node.task_queue):
                break
            if node.task_queue[i][0] == node.service_list[j].kind:
                if node.service_list[j].available_time > cur_time:
                    if cur_time >= node.task_queue[i][2] - task_type_estimates.get(node.task_queue[i][0], 2.5):
                        num, done_tmp, done_kind_tmp = check_timeout_queue_task(node.task_queue[i], cloud, master1, master2, cur_time,deploy_state, POD_CPU, service_coefficient,task_type_estimates,task_type_actuals,lambda_factor)
                        if num == 1:
                            flag = 1
                            done = [x + y for x, y in zip(done, done_tmp)]
                            done_kind.extend(done_kind_tmp)
                            del node.task_queue[i]
                            break
                        else:

                            break
                    continue
                if node.service_list[j].available_time <= cur_time:

                    to_do = (node.task_queue[i][3]) / node.service_list[j].cpu
                    if node.id == 12:
                        to_do = to_do + 1
                    if cur_time + to_do <= node.task_queue[i][2] and (node.cpu_max - node.cpu) >= POD_CPU * \
                            service_coefficient[node.task_queue[i][0]]:
                        node.cpu = node.cpu + POD_CPU * service_coefficient[node.task_queue[i][0]]
                        node.service_list[j].start_time = cur_time
                        node.service_list[j].available_time = cur_time + to_do
                        node.service_list[j].doing_task = node.task_queue[i]
                        del node.task_queue[i]
                        flag = 1
                        break
                    elif cur_time + to_do > node.task_queue[i][2] and node.id != 12:

                        undone[node.task_queue[i][5]] = undone[node.task_queue[i][5]] + 1
                        undone_kind.append(node.task_queue[i][0])
                        del node.task_queue[i]
                        flag = 1
                    elif (node.cpu_max - node.cpu) < POD_CPU * service_coefficient[node.task_queue[i][0]]:
                        pass

        if flag == 1:
            flag = 0
        else:
            i = i + 1
    return node, undone, done, done_kind, undone_kind


def update_all_docker_states(master1, master2, cloud, cur_time, service_coefficient, POD_CPU, deploy_state,task_type_estimates,task_type_actuals,lambda_factor):
    for i in range(6):
        master1.node_list[i].task_queue = sorted(master1.node_list[i].task_queue,
                                                 key=lambda x: (x[2] - x[1], x[2], x[1]))
        master1.node_list[i], undone, done, done_kind, undone_kind = update_docker(
            cloud, master1, master2, master1.node_list[i], cur_time, service_coefficient, POD_CPU, deploy_state,task_type_estimates,task_type_actuals,lambda_factor
        )
        for j in done_kind:
            master1.done_kind[j] += 1
        for j in undone_kind:
            master1.undone_kind[j] += 1
        master1.undone = master1.undone + undone[0]
        master2.undone = master2.undone + undone[1]
        master1.done = master1.done + done[0]
        master2.done = master2.done + done[1]


    for i in range(6):
        master2.node_list[i].task_queue = sorted(master2.node_list[i].task_queue,
                                                 key=lambda x: (x[2] - x[1], x[2], x[1]))
        master2.node_list[i], undone, done, done_kind, undone_kind = update_docker(
            cloud, master1, master2, master2.node_list[i], cur_time, service_coefficient, POD_CPU, deploy_state,task_type_estimates,task_type_actuals,lambda_factor
        )
        for j in done_kind:
            master2.done_kind[j] += 1
        for j in undone_kind:
            master2.undone_kind[j] += 1
        master1.undone = master1.undone + undone[0]
        master2.undone = master2.undone + undone[1]
        master1.done = master1.done + done[0]
        master2.done = master2.done + done[1]


    cloud.task_queue = sorted(cloud.task_queue, key=lambda x: (x[2] - x[1], x[2], x[1]))
    cloud, undone, done, done_kind, undone_kind = update_docker(
        cloud, master1, master2, cloud, cur_time, service_coefficient, POD_CPU, deploy_state,task_type_estimates,task_type_actuals,lambda_factor
    )
    if done[0] >0:
        for j in done_kind:
            master1.done_kind[j] += 1
        for j in undone_kind:
            master1.undone_kind[j] += 1
    else:
        for j in done_kind:
            master2.done_kind[j] += 1
        for j in undone_kind:
            master2.undone_kind[j] += 1
    master1.undone = master1.undone + undone[0]
    master2.undone = master2.undone + undone[1]
    master1.done = master1.done + done[0]
    master2.done = master2.done + done[1]

    return master1, master2, cloud
