import json
import numpy as np


class Cloud:
    def __init__(self,id, task_queue, service_list, cpu, mem, max_cpu, max_mem):
        self.task_queue = task_queue
        self.id = id
        self.service_list = service_list
        self.cpu = cpu  # GHz
        self.mem = mem  # GB
        self.cpu_max = max_cpu
        self.mem_max = max_mem

class Node:
    def __init__(self,id, cpu, mem, max_cpu, max_mem, service_list, task_queue):
        self.id = id
        self.cpu = cpu
        self.cpu_max = max_cpu
        self.mem = mem
        self.mem_max = max_mem
        self.service_list = service_list
        self.task_queue = task_queue


class Master:
    def __init__(self,id, cpu, mem, node_list, task_queue, all_task, all_task_index, done, undone, done_kind, undone_kind):
        self.id = id
        self.cpu = cpu  # GHz
        self.mem = mem  # MB
        self.node_list = node_list
        self.task_queue = task_queue
        self.all_task = all_task
        self.all_task_index = all_task_index
        self.done = done
        self.undone = undone
        self.done_kind = done_kind
        self.undone_kind = undone_kind


class Docker:
    def __init__(self, mem, cpu, available_time,start_time, kind, doing_task):
        self.mem = mem
        self.cpu = cpu
        self.available_time = available_time
        self.start_time = start_time
        self.kind = kind
        self.doing_task = doing_task



