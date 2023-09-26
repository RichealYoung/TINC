import subprocess
from subprocess import Popen, PIPE
import os
import sys
import time
from typing import List
import psutil
import pynvml
import time
import copy
import signal
from urllib import request, parse

USER = os.environ['USER']
MIAOCODE = {'yrz':'t9qfTi1',
            'bbncyrz':'t9qfTi1',
            'xtx':'tubrHSG', 
            'cyx':'', 
            'qjy':'',
            'yangrunzhao':''} # tuLGGSK
def reminding(miao_code):
    page = request.urlopen("http://miaotixing.com/trigger?" + parse.urlencode({"id":miao_code, "templ":'pmbXPGS,0,,,,,', "type":"json"}))
    page.read()

UNIT = 1024 * 1024
pynvml.nvmlInit()

stop = False
def my_handler(signum, frame):
    global stop
    stop = True
    print("Stop all compression tasks!")
class Task:
    def __init__(self, command:str, stdout:str='', devide:bool=False):
        self.command = command
        self.stdout = stdout
        self.devide = devide
        self.gpu_use = []
    
    def command_refine(self, batch_compress:bool=False):
        gpu = str(self.gpu_use[0])
        if batch_compress:
            for i in range(1,len(self.gpu_use)):
                gpu += ',' + str(self.gpu_use[i])
            self.command = self.command + ' -g {}'.format(gpu)
        else:
            self.command = f'CUDA_VISIBLE_DEVICES={gpu} ' + self.command
    
    def start(self):
        if self.stdout == '':
            self.p = subprocess.Popen([self.command], shell=True)
        else:
            self.stderr_handler = open(self.stdout, 'w')
            self.p = subprocess.Popen([self.command], shell=True, stdout=self.stderr_handler)
    
    def update(self):
        self.returncode = self.p.poll()
    
    def stop(self):
        self.p.terminate()

class Queue:
    def __init__(self, task_list: List[Task], gpu_list: List[int]=[0]):
        self.pending_list :List[Task] = task_list
        self.running_list :List[Task] = []
        self.finish_list :List[Task] = []
        self.gpu_list = gpu_list
        self.task_len = len(task_list)
        self.gpu_free = []  # gpuMemoryRate < 0.8 and gpuUtilRate
        self.gpu_use = []   # task using
    
    def run(self, batch_compress:bool=False):
        self.find_gpu_free()
        if len(self.gpu_free) == 0:
            return
        # judge the running task
        count = 0
        run_len = len(self.running_list)
        for i in range(run_len):
            task_order = i-count
            task = self.running_list[task_order]
            task.update()
            if task.returncode == None:
                continue
            elif task.returncode == 0:
                self.finish_list.append(task)
                task.stop()
                self.running_list.pop(task_order)
                for g in task.gpu_use:
                    self.gpu_use.remove(g)
                print('Finish: ' + task.command)
                count += 1
            else:
                self.stop()
                raise Exception("Subtask running error!") 
        # update gpu available
        gpu_list = []
        for g in self.gpu_free:
            if not g in self.gpu_use:
                gpu_list.append(g)
        if batch_compress and len(gpu_list)!=0 and len(self.pending_list) > 0:
            print('List of GPUs available: {}'.format(gpu_list))
        # run the pending task
        for i in range(len(gpu_list)):
            if len(self.pending_list) > 0:
                task = self.pending_list.pop(0)
                if batch_compress and task.devide == True:  # if batch compress and devide, use all the gpu
                    for j in range(i, len(gpu_list)):
                        self.gpu_use.append(gpu_list[j])
                        task.gpu_use.append(gpu_list[j])
                    task.command_refine(batch_compress)
                    task.start()
                    print('Command: ' + task.command)
                    self.running_list.append(task)
                    break
                else:
                    self.gpu_use.append(gpu_list[i])
                    task.gpu_use.append(gpu_list[i])
                    task.command_refine(batch_compress)
                    task.start()
                    print('Command: ' + task.command)
                    self.running_list.append(task)


    def start(self, time_interval:float=2, remind:bool=False, batch_compress:bool=False):
        try:
            signal.signal(signal.SIGINT, my_handler)
            while True:
                self.run(batch_compress)
                if len(self.finish_list) == self.task_len or stop:
                    self.stop(remind)
                    break
                time.sleep(time_interval)
        except Exception as e:
            print(e)
            self.stop(remind)

    def find_gpu_free(self):
        self.gpu_free = []
        for g in self.gpu_list:
            handle = pynvml.nvmlDeviceGetHandleByIndex(g)
            memoryInfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpuMemoryRate = memoryInfo.used/memoryInfo.total
            gpuUtilRate = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu

            # if gpuMemoryRate < 0.8 and gpuUtilRate < 0.5:
            if gpuMemoryRate < 0.8:
                self.gpu_free.append(g)
                
        return self.gpu_free
    
    def stop(self, remind=True):
        all_task_list = self.pending_list + self.running_list + self.finish_list
        for task in all_task_list:
            task.stop()
        if remind:
            try:
                miao_code = MIAOCODE[USER]
                reminding(miao_code)
            except:
                print("can't connect to miao_reminding or miao_code not exist")


if __name__=="__main__":
    task_list = [
        Task('build/BRIEF -p data/config.json', 'outputs/HiPCT_2022_0818_170705/stdout/0_64_0_64_0_64.log'),
        Task('build/BRIEF -p data/config.json', 'outputs/HiPCT_2022_0818_170705/stdout/0_64_0_64_64_128.log'),
        Task('build/BRIEF -p data/config.json', 'outputs/HiPCT_2022_0818_170705/stdout/0_64_64_128_0_64.log'),
        Task('build/BRIEF -p data/config.json', 'outputs/HiPCT_2022_0818_170705/stdout/0_64_64_128_64_128.log'),
        Task('build/BRIEF -p data/config.json', 'outputs/HiPCT_2022_0818_170705/stdout/0_128_0_128_128_256.log'),
        Task('build/BRIEF -p data/config.json', 'outputs/HiPCT_2022_0818_170705/stdout/0_128_128_256_0_128.log')
    ]

    gpu_list = [2,3]
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in gpu_list])
    queue = Queue(task_list, gpu_list)
    queue.start(time_interval=2, remind=False)