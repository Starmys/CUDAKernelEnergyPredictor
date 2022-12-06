import time
import threading

import pynvml
import numpy as np


class GPUSampler(threading.Thread):

    MB2BYTE = 1024 ** 2
    GB2BYTE = 1024 ** 3

    def __init__(self, gpu_id = 0, sampling_interval = 0.05, verbose = False):
        threading.Thread.__init__(self)
        if sampling_interval < 0.02:
            print('Error: The minimum sampling interval supported is 0.02s (20ms).')
            return
        # setup nvml
        pynvml.nvmlInit()
        total_devices = pynvml.nvmlDeviceGetCount()
        if gpu_id >= total_devices:
            print('Error: invalid gpu_id, total number of gpu is {}'.format(total_devices))
            pynvml.nvmlShutdown()
            return
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        if verbose:
            current_idx = pynvml.nvmlDeviceGetIndex(self.handle)
            name = pynvml.nvmlDeviceGetName(self.handle)
            print('Created sampler for gpu #{}, name: {}'.format(current_idx, name))
        # member variables
        self.sampling_interval = sampling_interval # sampling period, in second
        self.running = False
        self.count = 0
        self.used_memory_readings = [] # used memory readings at sampling time, MB
        self.power_readings = [] # power readings at sampling time, W
        self.temperature_readings = [] # temperature readings at sampling time, C
        self.mem_util_readings = [] # memory util readings at sampling time, % of time
        self.gpu_util_readings = [] # gpu util readings at sampling time, % of time
        self.time_stamps = [] # time stamp of each reading wrt starting time, s
    
    def __del__(self):
        pynvml.nvmlShutdown()
    
    def run(self):
        start_t = time.perf_counter()
        self.running = True
        while self.running:
            # start timer
            begin_time = time.perf_counter()
            # read from nvml
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self.handle)
            temperature = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            # append to list
            # self.used_memory_readings.append(meminfo.used / GPUSampler.MB2BYTE)
            self.used_memory_readings.append(meminfo.used / meminfo.total * 100)
            self.power_readings.append(power_mw / 1e3)
            self.temperature_readings.append(temperature)
            self.mem_util_readings.append(util.memory)
            self.gpu_util_readings.append(util.gpu)
            self.time_stamps.append(time.perf_counter() - start_t)
            self.count += 1
            # end timer
            end_time = time.perf_counter()
            # sleep
            sleep_time = self.sampling_interval - (end_time - begin_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def terminate(self):
        self.running = False

    def calculate_energy(self):
        # total_energy = sum(self.power_readings) * self.sampling_interval
        total_energy = np.trapz(self.power_readings, self.time_stamps)
        return total_energy
    
    def export_readings(self):
        return {
            'used_memory' : self.used_memory_readings,
            'power' : self.power_readings,
            'temperature' : self.temperature_readings,
            'memory_util' : self.mem_util_readings,
            'gpu_util' : self.gpu_util_readings,
            'time_stamp' : self.time_stamps,
            'sampling_interval' : self.sampling_interval,
        }
