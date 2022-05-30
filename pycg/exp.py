import sys
import re
from collections import OrderedDict, defaultdict
import numpy as np
import argparse
import yaml
import json
import pdb
import time
import pickle
import traceback
from pathlib import Path
from typing import Tuple, Union
from omegaconf import OmegaConf
import os
from contextlib import ContextDecorator

# if 'MEM_PROFILE' in os.environ.keys():
#     from pytorch_memlab.line_profiler.profile import global_line_profiler
#
#     global_line_profiler.disable()

# Load and cache PT_PROFILE key
enable_pt_profile = 'PT_PROFILE' in os.environ.keys()


def parse_config_json(json_path: Path, args: argparse.Namespace = None):
    """
    Parse a json file and add key:value to args namespace.
    Json file format [ {attr}, {attr}, ... ]
        {attr} = { "_": COMMENT, VAR_NAME: VAR_VALUE }
    """
    if args is None:
        args = argparse.Namespace()

    with json_path.open() as f:
        json_text = f.read()

    try:
        raw_configs = json.loads(json_text)
    except:
        # Do some fixing of the json text
        json_text = json_text.replace("\'", "\"")
        json_text = json_text.replace("None", "null")
        json_text = json_text.replace("False", "false")
        json_text = json_text.replace("True", "true")
        raw_configs = json.loads(json_text)

    if isinstance(raw_configs, dict):
        raw_configs = [raw_configs]
    configs = {}
    for raw_config in raw_configs:
        for rkey, rvalue in raw_config.items():
            if rkey != "_":
                configs[rkey] = rvalue

    if configs is not None:
        for ckey, cvalue in configs.items():
            args.__dict__[ckey] = cvalue
    return args


def parse_config_yaml(yaml_path: Path, args: Union[argparse.Namespace, OmegaConf] = None,
                      override: bool = True) -> OmegaConf:
    if args is None:
        args = OmegaConf.create()
    if isinstance(args, argparse.Namespace):
        args = OmegaConf.create(args.__dict__)

    configs = OmegaConf.load(yaml_path)
    if "include_configs" in configs:
        base_configs = configs["include_configs"]
        del configs["include_configs"]
        if isinstance(base_configs, str):
            base_configs = [base_configs]
        # Update the config from top to down.
        base_cfg = OmegaConf.create()
        for base_config in base_configs:
            base_config_path = yaml_path.parent / Path(base_config)
            base_cfg = parse_config_yaml(base_config_path, base_cfg)
        configs = OmegaConf.merge(base_cfg, configs)

    if "assign" in configs:
        overlays = configs["assign"]
        del configs["assign"]
        assign_config = OmegaConf.from_dotlist([f"{k}={v}" for k, v in overlays.items()])
        configs = OmegaConf.merge(configs, assign_config)

    if override:
        return OmegaConf.merge(args, configs)
    else:
        return OmegaConf.merge(configs, args)


def dict_to_args(data, recursive: bool = False):
    args = argparse.Namespace()
    if hasattr(data, '__dict__'):
        # This enables us to also process namespace.
        data = data.__dict__
    for ckey, cvalue in data.items():
        if recursive:
            if isinstance(cvalue, dict):
                cvalue = dict_to_args(cvalue, recursive)
        args.__dict__[ckey] = cvalue
    return args


class ArgumentParserX(argparse.ArgumentParser):
    def __init__(self, base_config_path=None, add_hyper_arg=True, to_oconf=True, **kwargs):
        super().__init__(**kwargs)
        self.add_hyper_arg = add_hyper_arg
        self.base_config_path = base_config_path
        self.to_oconf = to_oconf
        if self.add_hyper_arg:
            self.add_argument('hyper', type=str, help='Path to the yaml parameter')
        self.add_argument('--exec', type=str, nargs='+', help='Extract code to modify the args')

    @staticmethod
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentError('Boolean value expected.')

    def parse_args(self, args=None, namespace=None, additional_args=None):
        # Parse arg for the first time to extract args defined in program.
        _args = self.parse_known_args(args, namespace)[0]
        # Add the types needed.
        file_args = OmegaConf.create()
        if self.base_config_path is not None:
            file_args = parse_config_yaml(Path(self.base_config_path), file_args)
        if additional_args is not None:
            file_args = OmegaConf.merge(file_args, additional_args)
        if self.add_hyper_arg and _args.hyper != "none":
            if _args.hyper.endswith("json"):
                file_args = parse_config_json(Path(_args.hyper), file_args)
            else:
                file_args = parse_config_yaml(Path(_args.hyper), file_args)
        for ckey, cvalue in file_args.items():
            try:
                if isinstance(cvalue, bool):
                    self.add_argument(*(["--" + ckey] if ckey != "visualize" else ['-v', '--' + ckey]),
                                      type=ArgumentParserX.str2bool, nargs='?',
                                      const=True, default=cvalue)
                else:
                    self.add_argument('--' + ckey, type=type(cvalue), default=cvalue, required=False)
            except argparse.ArgumentError:
                continue
        # Parse args fully to extract all useful information
        _args = super().parse_args(args, namespace)
        if self.to_oconf:
            _args = OmegaConf.create(_args.__dict__)
        # After that, execute exec part.
        exec_code = _args.exec
        if exec_code is not None:
            for exec_cmd in exec_code:
                exec_cmd = "_args." + exec_cmd.strip()
                exec(exec_cmd)
        return _args


class TorchLossMeter:
    """
    Weighted loss calculator, for tracing all the losses generated and print them.
    """
    def __init__(self):
        self.loss_dict = {}

    def add_loss(self, name, loss, weight=1.0):
        if weight == 0.0:
            return
        if hasattr(loss, "numel"):
            assert loss.numel() == 1, f"Loss must contains only one item, instead of {loss.numel()}."
        assert name not in self.loss_dict.items(), f"{name} already in loss!"
        self.loss_dict[name] = (weight, loss)

    def get_sum(self):
        import torch
        for n, (w, l) in self.loss_dict.items():
            if isinstance(l, torch.Tensor) and torch.isnan(l):
                print(f"Warning: Loss {n} with weight {w} has NaN loss!")
            # Disabled because this can also be used during validation/testing.
            # if l.grad_fn is None:
            #     print(f"Warning: Loss {n} with value {l} does not have grad_fn!")
        sum_arr = [w * l for (w, l) in self.loss_dict.values()]
        return sum(sum_arr)

    def items(self):
        # Standard iterator
        for n, (w, l) in self.loss_dict.items():
            yield n, w * l

    def get_printable_mean(self):
        text = "\033[94m"
        for n, (w, l) in self.loss_dict.items():
            text += "(%s: %.2f * %.4f = %.4f) " % (n, w, l, w * l)
        text += " sum = %.4f" % self.get_sum()
        return text + '\033[0m'


class AverageMeter:
    """
    Maintain named lists of numbers. Compute their average to evaluate dataset statistics.
    This can not only used for loss, but also for progressive training logging, supporting import/export data.
    """
    def __init__(self):
        self.loss_dict = OrderedDict()

    def export(self, f):
        if isinstance(f, str):
            f = open(f, 'wb')
        pickle.dump(self.loss_dict, f)

    def load(self, f):
        if isinstance(f, str):
            f = open(f, 'rb')
        self.loss_dict = pickle.load(f)
        return self

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            # loss_val = float(loss_val)
            # if np.isnan(loss_val):
            #     continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: [loss_val]})
            else:
                self.loss_dict[loss_name].append(loss_val)

    def get_mean_loss_dict(self):
        loss_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            loss_dict[loss_name] = sum(loss_arr) / len(loss_arr)
        return loss_dict

    def get_mean_loss(self):
        mean_loss_dict = self.get_mean_loss_dict()
        if len(mean_loss_dict) == 0:
            return 0.0
        else:
            return sum(mean_loss_dict.values()) / len(mean_loss_dict)

    def get_printable_mean(self):
        text = ""
        all_loss_sum = 0.0
        for loss_name, loss_mean in self.get_mean_loss_dict().items():
            all_loss_sum += loss_mean
            text += "(%s:%.4f) " % (loss_name, loss_mean)
        text += " sum = %.4f" % all_loss_sum
        return text

    def get_newest_loss_dict(self, return_count=False):
        loss_dict = {}
        loss_count_dict = {}
        for loss_name, loss_arr in self.loss_dict.items():
            if len(loss_arr) > 0:
                loss_dict[loss_name] = loss_arr[-1]
                loss_count_dict[loss_name] = len(loss_arr)
        if return_count:
            return loss_dict, loss_count_dict
        else:
            return loss_dict

    def get_printable_newest(self):
        nloss_val, nloss_count = self.get_newest_loss_dict(return_count=True)
        return ", ".join([f"{loss_name}[{nloss_count[loss_name] - 1}]: {nloss_val[loss_name]}"
                          for loss_name in nloss_val.keys()])

    def print_format_loss(self, color=None):
        if hasattr(sys.stdout, "terminal"):
            color_device = sys.stdout.terminal
        else:
            color_device = sys.stdout
        if color == "y":
            color_device.write('\033[93m')
        elif color == "g":
            color_device.write('\033[92m')
        elif color == "b":
            color_device.write('\033[94m')
        print(self.get_printable_mean(), flush=True)
        if color is not None:
            color_device.write('\033[0m')


class RunningAverageMeter:
    """
    new_mean = alpha * old_mean + (1 - alpha) * cur_value
        - the smaller alpha is, the more prone to change according to newest value.
    # TODO: Just merge this as a function get_running_loss_dict() / get_printable_running().
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.loss_dict = OrderedDict()

    def append_loss(self, losses):
        for loss_name, loss_val in losses.items():
            if loss_val is None:
                continue
            loss_val = float(loss_val)
            if np.isnan(loss_val):
                continue
            if loss_name not in self.loss_dict.keys():
                self.loss_dict.update({loss_name: loss_val})
            else:
                old_mean = self.loss_dict[loss_name]
                self.loss_dict[loss_name] = self.alpha * old_mean + (1 - self.alpha) * loss_val

    def get_loss_dict(self):
        return {k: v for k, v in self.loss_dict.items()}


class AnalysisEnv:
    """
    Print debug info if exceptions are triggered w/o handling.
    """
    def __init__(self, *exceptions, enabled: bool = True):
        self.exceptions = exceptions
        self.enabled = enabled
        if len(self.exceptions) == 0:
            self.exceptions = [Exception, RuntimeError]

    def __enter__(self):
        from pytorch_memlab.line_profiler.profile import global_line_profiler

        if self.enabled:
            global_line_profiler.enable()

    def __exit__(self, exc_type, exc_val, exc_tb):
        from pytorch_memlab.line_profiler.profile import global_line_profiler

        if any([isinstance(exc_val, cls) for cls in self.exceptions]):
            traceback.print_exc()
            pdb.post_mortem(exc_tb)
            sys.exit(0)
        global_line_profiler.disable()


class Timer:
    
    def __init__(self, enabled: bool = True, cuda_sync: bool = False, color: str = "yellow"):
        self.enabled = enabled
        self.cuda_sync = cuda_sync
        self.time_names = ["Timer Created"]
        self.time_points = [time.perf_counter()]
        self.report_up_to_date = False
        self.console_color = {
            "yellow": '\033[93m',
            "blue": '\033[94m'
        }[color]
        self.forbid_report_on_exit = False

    def toc(self, name):
        if not self.enabled:
            return

        if self.cuda_sync:
            import torch
            torch.cuda.synchronize()

        self.time_points.append(time.perf_counter())
        self.time_names.append(name)
        self.report_up_to_date = False

    def report(self, merged=False):
        if self.enabled:
            sys.stdout.write(self.console_color)
            print("==========  TIMER  ==========")
            if not merged:
                for time_i in range(len(self.time_names) - 1):
                    print(f"{self.time_names[time_i]} --> {self.time_names[time_i + 1]}: "
                          f"{self.time_points[time_i + 1] - self.time_points[time_i]}s")
            else:
                merged_times = defaultdict(list)
                for time_i in range(len(self.time_names) - 1):
                    interval_name = f"{self.time_names[time_i]} --> {self.time_names[time_i + 1]}"
                    interval_time = self.time_points[time_i + 1] - self.time_points[time_i]
                    merged_times[interval_name].append(interval_time)
                for tname, tarr in merged_times.items():
                    print(f"{tname}: {np.mean(tarr)} +/- {np.std(tarr)}s")
            print("=============================")
            sys.stdout.write('\033[0m')
        self.report_up_to_date = True
        if merged:
            self.forbid_report_on_exit = True

    def toc_and_report(self, name, merged=False):
        self.toc(name)
        self.report(merged=merged)

    def __del__(self):
        if not self.report_up_to_date and not self.forbid_report_on_exit:
            self.toc_and_report("Timer Deleted")


class TimerCollections:
    def __init__(self):
        self.timers = {}
        self.is_persistent = {}

    def enable(self, name, cuda_sync: bool, persistent: bool = False):
        """
        Enable a timer, with global visibility.
        :param name: name of the timer
        :param cuda_sync: whether to sync cuda
        :param persistent: whether to really delete the timer when finalize.
        :return:
        """
        if name in self.timers.keys():
            assert persistent
        else:
            self.timers[name] = Timer(cuda_sync=cuda_sync)
            self.timers[name].forbid_report_on_exit = True
            self.is_persistent[name] = persistent
        self.timers[name].toc("activated")

    def finalize(self, name, merged=False):
        if name in self.timers.keys():
            self.timers[name].toc_and_report("Finalize", merged=merged)
            if not self.is_persistent[name]:
                del self.timers[name]
                del self.is_persistent[name]

    def toc(self, name, message):
        if name in self.timers.keys():
            self.timers[name].toc(message)

    def report(self, name):
        self.timers[name].report(merged=True)

global_timers = TimerCollections()


def natural_time(elapsed):
    if elapsed > 1.0:
        return f"{elapsed:.3f}s"
    else:
        return f"{elapsed * 1000:.3f}ms"


def performance_counter(n_iter: int = 1):
    """
    Usage:
    for _ in performance_counter(5):
        some_code()
    :param n_iter: number of iterations
    """
    all_times = []
    for i_iter in range(n_iter):
        try:
            start_time = time.perf_counter()
            yield i_iter
            all_times.append(time.perf_counter() - start_time)
        except Exception as e:
            print(e)
    print(f" + Perf Success {len(all_times)} / {n_iter}, "
          f"time = {natural_time(np.mean(all_times))} +/- {natural_time(np.std(all_times))}")


def profile(func):
    """
    Usage: (for line profiling)
        - Add a `@pycg.exp.profile` to the function you want to test
        - Run: `kernprof -l main.py`
        - Check Result: `python -m line_profiler 1.lprof`
    """
    import builtins
    # Detect kernprof line-profiler environment
    if "profile" in builtins.__dict__:
        print("Kernprof function added!")
        return builtins.__dict__["profile"](func)
    else:
        return func


class pt_profile_named(ContextDecorator):
    """
    Pytorch Profiler utility usage:
     - Annotate function with `@pycg.exp.pt_profile` or label a block using 'with pycg.exp.pt_profile_named('NAME')'
     - Run with ENV variable 'PT_PROFILE' set. (if it is 1, then only cpu profile, 2 is cuda profile)
    """

    profiler = None

    def __init__(self, name: str, trace_file: str = "pt_profile.json"):
        self.name = name
        self.trace_file = trace_file
        self.tagger = None
        self.is_top = False

    def __enter__(self):
        if not enable_pt_profile:
            return

        from torch.profiler import profile, record_function, ProfilerActivity

        if pt_profile_named.profiler is None:
            self.is_top = True
            if int(os.environ['PT_PROFILE']) > 1:
                act = [ProfilerActivity.CUDA, ProfilerActivity.CPU]
            else:
                act = [ProfilerActivity.CPU]
            pt_profile_named.profiler = profile(activities=act, record_shapes=True)
            pt_profile_named.profiler.__enter__()

        self.tagger = record_function(self.name)
        self.tagger.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not enable_pt_profile:
            return

        self.tagger.__exit__(exc_type, exc_val, exc_tb)
        if self.is_top:
            pt_profile_named.profiler.__exit__(exc_type, exc_val, exc_tb)
            pt_profile_named.profiler.export_chrome_trace(self.trace_file)
            pt_profile_named.profiler = None

        del self.tagger


def pt_profile(func):
    assert not isinstance(func, str), "Please use pt_profile_named."
    if enable_pt_profile:
        def new_func(*args, **kwargs):
            with pt_profile_named(func.__name__):
                return func(*args, **kwargs)
        return new_func
    else:
        return func


def mem_profile(func):
    """
    Usage:
        - Add a `@pycg.exp.mem_profile' to the function you want to test
        - Run the script with environment variable 'MEM_PROFILE=1' set.
        - When the program ends, it will print the profiling result.
    """
    if 'MEM_PROFILE' in os.environ.keys():
        from pytorch_memlab import profile
        return profile(func)
    else:
        return func


def memory_usage(tensor):
    import torch
    if isinstance(tensor, torch.Tensor):
        size_mb = tensor.element_size() * tensor.nelement() / 1024 / 1024
        return f"Torch tensor {list(tensor.size())}, memory = {size_mb:.2f}MB."
    else:
        return "Memory usage not supported."


def deterministic_hash(data):
    """
    :param data: Any type
    :return: a deterministic hash value of integer type (32bit)
    """
    import zlib
    jval = json.dumps(data, ensure_ascii=False, sort_keys=True,
                      indent=None, separators=(',', ':'))
    return zlib.adler32(jval.encode('utf-8'))


# GPU monitoring stuff
class ComputeDevice:
    def __init__(self):
        self.server_name = None
        self.gpu_id = None
        self.gpu_mem_usage = None
        self.gpu_compute_usage = None
        self.processes = []

    def __repr__(self):
        return f"{self.server_name}-GPU-{self.gpu_id}: Mem: {self.gpu_mem_usage * 100:.2f}%, " \
               f"Util: {self.gpu_compute_usage * 100:.2f}%"


class ComputeProcess:
    def __init__(self):
        self.server_name = None
        self.gpu_id = None
        self.pid = None
        self.cwd = None


def get_gpu_status(server_name, get_process_info: bool = False, use_nvml: bool = True):
    import subprocess

    def run_command(cmd):
        if cmd[0] == 'ssh' and cmd[1] == 'localhost':
            cmd = cmd[2:]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = result.stdout
        return result.decode('utf-8')

    # Use nvml if possible
    all_devs = []
    if server_name == 'localhost' and use_nvml and not get_process_info:
        import pynvml
        pynvml.nvmlInit()
        for gpu_id in range(pynvml.nvmlDeviceGetCount()):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            except pynvml.NVMLError_GpuIsLost:
                print(f"Warning: GPU {gpu_id} is lost.")
                continue
            cur_dev = ComputeDevice()
            cur_dev.server_name = server_name
            cur_dev.gpu_id = gpu_id
            cur_dev.gpu_compute_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu / 100
            cur_dev.gpu_mem_usage = pynvml.nvmlDeviceGetUtilizationRates(handle).memory / 100
            all_devs.append(cur_dev)
        pynvml.nvmlShutdown()
        proc_info = []
    else:
        nv_output = run_command(['ssh', server_name, 'nvidia-smi'])
        nv_info = re.findall(r'(\d+)MiB / (\d+)MiB.*?(\d+)%', nv_output)
        proc_info = re.findall(r'(\d+).*?N/A.*?(\d+)\s+[CG]', nv_output)
        for gpu_id, (cur_mem, all_mem, cur_util) in enumerate(nv_info):
            cur_dev = ComputeDevice()
            cur_dev.server_name = server_name
            cur_dev.gpu_id = gpu_id
            cur_dev.gpu_mem_usage = int(cur_mem) / int(all_mem)
            cur_dev.gpu_compute_usage = int(cur_util) / 100
            all_devs.append(cur_dev)

    # Get current working directory...
    proc_cwds = {}
    if get_process_info:
        cwd_output = run_command(['ssh', server_name, 'pwdx', ' '.join([t[1] for t in proc_info])])
        proc_cwds_list = cwd_output.strip()
        if proc_cwds_list:
            for p in proc_cwds_list.split('\n'):
                colon_pos = p.find(': ')
                proc_cwds[int(p[:colon_pos])] = p[colon_pos + 2:]

    for proc_gpu, proc_pid in proc_info:
        proc_gpu = int(proc_gpu)
        proc_pid = int(proc_pid)
        cur_proc = ComputeProcess()
        cur_proc.server_name = server_name
        cur_proc.gpu_id = proc_gpu
        cur_proc.pid = proc_pid
        cur_proc.cwd = proc_cwds.get(proc_pid, None)
        all_devs[proc_gpu].processes.append(cur_proc)

    return all_devs
