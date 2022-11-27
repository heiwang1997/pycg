"""
Copyright 2022 by Jiahui Huang. All rights reserved.
This file is part of PyCG toolbox and is released under "MIT License Agreement".
Please see the LICENSE file that should have been included as part of this package.
"""

import os
import sys
import psutil
import importlib
from pathlib import Path


sync_module = None
cached_parsed_path = {}


def init():
    # # Find sync util code: Either at home or at the root of network mounts
    # sync_util_base = None
    #
    # sync_path = Path(os.path.expanduser("~/sync-util"))
    # if sync_path.exists() and sync_path.is_dir():
    #     sync_util_base = sync_path
    # sync_path = Path(os.path.expanduser("~/nas-home/sync-util"))
    # if sync_path.exists() and sync_path.is_dir():
    #     sync_util_base = sync_path

    try:
        sync_util_base = os.environ["JH_UTIL_DIR"]
        sync_util_base = Path(sync_util_base) / "synch"
    except KeyError:
        sync_util_base = Path("/home/huangjh/shared-home/jh-util/synch")

    assert sync_util_base.exists(), "No sync-util is found."

    print("sync-util found in", sync_util_base)
    sys.path.append(str(sync_util_base))

    global sync_module
    sync_module = importlib.import_module("synch")


def is_network_mount(path):
    path = os.path.abspath(path)
    while not os.path.ismount(path):
        path = os.path.dirname(path)

    for partition_data in psutil.disk_partitions(all=True):
        # suppose one with colon is network mount.
        if ":" in partition_data.device or "//" in partition_data.device:
            mount_point = partition_data.mountpoint
            if mount_point == path:
                return True

    return False


def parse_input_path(path_str, do_sync: bool = True, force_remote: bool = False, verbose: bool = True):
    if 'NO_SYNC' in os.environ.keys():
        do_sync = False

    if path_str in cached_parsed_path.keys():
        return cached_parsed_path[path_str]
    else:
        parsed_path = _parse_input_path(path_str, do_sync, force_remote)
        cached_parsed_path[path_str] = parsed_path
        if verbose:
            print(f"{path_str} parsed as {parsed_path}")
        return parsed_path


def _parse_input_path(path_str, do_sync, force_remote):
    if "://" in path_str:
        path_proto, path_str = path_str.split("://")
        if "$" in path_proto:
            path_proto, proto_loc = path_proto.split("$")
        else:
            proto_loc = "remote"
        assert path_proto == "cached", f"Only cached protocol is accepted"
        if force_remote:
            proto_loc = "remote"
        source_path = Path(path_str)
        # source_path.mkdir(parents=True, exist_ok=True)
        source_path = source_path.resolve(strict=False)
        dest_path = sync_module.get_pair_dest_path(source_path)
        assert dest_path is not None, f"{path_str} at {proto_loc} not found."

        source_is_network = is_network_mount(source_path)
        dest_is_network = is_network_mount(dest_path)

        if not source_is_network and not dest_is_network:
            return str(source_path)

        if proto_loc == "remote":
            return str(source_path) if source_is_network else str(dest_path)
        else:
            if source_is_network:
                if source_path.is_dir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                else:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                if do_sync:
                    sync_module.perform_sync(source_path, dest_path, False)
                return str(dest_path)
            else:
                if dest_path.is_dir():
                    source_path.mkdir(parents=True, exist_ok=True)
                else:
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                if do_sync:
                    sync_module.perform_sync(dest_path, source_path, False)
                return str(source_path)
    else:
        return path_str


init()

