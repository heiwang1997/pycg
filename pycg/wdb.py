import wandb
import os.path
import re
from datetime import datetime

import flatten_dict
from pathlib import Path
from pycg.exp import ConsoleColor


def _rebuild_cfg(d):
    if isinstance(d, dict):
        return {k: _rebuild_cfg(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_rebuild_cfg(t) for t in d]
    elif isinstance(d, str):
        if d == 'None':
            return None
        elif len(d) > 0 and d[0] == '{':
            # Deeper keys are collapsed. We need to rescue them:
            return eval(d)
        return d
    else:
        return d


def recover_from_wandb_config(wandb_dict):
    f_dict = flatten_dict.unflatten(wandb_dict, splitter='path')
    f_dict = _rebuild_cfg(f_dict)
    return f_dict


def get_epoch_checkpoints(ckpt_base: Path, match_case: str = r"epoch=(\d+)"):
    all_files = list(ckpt_base.glob("*.ckpt"))
    all_ckpts = {}
    for f in all_files:
        match_res = re.match(match_case, f.name)
        if match_res is not None:
            all_ckpts[int(match_res[1])] = f
    return all_ckpts


def get_wandb_run(wdb_url: str, wdb_base: str, default_ckpt: str = "all"):
    """
    Retrieves a Weights & Biases run and its associated checkpoint based on a URL-like identifier.
    
    This function parses a W&B URL to locate a specific run and optionally retrieve a checkpoint
    from that run. It supports both run ID and run name based lookups, with various checkpoint
    selection modes. 
    Note that for the latter case, multiple runs can be returned and user is prompted to select one.
    If ckpt-id is provided, possible path to the checkpoints will also be provided.

    Args:
        wdb_url (str): URL-like identifier for the W&B run. Can be in two formats:
            - [wdb:]<USER-NAME>/<PROJ-NAME>/<RUN-ID>[:ckpt-id]
            - [wdb:]<USER-NAME>/<PROJ-NAME>/<RUN-NAME>[:ckpt-id]
        wdb_base (str): Base directory where W&B checkpoints are stored locally
        default_ckpt (str, optional): Default checkpoint mode if not specified in URL. 
            Defaults to "all". Valid modes include:
            - "last": Latest checkpoint
            - "best"/"epoch": Checkpoint with lowest validation loss
            - "all": Return checkpoint directory
            - "test_auto": Automatically determine best checkpoint
            - "<number>": Specific epoch number

    Returns:
        tuple: (wandb.Run, Path)
            - wandb.Run: The retrieved W&B run object
            - Path: Path to the selected checkpoint file or directory.
              Returns None if no checkpoint was requested.

    Raises:
        FileNotFoundError: If the requested checkpoint cannot be found
        NotImplementedError: If an invalid checkpoint mode is specified
        AssertionError: If no matching run is found or checkpoint selection fails
    """
    # Strip 'wdb:' prefix if present
    if wdb_url.startswith("wdb:"):
        wdb_url = wdb_url[4:]

    # Split checkpoint name from URL if present
    if ":" in wdb_url:
        wdb_url, ckpt_name = wdb_url.split(":")
    else:
        ckpt_name = default_ckpt

    # Parse URL components
    wdb_url = wdb_url.split("/")
    user_name = wdb_url[0]
    project_name = wdb_url[1]
    other_name = "/".join(wdb_url[2:])

    # Try to fetch run - either by ID or by name
    if len(other_name) == 8 and "/" not in other_name and "_" not in other_name and "-" not in other_name:
        # Looks like a run ID (8 chars, no special characters), just fetch it
        run_id = f"{user_name}/{project_name}/{other_name}"
        wdb_run = wandb.Api().run(run_id)
    else:
        # Assume it's a run name and search for matches
        all_runs = wandb.Api().runs(
            f"{user_name}/{project_name}",
            filters={"display_name": other_name}
        )
        assert len(all_runs) > 0, f"No run is found for name {other_name}"
        
        # Handle multiple matching runs
        sel_idx = 0
        if len(all_runs) > 1:
            print("\nMore than 1 runs found")
            for rid, run in enumerate(all_runs):
                print(f"   {rid}: {run}")
            sel_idx = int(input("Please select one:"))
        wdb_run = all_runs[sel_idx]
        print("Target run", wdb_run)

    # Return early if no checkpoint was requested
    if ckpt_name is None:
        return wdb_run, None

    # Construct base path for checkpoints
    ckpt_base = Path(f"{wdb_base}/{wdb_run.project}/{wdb_run.id}/checkpoints/")

    # Handle different checkpoint selection modes
    if ckpt_name == "last":
        # Check if last checkpoint exists and verify its timestamp
        ckpt_path = ckpt_base / "last.ckpt"
        assert ckpt_path.exists(), f"{ckpt_path} not exist!"
        existing_mtime = os.path.getmtime(ckpt_path)
        last_mtime = wdb_run.history(keys=['val_loss', '_timestamp'], samples=1000).iloc[-1]._timestamp
        if last_mtime - existing_mtime > 600:       # 10min threshold
            print(f'{ConsoleColor.YELLOW} Warning: The existing "last" checkpoint has timestamp '
                  f'{datetime.fromtimestamp(existing_mtime).isoformat()}, '
                  f'but actually the newest one should have timestamp '
                  f'{datetime.fromtimestamp(last_mtime).isoformat()}! {ConsoleColor.RESET}')

    elif ckpt_name == "best" or ckpt_name == "epoch":
        epoch_ckpts = get_epoch_checkpoints(ckpt_base)
        loss_table = wdb_run.history(keys=['val_loss', 'epoch'], samples=1000)
        loss_table.set_index('epoch', inplace=True)
        best_epoch_idx = loss_table['val_loss'].astype(float).argmin().item()
        if best_epoch_idx in epoch_ckpts:
            ckpt_path = epoch_ckpts[best_epoch_idx]
        else:
            print(f'{ConsoleColor.YELLOW} Best epoch is {best_epoch_idx} but not found, existed: {list(epoch_ckpts.keys())} ')
            print(loss_table)
            print('')
            raise FileNotFoundError

    elif ckpt_name == "all":
        ckpt_path = ckpt_base

    elif ckpt_name == "test_auto":
        # Try to load last, if best and last do not align, then prompt
        epoch_ckpts = get_epoch_checkpoints(ckpt_base)
        if len(epoch_ckpts) == 0:
            print(f'{ConsoleColor.YELLOW} Please check the folder of {ckpt_base}, {list(ckpt_base.glob("*.ckpt"))} {ConsoleColor.RESET}')
            raise FileNotFoundError
        existing_last_epoch_idx = max(list(epoch_ckpts.keys()))
        loss_table = wdb_run.history(keys=['val_loss', 'epoch'], samples=1000)
        loss_table.set_index('epoch', inplace=True)
        best_epoch_idx = loss_table['val_loss'].argmin().item()
        last_epoch_idx = loss_table.iloc[-1].name.item()
        if existing_last_epoch_idx == best_epoch_idx == last_epoch_idx:
            print(f'{ConsoleColor.YELLOW} Warning: ckpt not specified, but {last_epoch_idx} (being both last and best) '
                  f'seems to be a valid choice. {ConsoleColor.RESET}')
            ckpt_path = epoch_ckpts[existing_last_epoch_idx]
        else:
            print(f'{ConsoleColor.YELLOW} Cannot determine correct ckpt_idx! Best = {best_epoch_idx}, Last = {last_epoch_idx}, '
                  f'Existing Last = {existing_last_epoch_idx} {ConsoleColor.RESET}')
            assert False

    elif ckpt_name.isdigit():
        epoch_idx = int(ckpt_name)
        epoch_ckpts = get_epoch_checkpoints(ckpt_base)
        if epoch_idx in epoch_ckpts:
            ckpt_path = epoch_ckpts[epoch_idx]
        else:
            print(f'{ConsoleColor.YELLOW} Epoch {epoch_idx} not found, existed: {list(epoch_ckpts.keys())} {ConsoleColor.RESET}')
            raise FileNotFoundError

    else:
        raise NotImplementedError

    return wdb_run, ckpt_path
