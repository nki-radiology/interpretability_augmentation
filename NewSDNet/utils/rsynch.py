### Not used in the end, but might be useful in the future ###

import hashlib
from pathlib import Path
import subprocess
from typing import List, Optional


def rsync_data(source_dir: str, destination_dir: str, paths: Optional[List] = None):
    """Rsyncs data from source to destination.

    Mainly used in the datamodule to transfer data to the scratch disk.

    Args:
        source_dir (str): root_dir from which to rsync. If no paths are given, everything from root dir is copied.
        destination_dir (str): new root_dir to rsync to. Will likely be /processing/<user.name> on RHPC
        paths (Optional[List], optional): list of relative file paths to rsync, which will be relative to the source_dir
    """
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    assert source_dir.is_dir(), f"Source dir {source_dir} does not exist."
    if not destination_dir.is_dir():
        print(f"Destination dir {destination_dir} does not exist, creating it.")
        destination_dir.mkdir(parents=True)
    assert source_dir != destination_dir, "Source and destination dirs are the same."

    if paths is not None:
        raise NotImplementedError("Rsyncing specific paths is not yet implemented.")
    else:
        try:
            if str(source_dir)[-1] != "/":
                source_dir = str(source_dir) + "/"
            source_directory_hash = hashlib.sha256(str(source_dir).encode()).hexdigest()
            destination_dir = destination_dir / source_directory_hash
            print(f"Rsyncing {source_dir} to {destination_dir}")
            subprocess.run(
                ["rsync", "-aq", source_dir, destination_dir],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            print(f"Rsyncing complete.")
            return destination_dir
        except Exception as e:
            print(f"Rsync failed with error code {e.returncode}.")
            print(f"Error message: {e.stderr.decode()}")
