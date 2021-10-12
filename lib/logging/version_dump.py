import os
import subprocess
from lib.config.config import config

def version_dump():
    os.makedirs(config.VERSION_DUMP_PATH, exist_ok=True)

    diff_path = os.path.join(config.VERSION_DUMP_PATH, 'DIFF')
    tracked_list = os.path.join(config.VERSION_DUMP_PATH, 'TRACKED')
    head_hash_path = os.path.join(config.VERSION_DUMP_PATH, 'HEAD_HASH')
    stash_hash_path = os.path.join(config.VERSION_DUMP_PATH, 'STASH_W_MODIFIED_HASH')
    zip_tracked = os.path.join(config.VERSION_DUMP_PATH, 'tracked.zip')
    zip_untracked = os.path.join(config.VERSION_DUMP_PATH, 'untracked.zip')

    # Determine hash of HEAD
    cmd = ['git', 'rev-list', '-n1', 'HEAD']
    p = subprocess.run(cmd, check=True, capture_output=True)
    head_hash = p.stdout.decode('utf-8').split('\n')[0]
    with open(head_hash_path, 'w') as f:
        f.writelines([head_hash])

    # Create a new "hidden" commit including all modifications of tracked files
    cmd = ['git', 'stash', 'create']
    p = subprocess.run(cmd, check=True, capture_output=True)
    stash_hash = p.stdout.decode('utf-8').split('\n')[0]
    with open(stash_hash_path, 'w') as f:
        f.writelines([stash_hash])

    # Store diff of working tree changes, including index, against last commit
    cmd = ['git', 'diff', 'HEAD']
    with open(diff_path, 'w') as f:
        subprocess.run(cmd, check=True, stdout=f)

    # Store list of untracked files
    cmd = ['git', 'ls-files', '--others', '--exclude-standard']
    with open(tracked_list, 'w') as f:
        subprocess.run(cmd, check=True, stdout=f)

    # Archive untracked files
    cmd = ['xargs', '-a', tracked_list, 'zip', zip_untracked]
    p = subprocess.run(cmd, check=True)

    # Archive tracked files
    cmd = ['git', 'archive', '-o', zip_tracked, stash_hash]
    p = subprocess.run(cmd, check=True)
