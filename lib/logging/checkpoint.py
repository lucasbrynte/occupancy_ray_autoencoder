import os
from collections import OrderedDict
import torch
from lib.config.config import config

def serialize_checkpoint_metadata(d):
    return '__'.join([key + '_' + val for key, val in d.items()])

def deserialize_checkpoint_metadata(s):
    def split_keys_from_vals(keyval):
        key, val = keyval.split('_')
        return key, val
    return OrderedDict(map(split_keys_from_vals, s.split('__')))

def find_latest_checkpoint_file(checkpoint_load_dir):
    serialized_checkpoint_names = [fname for fname in os.listdir(checkpoint_load_dir) if fname.lower() not in ['.ds_store', 'thumbs.db', 'desktop.ini']]
    epoch_list = [int(deserialize_checkpoint_metadata(fname)['epoch']) for fname in serialized_checkpoint_names]
    fname_latest, latest_epoch = max(zip(serialized_checkpoint_names, epoch_list), key=lambda x: x[1])
    config.CHECKPOINT_LOAD_PATH = os.path.join(config.OLD_EXP_DIR, 'checkpoints', fname_latest)
    assert os.path.exists(config.CHECKPOINT_LOAD_PATH)
    return fname_latest, latest_epoch

def save_checkpoint(
    checkpoint_path,
    epoch,
    n_samples_processed,
    occ_ray_encoder,
    occ_ray_decoder,
    optimizer,
):
    torch.save(
        {
            'epoch': epoch,
            'n_samples_processed': n_samples_processed,
            'occ_ray_encoder_state_dict': occ_ray_encoder.state_dict(),
            'occ_ray_decoder_state_dict': occ_ray_decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
        checkpoint_path,
    )

def load_checkpoint(
    checkpoint_path,
    occ_ray_encoder,
    occ_ray_decoder,
    optimizer=None,
):
    checkpoint = torch.load(checkpoint_path)
    occ_ray_encoder.load_state_dict(checkpoint['occ_ray_encoder_state_dict'])
    occ_ray_decoder.load_state_dict(checkpoint['occ_ray_decoder_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
