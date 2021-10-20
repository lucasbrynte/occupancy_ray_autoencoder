from collections import OrderedDict
import torch

def serialize_checkpoint_metadata(d):
    return '__'.join([key + '_' + val for key, val in d.items()])

def deserialize_checkpoint_metadata(s):
    def split_keys_from_vals(keyval):
        key, val = keyval.split('_')
        return key, val
    return OrderedDict(map(split_keys_from_vals, s.split('__')))

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
