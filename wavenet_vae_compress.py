import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
import sys
sys.path.append('.')
sys.path.append('/home/robert/git/pytorch-wavenet')
import os
import numpy as np
import math
import util
import rans
from torch_vae import tvae_utils
import time
import json
from dataset import AudioDataset
from vae import VAE, OneHotConvolutionalEncoder, BetterWaveNetDecoder
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from scipy.special import softmax
from wavenet_utils import one_hot

def parameter_count(module):
    par = list(module.parameters())
    s = sum([np.prod(list(d.size())) for d in par])
    return s

np.seterr(over='raise')
prfx = "/home/robert/git/pytorch-wavenet/"
rng = np.random.RandomState(0)
prior_precision = 8
obs_precision = 14
compress_lengths = []
snapshot_name = "51"

with open(f"{prfx}/configs/{snapshot_name}.json") as f:
    data = f.read()
config = json.loads(data)
encoder_wavenet_args = config["encoder_wavenet_args"]
decoder_wavenet_args = config["decoder_wavenet_args"]
train_args = config["train_args"]
batch_size = train_args["batch_size"]
snapshot_path = f"snapshots/{snapshot_name}"
snapshot_interval = train_args["snapshot_interval"]
ar_factor = train_args.get("ar_factor", None)
hard_gumbel_softmax = train_args.get("hard_gumbel_softmax", False)
gumbel_softmax_temperature = train_args["gumbel_softmax_temperature"]
data_length = config.get("data_length", None)
model_type = config.get("type", None)

model = VAE(
    OneHotConvolutionalEncoder(encoder_wavenet_args),
    BetterWaveNetDecoder(decoder_wavenet_args),
    hard_gumbel_softmax=hard_gumbel_softmax,
) 
checkpoint_dict = torch.load(f"{prfx}/snapshots/{snapshot_name}/{snapshot_name}_120000", map_location='cpu')
model = nn.DataParallel(model)
model.load_state_dict(checkpoint_dict['model'])
model = model.module
model = model.to('cpu')
model.eval()

if data_length is None:
    receptive_field_closest_power_of_3 = int(3**math.ceil(math.log(model.decoder.wavenet.receptive_field, 3)))
    data_length = receptive_field_closest_power_of_3 * 3
dataset = AudioDataset(prfx+"wav_audio/small_wav_tensor.pt", data_length)        
print('the dataset has ' + str(len(dataset)) + ' items')
print(f'each item has length {dataset.len_sample}')
print(f'The WaveNetVAE has {parameter_count(model.encoder) + parameter_count(model.decoder)} parameters')
print(f'The encoder has {parameter_count(model.encoder)} parameters')
print(f'The decoder has {parameter_count(model.decoder)} parameters')

gumbel_softmax_temperature = 1.
x = dataset[50].long().unsqueeze(0)
# print("x.size(-1)", x.size(-1))
INPUT_LENGTH = 3**6
# INPUT_LENGTH = x.size(-1)
x = x[:, 0:INPUT_LENGTH]
# print("x", x)
# Adapt as appropriate for the model
if model_type == "betterwavenet":
    latent_shape = (x.size(0), x.size(1)//(3**4))
else:
    raise Exception(f"no latent shape specified for model type {model_type}")
# print("latent_shape", latent_shape)

p_x, q_z = model(x, gumbel_softmax_temperature)
d = model.loss(p_x, x, q_z, 0., ar_factor)
loss = d['loss']
cross_entropy = d['cross_entropy']
kl_divergence = d['kl_divergence']
posterior_entropy_penalty = d['posterior_entropy_penalty']
bits_per_dimension = ((kl_divergence+cross_entropy)/math.log(2)) / x.size(-1)
print("ELBO in bits per dimension:", bits_per_dimension)

def cvae_append(latent_shape, rec_net, prior_prec=8,
               latent_prec=14):
    """
    VAE with discrete latents, parameterized by categorical dist and
    with uniform prior
    """
    def post_pop(data):
        # print("post_pop data.shape", data.shape)
        torch.manual_seed(0)
        posterior_probs = rec_net(data.unsqueeze(0))
        # print("post_pop posterior_probs.shape", posterior_probs.shape)
        return util.categoricals_pop(posterior_probs, latent_prec)
    def lik_append(latents):
        '''
        returns: function(state, data) that returns data appended to state
        '''
        def lik_append_(state, data):
            '''
            data: (T,) tensor
            '''
            # print("lik_append_ data.shape", data.shape,)
            print("lik_append_ latents", latents, latents.shape)
            one_hot_x = one_hot(data.unsqueeze(0), model.decoder.wavenet.out_channels)
            one_hot_c = one_hot(torch.tensor(latents).unsqueeze(0), model.decoder.wavenet.cin_channels)
            torch.manual_seed(0)
            logits_list = model.decoder.wavenet.incremental_forward(all_x=one_hot_x, c=one_hot_c)
            print("logits_list[0].shape", logits_list[0].shape)
            probs_list = [F.softmax(outputs[-1], dim=-1).numpy() for outputs in logits_list]
            probs = np.stack(probs_list)
            print("probs.shape", probs.shape)
            # probs = F.softmax(logits, dim=1).squeeze().numpy().transpose()
            torch.save(probs, "lik_append_probs_2.pt")
            state = util.categoricals_append(probs, obs_precision)(state, data.numpy())
            return state
        return lik_append_
    prior_append = util.uniforms_append(prior_prec)
    return util.bb_ans_append(post_pop, lik_append, prior_append)

def cvae_pop(
        latent_shape, rec_net, prior_prec=8, latent_prec=14):
    """
    VAE with discrete latents, parameterized by categorical dist and
    with uniform prior
    """
    prior_pop = util.uniforms_pop(prior_prec, np.prod(latent_shape))
    def lik_pop(latents):
        '''
        returns: function(state) that returns (state, data)
        '''
        def lik_pop_(state):
            # print("type(state)", type(state))
            print("lik_pop_ latents", latents, latents.shape)
            one_hot_c = one_hot(torch.tensor(latents).unsqueeze(0), model.decoder.wavenet.cin_channels)
            torch.manual_seed(0)
            state, data, all_probs = model.decoder.wavenet.incremental_forward_recover(state, length=INPUT_LENGTH, precision=obs_precision, categoricals_pop=util.categoricals_pop, c=one_hot_c)
            return state, data
        return lik_pop_
    def post_append(data):
        # print("post_append data", data)
        # print("data.dtype", data.dtype)
        torch.manual_seed(0)
        posterior_probs = rec_net(np.expand_dims(data, 0))
        # print("post_append posterior_probs", posterior_probs)
        return util.categoricals_append(posterior_probs, latent_prec)
    return util.bb_ans_pop(prior_pop, lik_pop, post_append)

def wavenet_logits_to_probs(logits_net):
    def probs_net(data):
        logits = logits_net(data).squeeze(0).transpose(1,0)
        return softmax(logits, axis=1)
    return probs_net

rec_net = wavenet_logits_to_probs(tvae_utils.torch_fun_to_numpy_fun(model.encode))
vae_append = cvae_append(latent_shape, rec_net, prior_precision)
vae_pop = cvae_pop(latent_shape, rec_net, prior_precision)

# randomly generate some 'other' bits
other_bits = rng.randint(1 << 32, size=latent_shape[-1]*2, dtype=np.uint32)
state = rans.x_init
state = util.uniforms_append(32)(state, other_bits)
# ---------------------------- ENCODE ------------------------------------
x = x.squeeze()
# print("x:", x, x.size())
state = vae_append(state, x)
compressed_message = rans.flatten(state)
print("Used " + str(32 * (len(compressed_message) - len(other_bits))) +
      " bits.")
message_length = x.size(-1)
compressed_message_length = (len(compressed_message) - len(other_bits))*4
compression_ratio = message_length / compressed_message_length
compressed_bits_per_dimension = 8 / compression_ratio
print("compression ratio:", compression_ratio)
print("compression bits per dimension:", compressed_bits_per_dimension)
# ---------------------------- DECODE ------------------------------------
state = rans.unflatten(compressed_message)
state, x_ = vae_pop(state)
# assert all(x == x_)
print("First entries in x: ", x[:10])
print("First entries in x_: ", x_[:10])

#  recover the other bits from q(y|x_0)
state, recovered_bits = util.uniforms_pop(32, other_bits.shape[0])(state)
assert all(other_bits == recovered_bits)
assert state == rans.x_init