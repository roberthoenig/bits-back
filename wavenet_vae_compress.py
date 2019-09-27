import matplotlib.pyplot as plt

import sys
sys.path.append('.')
sys.path.append('/home/robert/git/pytorch-wavenet')
import os
import torch
import numpy as np
import util
import rans
from torch_vae.tvae_beta_binomial import BetaBinomialVAE
from torch_vae import tvae_utils
from torchvision import datasets, transforms
import time
import json

from audio_data import AudioDataset
from vae import VAE, WaveNetEncoder, WaveNetDecoder

import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from scipy.special import softmax

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

pytorch_wavenet_path = "/home/robert/git/pytorch-wavenet/"

rng = np.random.RandomState(0)
np.seterr(over='raise')

prior_precision = 8
obs_precision = 14
q_precision = 14

compress_lengths = []

latent_shape = (1, 508)

config_file = pytorch_wavenet_path+"medium_lr3.json"
checkpoint_file = pytorch_wavenet_path+"snapshots/slow_anneal_temp_linear_classic_music_medium_lr3_500000"

with open(config_file) as f:
    data = f.read()
config = json.loads(data)
encoder_wavenet_args = config["encoder_wavenet_args"]
decoder_wavenet_args = config["decoder_wavenet_args"]
model = VAE(
    WaveNetEncoder(encoder_wavenet_args),
    WaveNetDecoder(decoder_wavenet_args),
    hard_gumbel_softmax=True
)
checkpoint_dict = torch.load(checkpoint_file, map_location='cpu')
model.load_state_dict(checkpoint_dict['model'])
model.eval();

dataset = AudioDataset(pytorch_wavenet_path+"wav_audio/small_wav_tensor.pt", model.encoder.wavenet.receptive_field*4)        
print('the dataset has ' + str(len(dataset)) + ' items')
print(f'each item has length {dataset.len_sample}')
print(f'The WaveNetVAE has {model.encoder.wavenet.parameter_count() + model.decoder.wavenet.parameter_count()} parameters')

gumbel_softmax_temperature = 1.# config['train_args']['gumbel_softmax_temperature']
x = dataset[2000].long().unsqueeze(0)

model.is_input_one_hot = True
p_x, q_z = model(x, .001)

loss = model.loss(p_x, x, q_z)
loss

```python
def bb_ans_append(post_pop, lik_append, prior_append):
    def append(state, data):  # vae_append
        state, latent = post_pop(data)(state)
        state = lik_append(latent)(state, data)
        state = prior_append(state, latent)
        return state
    return append   # becomes vae_append

def bb_ans_pop(prior_pop, lik_pop, post_append):
    def pop(state):
        state, latent = prior_pop(state)
        state, data = lik_pop(latent)(state)
        state = post_append(data)(state, latent)
        return state, data
    return pop
```

def cvae_append(latent_shape, gen_net, rec_net, obs_append, prior_prec=8,
               latent_prec=14):
    """
    VAE with discrete latents, parameterized by categorical dist and
    with uniform prior
    """
    def post_pop(data):
        posterior_probs = rec_net(data.unsqueeze(0))
        print("posterior_probs.shape", posterior_probs.shape)
        return util.categoricals_pop(posterior_probs, latent_prec)

    def lik_append(latents):
        print("lik_append latents.shape", latents.shape)
        print("lik_append latents", latents)
        obs_params = gen_net(np.reshape(latents, latent_shape))
        print("obs_params.shape", obs_params.shape)
        return obs_append(obs_params)

    prior_append = util.uniforms_append(prior_prec)
    return util.bb_ans_append(post_pop, lik_append, prior_append)

def cvae_pop(
        latent_shape, gen_net, rec_net, obs_pop, prior_prec=8, latent_prec=12):
    """
    VAE with discrete latents, parameterized by categorical dist and
    with uniform prior
    """
    prior_pop = util.uniforms_pop(prior_prec, np.prod(latent_shape))

    def lik_pop(latents):
        print("lik_pop latents.shape", latents.shape)
        print("lik_pop latents", latents)
        obs_params = gen_net(np.reshape(latents, latent_shape))
        print("obs_params.shape", obs_params.shape)
        return obs_pop(obs_params)

    def post_append(data):
        print("post_append data.shape", data.shape)
        posterior_probs = rec_net(np.expand_dims(data, 0))
        print("posterior_probs.shape", posterior_probs.shape)
        return util.categoricals_append(posterior_probs, latent_prec)

    return util.bb_ans_pop(prior_pop, lik_pop, post_append)

model.decoder.is_input_one_hot = False

def wavenet_logits_to_probs(logits_net):
    def probs_net(data):
        logits = logits_net(data).squeeze(0).transpose(1,0)
        return softmax(logits, axis=1)
    return probs_net

rec_net = wavenet_logits_to_probs(tvae_utils.torch_fun_to_numpy_fun(model.encode))
gen_net = wavenet_logits_to_probs(tvae_utils.torch_fun_to_numpy_fun(model.decode))

# obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
# obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)

def get_obs_append(precision):
    def obs_append(obs_params):
        return util.categoricals_append(obs_params, precision)
    return obs_append
def get_obs_pop(precision):
    def obs_pop(obs_params):
        return util.categoricals_pop(obs_params, precision)
    return obs_pop

obs_append = get_obs_append(obs_precision)
obs_pop = get_obs_pop(obs_precision)

vae_append = cvae_append(latent_shape, gen_net, rec_net, obs_append,
                             prior_precision, q_precision)
vae_pop = cvae_pop(latent_shape, gen_net, rec_net, obs_pop,
                       prior_precision, q_precision)

# randomly generate some 'other' bits
other_bits = rng.randint(1 << 32, size=508*10, dtype=np.uint32)
state = rans.x_init
state = util.uniforms_append(32)(state, other_bits)

other_bits.shape, other_bits.max(), other_bits.min(), other_bits.dtype

# ---------------------------- ENCODE ------------------------------------
x = x.squeeze()
state = vae_append(state, x)
compressed_message = rans.flatten(state)

print("Used " + str(32 * (len(compressed_message) - len(other_bits))) +
      " bits.")

# ---------------------------- DECODE ------------------------------------
state = rans.unflatten(compressed_message)

state, x_ = vae_pop(state)
# assert all(x == x_)
x[0:10], x_[0:10]

#  recover the other bits from q(y|x_0)
state, recovered_bits = util.uniforms_pop(32, other_bits.shape[0])(state)
assert all(other_bits == recovered_bits)
assert state == rans.x_init