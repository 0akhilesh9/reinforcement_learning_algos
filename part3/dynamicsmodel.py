import torch
from torch import nn
import torch.nn.functional as F

import vae
import statetransition

input_dim = 64
output_dim = input_dim
z_dim = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mse = nn.MSELoss()
l1loss = torch.nn.SmoothL1Loss()

# VAE loss function
def vae_loss_function(recon_x, x, mu, logvar, batch_size):
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduce=False)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size * (output_dim * output_dim)

    return (BCE + KLD).mean([1,2,3])

# Reward network loss function
def reward_loss_function(actual_reward, predicted_reward, actual_done, predicted_done):
    x = actual_reward.reshape(-1)
    y = predicted_reward*(1-predicted_done).reshape(-1)
    l1 = mse(x,y)
    l2=mse(actual_done, predicted_done)
    return l1+l2

# State transition network loss function
def state_transition_loss(state_trans_inp, enc_next_img, dec_state_trans_inp, next_img):
    l1 = F.binary_cross_entropy_with_logits(state_trans_inp, enc_next_img, reduce=False).mean([1,2,3])
    l2 = F.binary_cross_entropy_with_logits(dec_state_trans_inp, next_img, reduce=False).mean([1,2,3])
    return torch.abs(l1 + l2)

# Class for Dyna-Q
class DynaQ(nn.Module):
    # Init method
    def __init__(self, input_dim, n_actions):
        super().__init__()
        n_action_dims = 1
        self.vae = vae.VAE(input_dim, z_dim).to(device)
        self.state_trans = statetransition.stateTransModel(n_action_dims)
        self.reward_conv = statetransition.rewardConv()

    # Forward method
    def forward(self, state_img, action, train_flag=False):
        # Get outputs from VAE
        enc_img, dec_img, mu, var = self.vae(state_img)
        # Get the next state from state-transition network
        state_trans_out = self.state_trans(enc_img, action.float())
        # Decode next state
        next_pred_img = self.vae.decoder(state_trans_out)
        # Get reward/done probabilities
        reward_prob, done_prob = self.reward_conv(state_trans_out)
        # Mean reward is 30 (obtained from breakout-ram reference page)
        reward_val = reward_prob * 30
        a = torch.ones(done_prob.shape).to(device)
        b = torch.zeros(done_prob.shape).to(device)
        done_val = torch.where(done_prob>0.5,a,b)

        if train_flag:
            return state_trans_out, next_pred_img, dec_img, reward_val, done_val, mu, var
        return next_pred_img, reward_val, done_val