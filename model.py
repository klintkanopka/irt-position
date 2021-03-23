import torch
import torch.nn as nn

class IRTP(nn.Module):
    def __init__(self, n_persons, n_items, n_params):

        super().__init__()
        
        self.n_params = n_params
        self.n_persons = n_persons
        self.n_items = n_items

        # person-side parameters
        self.theta = nn.Parameter(torch.randn(self.n_persons))
        self.k = nn.Parameter(0.5 * torch.ones(self.n_persons))
        self.c = nn.Parameter(1.0 * torch.ones(self.n_persons))

        # item-side parameters
        self.beta_e = nn.Parameter(torch.zeros(self.n_items))
        self.beta_l = nn.Parameter(torch.zeros(self.n_items))
        self.alpha_e = nn.Parameter(torch.ones(self.n_items)) 
        self.alpha_l = nn.Parameter(torch.ones(self.n_items))
        
        if self.n_params < 2:
            self.alpha_e.requires_grad = False 
            self.alpha_l.requires_grad = False

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """
        Input: n x 3 tensor with person_id, item_id, item_position, and response
        Return: n x 1 tensor of probabilities of correct response
        """

        p_idx = X[:,0].long()
        i_idx = X[:,1].long()
        i_pos = X[:,2]

        pi = self.sigmoid(self.c[p_idx] * (self.k[p_idx] - i_pos))

        if self.n_params < 2:
            theta_scale = 1
        else:
            theta_scale = self.theta.std()

        p_e = self.sigmoid(self.alpha_e[i_idx] * (self.theta[p_idx]/theta_scale - (self.beta_e[i_idx] - self.beta_e.mean())))
        p_l = self.sigmoid(self.alpha_l[i_idx] * (self.theta[p_idx]/theta_scale - self.beta_l[i_idx]))

        p = pi * p_e + (1-pi) * p_l

        return p

    
