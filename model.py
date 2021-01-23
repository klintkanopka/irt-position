import torch
import torch.nn as nn

class IRTP(nn.Module):
    def __init__(self, n_persons, n_items, n_params):

        super().__init__()
        
        # person-side parameters
        self.theta = nn.Parameter(torch.zeros(n_persons))
        self.k = nn.Parameter(20 * torch.ones(n_persons))
        self.c = nn.Parameter(0.3 * torch.ones(n_persons))

        # item-side parameters
        self.beta_e = nn.Parameter(torch.zeros(n_items))
        self.beta_l = nn.Parameter(torch.zeros(n_items))
        self.alpha_e = nn.Parameter(torch.ones(n_items)) 
        self.alpha_l = nn.Parameter(torch.ones(n_items))
        
        if (n_params < 2):
            self.alpha_e.requires_grad = False 
            self.alpha_l.requires_grad = False

        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        """
        Input: n x 3 tensor with person_id, item_id, item_position, and response
        Return: n x 1 tensor of probabilities of correct response
        """

        p_idx = X[:,0]
        i_idx = X[:,1]
        i_pos = X[:,2]

        pi = self.sigmoid(self.c[p_idx] * (self.k[p_idx] - i_pos))

        p_e = self.sigmoid(self.alpha_e[i_idx] * (self.theta[p_idx] - (self.beta_e[i_idx] - self.beta_e.mean())))
        p_l = self.sigmoid(self.alpha_l[i_idx] * (self.theta[p_idx] - self.beta_l[i_idx]))

        p = pi * p_e + (1-pi) * p_l

        return p

    
