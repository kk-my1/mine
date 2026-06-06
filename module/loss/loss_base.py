import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
EPISILON=1e-10
class NCELoss(torch.nn.Module):
    def __init__(self, temperature=1,d=0.1):
        super(NCELoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=1)
        self.d = d
    def where(self, cond, x_1, x_2):
        cond = cond.type(torch.float32)
        return (cond * x_1) + ((1 - cond) * x_2)
    def forward(self, f1, f2, targets):
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)
        if(self.d > 0):
            distance_targets = abs(targets.unsqueeze(1) - targets)
            mask_distance = (distance_targets>self.d).bool()
            polarity_targets = (targets<0).int()
            mask_polarity = abs(polarity_targets.unsqueeze(1)-polarity_targets).bool()
            self_mask = (mask_distance + mask_polarity).int()
        else:
            mask = targets.unsqueeze(1) - targets
            self_mask = (torch.zeros_like(mask) != mask).int() # remove negative term
        dist = (f1.unsqueeze(1) - f2).pow(2).sum(2)
        cos = 1 - 0.5 * dist
        pred_softmax = F.softmax(cos / self.temperature,dim=1) 
        log_pos_softmax = - torch.log(pred_softmax + EPISILON) * (1 - self_mask.float())
        log_neg_softmax = - torch.log(1 - pred_softmax + EPISILON) * self_mask.float()
        log_softmax = log_pos_softmax.sum(1) / (1 - self_mask).sum(1).float() + log_neg_softmax.sum(1) / self_mask.sum(1).float()
        loss = log_softmax
        return loss.mean()

import torch
import torch.nn.functional as F




class base(torch.nn.Module):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=None, q=None):
        super(base,self).__init__()
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.iterations = iterations
        self.lam = lam
        self.q = q
    
    def compute_cost(self, z):
        z =F.normalize(z, dim=1)
        z_scores = (z @ z.t()).clamp(min=1e-7) 
        Cost = 1 - z_scores
        C = Cost.max()  # Assume C is large enough, using max value in Cost matrix
        diags = C * torch.eye(Cost.shape[0]).to(Cost.device)
        diagsoff = (1 - torch.eye(Cost.shape[0])).to(Cost.device) * Cost
        Cost = diags + diagsoff
        return Cost
    
    def gibs_kernel(self, x):
        Cost = self.compute_cost(x)
        # C_scale =(1-sim_scores)/self.epsilon
        # C= C_scale+  torch.eye(C_scale.size(0)).to(C_scale.device) * 1e5
        kernel = torch.exp(-Cost / self.epsilon)
        return kernel
    
    def forward(self,z,C=None):
        raise NotImplementedError




class gca_uot(base):
    def __init__(self, batch_size, epsilon, iterations=10, lam=0.01, q=0.6,relax_items1=1e-4, relax_items2=1e-4,r1=1.0, r2=1.0):
        super(gca_uot,self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.tau=1e5 #1e5
        self.stopThr=1e-16
        self.relax_items1=relax_items1
        self.relax_items2=relax_items2
        self.lam=lam
        self.q=q
        self.r1=r1
        self.r2=r2
    
    def forward(self,z,C=None):
        # batch_size=z.size(0)//2
        z=F.normalize(z,dim=1)
        # P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        # P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        # mask = torch.eye(P_tgt.shape[0]).to(z.device)
        # P_tgt = (P_tgt - mask)/z.size(0)
        ids = torch.div(torch.arange(z.size(0), device=z.device), 2, rounding_mode='trunc')   
        P_tgt = (ids[:, None] == ids[None, :]).float()
        P_tgt.fill_diagonal_(0)  

        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        K = self.gibs_kernel(z)
        a,b=u,v
        f1 = self.relax_items1 / (self.relax_items1 + self.epsilon)
        f2 = self.relax_items2 / (self.relax_items2+ self.epsilon)
        f = torch.zeros_like(u, requires_grad=False)
        g = torch.zeros_like(v, requires_grad=False)
        C = self.compute_cost(z) if C is None else C
        for i in range(self.iterations):
            uprev = u
            vprev = v
            f_ = torch.exp(- f / (self.epsilon + self.relax_items1))
            g_ = torch.exp(- g / (self.epsilon + self.relax_items2))
            u = ((a / (K@v + 1e-16)) ** f1) * f_
            v = ((b / (K.T@u + 1e-16)) ** f2) * g_
            if torch.any(u > self.tau) or torch.any(v > self.tau):
                f = f + self.epsilon * torch.log(torch.max(u))
                g = g + self.epsilon * torch.log(torch.max(v))
                K = torch.exp((f[:, None] + g[None, :] - C) / self.epsilon)
                v = torch.ones_like(v)
            if (torch.any(K.T@u == 0.) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                #warnings.warn('We have reached the machine precision %s' % i)
                u = uprev
                v = vprev
                break
        logu = f / self.epsilon + torch.log(u)
        logv = g / self.epsilon + torch.log(v)
        P = torch.exp(logu[:, None] + logv[None, :] - C / self.epsilon)
        # Selecting the positive pairs
        targets = torch.arange(z.size()[0])
        targets[::2] += 1  # target of 2k element is 2k+1
        targets[1::2] -= 1  # target of 2k+1 element is 2k
        kl_logits=F.cross_entropy(P.log(), targets.long().to(P.device))
        logits=self.r2*(-(P[P_tgt.bool()]/u).pow(self.q)/self.q+(self.lam*(P_tgt/u).sum(axis=1)).pow(self.q)/self.q)+self.r1*kl_logits
        return logits.mean()


class hs_rince(base):
    def  __init__(self, batch_size, epsilon, iterations=10, lam=0.01, q=0.6):
        super(hs_rince, self).__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.lam = lam
        self.q = q
    
    def forward(self,z,C=None):
        # batch_size=z.size(0)//2
        # P_tgt = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        # P_tgt = ((P_tgt.unsqueeze(0) == P_tgt.unsqueeze(1)).float()).to(z.device)
        # mask = torch.eye(P_tgt.shape[0]).to(z.device)
        # P_tgt = (P_tgt - mask)/z.size(0)
        ids = torch.arange(z.size(0), device=z.device) // 2   # [0,0,1,1,2,2,...]
        P_tgt = (ids[:, None] == ids[None, :]).float()
        P_tgt.fill_diagonal_(0)  
        u = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) /z.size(0)
        K = self.gibs_kernel(z)
        # Normalize to make them probability distributions
        u1 = u/(K @ v)
        P = (torch.diag(u1)) @ K @ (torch.diag(v))
        logits=-(P[P_tgt.bool()]/u1).pow(self.q)/self.q+(self.lam*(P_tgt/u1).sum(axis=1)).pow(self.q)/self.q
        return logits.sum()/z.size(0)


class gca_uot_labelaware(base):
    def __init__(self, batch_size, epsilon, iterations=10, lam=0.01, q=0.6,
                 relax_items1=1e-4, relax_items2=1e-4,
                 r1=1.0, r2=1.0, alpha=1.0, beta=0.2, sigma=0.5, d=None,
                 same_polarity=True):
        super().__init__(batch_size, epsilon, iterations=iterations, lam=lam, q=q)
        self.tau = 1e5
        self.stopThr = 1e-16
        self.relax_items1 = relax_items1
        self.relax_items2 = relax_items2
        self.lam = lam
        self.q = q
        self.r1 = r1
        self.r2 = r2
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.d = d
        self.same_polarity = same_polarity

    def build_label_aware_ptgt(self, labels, device):
        B = labels.size(0)
        N = 2 * B
        y = labels.float().repeat_interleave(2)
        idx = torch.arange(N, device=device)

        pair_mask = (idx[:, None] // 2 == idx[None, :] // 2).float()
        pair_mask.fill_diagonal_(0.)

        cross_view_mask = ((idx[:, None] % 2) != (idx[None, :] % 2)).float()

        dist = (y[:, None] - y[None, :]).abs()
        if self.d is not None:
            label_aff = (dist <= self.d).float()
        else:
            label_aff = torch.exp(-dist / self.sigma)

        if self.same_polarity:
            pol = ((y < 0)[:, None] == (y < 0)[None, :]).float()
            label_aff = label_aff * pol

        label_aff = label_aff * cross_view_mask
        label_aff.fill_diagonal_(0.)
        label_aff = label_aff * (1. - pair_mask)

        P_tgt = self.alpha * pair_mask + self.beta * label_aff
        P_tgt = P_tgt / (P_tgt.sum(dim=1, keepdim=True) + 1e-12)

        pair_target = pair_mask / (pair_mask.sum(dim=1, keepdim=True) + 1e-12)
        return P_tgt, pair_target

    def forward(self, z, labels, C=None):
        z = F.normalize(z, dim=1)

        u = z.new_ones((z.size(0),), requires_grad=False) / z.size(0)
        v = z.new_ones((z.size(0),), requires_grad=False) / z.size(0)
        K = self.gibs_kernel(z)
        a, b = u, v

        f1 = self.relax_items1 / (self.relax_items1 + self.epsilon)
        f2 = self.relax_items2 / (self.relax_items2 + self.epsilon)
        f = torch.zeros_like(u, requires_grad=False)
        g = torch.zeros_like(v, requires_grad=False)
        C = self.compute_cost(z) if C is None else C

        for _ in range(self.iterations):
            uprev = u
            vprev = v
            f_ = torch.exp(-f / (self.epsilon + self.relax_items1))
            g_ = torch.exp(-g / (self.epsilon + self.relax_items2))
            u = ((a / (K @ v + 1e-16)) ** f1) * f_
            v = ((b / (K.T @ u + 1e-16)) ** f2) * g_

            if torch.any(u > self.tau) or torch.any(v > self.tau):
                f = f + self.epsilon * torch.log(torch.max(u))
                g = g + self.epsilon * torch.log(torch.max(v))
                K = torch.exp((f[:, None] + g[None, :] - C) / self.epsilon)
                v = torch.ones_like(v)

            if (torch.any(K.T @ u == 0.) or torch.any(torch.isnan(u)) or torch.any(torch.isnan(v))
                    or torch.any(torch.isinf(u)) or torch.any(torch.isinf(v))):
                u = uprev
                v = vprev
                break

        logu = f / self.epsilon + torch.log(u + 1e-12)
        logv = g / self.epsilon + torch.log(v + 1e-12)
        P = torch.exp(logu[:, None] + logv[None, :] - C / self.epsilon)

        P_row = P / (P.sum(dim=1, keepdim=True) + 1e-12)
        P_tgt, pair_target = self.build_label_aware_ptgt(labels, z.device)

        loss_pair = -(pair_target * torch.log(P_row + 1e-12)).sum(dim=1).mean()
        loss_soft = F.kl_div((P_row + 1e-12).log(), P_tgt, reduction='batchmean')

        loss = self.r1 * loss_pair + self.r2 * loss_soft
        return loss