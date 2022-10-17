# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
## IRM -MMD only chay 3 lan do loi, bjo chay lai het xem


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import math
from networks import Decoder
import copy
import numpy as np
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

import networks
from misc import random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts



ALGORITHMS = [
    'ERM',
    #'Fish',
    #'IRM',
    #'GroupDRO',
    #'Mixup',
    #'MLDG',
    #'CORAL',
    #'MMD',
    #'DANN',
    #'CDANN',
    #'MTL',
    #'SagNet',
    #'ARM',
    #'VREx',
    #'RSC',
    #'SD',
    #'ANDMask',
    #'SANDMask',
    #'IGA',
    'SelfReg',
    #"Fishr",
    #'TRM',
    #'IB_ERM',
    #'IB_IRM',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

#### original/traditional ERM
class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)
        
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}
    

    def predict(self, x):
        return self.network(x)

#### Original IRM
class IRM(ERM):
    """Invariant Risk Minimization"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def _irm_penalty(logits, y):
        device = "cuda" if logits[0][0].is_cuda else "cpu"
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        penalty_weight = (self.hparams['irm_lambda'] if self.update_count
                          >= self.hparams['irm_penalty_anneal_iters'] else
                          1.0)
        nll = 0.
        penalty = 0.

        all_x = torch.cat([x for x,y in minibatches])
        all_logits = self.network(all_x)
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += F.cross_entropy(logits, y)
            penalty += self._irm_penalty(logits, y)
        nll /= len(minibatches)
        penalty /= len(minibatches)
        loss = nll + (penalty_weight * penalty)

        if self.update_count == self.hparams['irm_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 'nll': nll.item(),
            'penalty': penalty.item()}

## Original MMD and CORAL
class AbstractMMD(ERM):
    """
    Perform ERM while matching the pair-wise domain feature distributions
    using MMD (abstract class)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, gaussian):
        super(AbstractMMD, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        if gaussian:
            self.kernel_type = "gaussian"
        else:
            self.kernel_type = "mean_cov"

    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                            1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y):
        if self.kernel_type == "gaussian":
            Kxx = self.gaussian_kernel(x, x).mean()
            Kyy = self.gaussian_kernel(y, y).mean()
            Kxy = self.gaussian_kernel(x, y).mean()
            return Kxx + Kyy - 2 * Kxy
        else:
            mean_x = x.mean(0, keepdim=True)
            mean_y = y.mean(0, keepdim=True)
            cent_x = x - mean_x
            cent_y = y - mean_y
            cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
            cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

            mean_diff = (mean_x - mean_y).pow(2).mean()
            cova_diff = (cova_x - cova_y).pow(2).mean()

            return mean_diff + cova_diff

    def update(self, minibatches, unlabeled=None):
        objective = 0
        penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]

        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i])
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)

        self.optimizer.zero_grad()
        (objective + (self.hparams['mmd_gamma']*penalty)).backward()
        self.optimizer.step()

        if torch.is_tensor(penalty):
            penalty = penalty.item()

        return {'loss': objective.item(), 'penalty': penalty}


class MMD(AbstractMMD):
    """
    MMD using Gaussian kernel
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(MMD, self).__init__(input_shape, num_classes,
                                          num_domains, hparams, gaussian=True)


class CORAL(AbstractMMD):
    """
    MMD using mean and covariance difference
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(CORAL, self).__init__(input_shape, num_classes,
                                         num_domains, hparams, gaussian=False)





#### 1. MMD + reconstruction loss             
class IB_IRM(ERM):
    """Information Bottleneck based IRM on feature with conditionning"""

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.register_buffer('update_count', torch.tensor([0]))

    @staticmethod
    def my_cdist(self, x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)

    def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
                                          1000]):
        D = self.my_cdist(x, y)
        K = torch.zeros_like(D)

        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))

        return K

    def mmd(self, x, y): ### using Gaussian kernel here - fix for MMD 
        Kxx = self.gaussian_kernel(x, x).mean()
        Kyy = self.gaussian_kernel(y, y).mean()
        Kxy = self.gaussian_kernel(x, y).mean()
        return Kxx + Kyy - 2 * Kxy

    def update(self, minibatches, unlabeled=None):
        device = "cuda" if minibatches[0][0].is_cuda else "cpu"
        ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
                          >= self.hparams['ib_penalty_anneal_iters'] else
                          0.0)
        mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
                  >= self.hparams['mmd_penalty_anneal_iters'] else 0.0)
                  
        objective = 0
        penalty = 0
        ib_penalty = 0
        nmb = len(minibatches)

        features = [self.featurizer(xi) for xi, _ in minibatches]
        classifs = [self.classifier(fi) for fi in features]
        targets = [yi for _, yi in minibatches]
        all_x = torch.cat([x for x,y in minibatches])
        all_features = self.featurizer(all_x)
        
        ###### MMD loss
        for i in range(nmb):
            objective += F.cross_entropy(classifs[i], targets[i]) ### emperical risk
            for j in range(i + 1, nmb):
                penalty += self.mmd(features[i], features[j])

        objective /= nmb
        if nmb > 1:
            penalty /= (nmb * (nmb - 1) / 2)*10
        loss = objective + penalty * mmd_penalty_weight  ### first + second term
        
        ##### Reconstruction loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        d_model = Decoder().to(device)
        latent_code = all_features ### same variable 
        batch_data =  all_x[:, :, :-1, :-1] ### same variabel but since 27x27 so just delete the last dimension
        recon_loss = nn.MSELoss().to(device)
        recon_image = d_model(latent_code)
        recon_image = recon_image[:, :-1, :, :] ### co on k khi ma minh bo 1 channel di?
        loss_recon = recon_loss(batch_data,recon_image)
        
        ## The final loss sumup of all 3 loss terms
        loss += ib_penalty * ib_penalty_weight  ## loss + third term

        if self.update_count == self.hparams['ib_penalty_anneal_iters'] or self.update_count == self.hparams['mmd_penalty_anneal_iters']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        return {'loss': loss.item(), 
                'IB_penalty': ib_penalty.item()}


# ### 2. IRM + reconstruction loss
# class IB_IRM(IRM):
#     """Information Bottleneck based IRM on feature with conditionning"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         self.optimizer = torch.optim.Adam(
#             list(self.featurizer.parameters()) + list(self.classifier.parameters()),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )
#         self.register_buffer('update_count', torch.tensor([0]))

#     @staticmethod
#     def _irm_penalty(logits, y):
#         device = "cuda" if logits[0][0].is_cuda else "cpu"
#         scale = torch.tensor(1.).to(device).requires_grad_()
#         loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
#         loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
#         grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
#         grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
#         result = torch.sum(grad_1 * grad_2)
#         return result

#     #### thuan added    
#     def my_cdist(self, x1, x2):
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#         res = torch.addmm(x2_norm.transpose(-2, -1),
#                           x1,
#                           x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
#         return res.clamp_min_(1e-30)

#     def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
#                                             1000]):
#         D = self.my_cdist(x, y)
#         K = torch.zeros_like(D)

#         for g in gamma:
#             K.add_(torch.exp(D.mul(-g)))

#         return K

#     def mmd(self, x, y): ### using Gaussian kernel here - fix for MMD 
#         Kxx = self.gaussian_kernel(x, x).mean()
#         Kyy = self.gaussian_kernel(y, y).mean()
#         Kxy = self.gaussian_kernel(x, y).mean()
#         return Kxx + Kyy - 2 * Kxy

#     def update(self, minibatches, unlabeled=None):
#         device = "cuda" if minibatches[0][0].is_cuda else "cpu"
#         irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                           >= self.hparams['irm_penalty_anneal_iters'] else
#                           1.0)
#         ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                           >= self.hparams['ib_penalty_anneal_iters'] else
#                           0.0)
#         mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
#                           >= self.hparams['mmd_penalty_anneal_iters'] else
#                           1.0)
#         recon_weight       = (self.hparams['recon_lambda'] if self.update_count
#                           >= self.hparams['recon_penalty_anneal_iters'] else
#                           1.0)

#         nll = 0.
#         irm_penalty = 0.
#         ib_penalty = 0.
#         mmd_penalty = 0.

#         all_x = torch.cat([x for x, _ in minibatches])
#         all_features = self.featurizer(all_x)
#         ###Thuan added - normalize the features
#         all_features = F.normalize(all_features, dim=-1)
#         all_logits = self.classifier(all_features)
#         all_logits_idx = 0
#         for i, (x, y) in enumerate(minibatches):
#             features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]# features is all_features which is inter_logits
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             irm_penalty += self._irm_penalty(logits, y)
#             #### THUAN add
#             class_loss = F.cross_entropy(logits,y).div(math.log(2)) ##H(Y|Z)
#             ib_penalty += features.var(dim=0).mean() ##H(Z)
#             ib_penalty += class_loss 

#         mmd_penalty = 0
#         nmb = len(minibatches)

#         features = [self.featurizer(xi) for xi, _ in minibatches]
       
#         for i in range(nmb):
#             for j in range(i + 1, nmb):
#                 mmd_penalty += self.mmd(features[i], features[j])

#         if nmb > 1:
#             mmd_penalty /= (nmb * (nmb - 1) / 2)*0.01
      
#         ##### Reconstruction loss
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         d_model = Decoder().to(device)
#         latent_code = all_features ### same variable 
#         batch_data =  all_x[:, :, :-1, :-1] ### same variabel but since 27x27 so just delete the last dimension
#         recon_loss = nn.MSELoss().to(device)
#         loss_recon = recon_loss(batch_data,d_model(latent_code))

        
#         ## final loss
#         nll /= len(minibatches) ### risk
#         irm_penalty /= len(minibatches) ### L_IRM
#         ib_penalty /= len(minibatches)
        
#         # Compile loss k muon random nen chon luon o day
        
#         irm_penalty_weight= 0 ### Lan sau: de irm_penalty_weight = 0 nhe!!!!!
#         ib_penalty_weight = 0 ### cai nay cho entropy nhe, de 0.
#         mmd_penalty_weight = 0
#         loss = nll ### emperical risk
#         loss += irm_penalty_weight * irm_penalty ### chi con IRM = ERM (nll) + IRM penalty
#         loss += ib_penalty_weight * ib_penalty*0  
#         loss += mmd_penalty * mmd_penalty_weight*0 
#         loss += loss_recon* recon_weight ### tuy chon recon_weight-> IRM +recont

#         if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
#             # Reset Adam, because it doesn't like the sharp jump in gradient
#             # magnitudes that happens at this step.
#             self.optimizer = torch.optim.Adam(
#                 list(self.featurizer.parameters()) + list(self.classifier.parameters()),
#                 lr=self.hparams["lr"],
#                 weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.update_count += 1
#         return {'loss': loss.item(), 
#                 'nll': nll.item(),
#                 'IRM_penalty': irm_penalty.item(), 
#                 'IB_penalty': ib_penalty.item()}


# ### 3. CORAL + reconstruction loss
# ### Coral and MMD are pretty similar, we just change a bit at MMD loss to achieve Coral loss
# class IB_IRM(IRM):
#     """Information Bottleneck based IRM on feature with conditionning"""

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(IB_IRM, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         self.optimizer = torch.optim.Adam(
#             list(self.featurizer.parameters()) + list(self.classifier.parameters()),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )
#         self.register_buffer('update_count', torch.tensor([0]))

#     @staticmethod
#     def _irm_penalty(logits, y):
#         device = "cuda" if logits[0][0].is_cuda else "cpu"
#         scale = torch.tensor(1.).to(device).requires_grad_()
#         loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
#         loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
#         grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
#         grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
#         result = torch.sum(grad_1 * grad_2)
#         return result

#     #### thuan added    
#     def my_cdist(self, x1, x2):
#         x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
#         x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
#         res = torch.addmm(x2_norm.transpose(-2, -1),
#                           x1,
#                           x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
#         return res.clamp_min_(1e-30)

#     def gaussian_kernel(self, x, y, gamma=[0.001, 0.01, 0.1, 1, 10, 100,
#                                             1000]):
#         D = self.my_cdist(x, y)
#         K = torch.zeros_like(D)

#         for g in gamma:
#             K.add_(torch.exp(D.mul(-g)))

#         return K

#     def mmd(self, x, y): ### this for CORAL - fix for CORAL
#         mean_x = x.mean(0, keepdim=True)
#         mean_y = y.mean(0, keepdim=True)
#         cent_x = x - mean_x
#         cent_y = y - mean_y
#         cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
#         cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

#         mean_diff = (mean_x - mean_y).pow(2).mean()
#         cova_diff = (cova_x - cova_y).pow(2).mean()

#         return mean_diff + cova_diff  

#     def update(self, minibatches, unlabeled=None):
#         device = "cuda" if minibatches[0][0].is_cuda else "cpu"
#         irm_penalty_weight = (self.hparams['irm_lambda'] if self.update_count
#                           >= self.hparams['irm_penalty_anneal_iters'] else
#                           1.0)
#         ib_penalty_weight = (self.hparams['ib_lambda'] if self.update_count
#                           >= self.hparams['ib_penalty_anneal_iters'] else
#                           0.0)
#         mmd_penalty_weight = (self.hparams['mmd_lambda'] if self.update_count
#                           >= self.hparams['mmd_penalty_anneal_iters'] else
#                           1.0)
#         recon_weight       = (self.hparams['recon_lambda'] if self.update_count
#                           >= self.hparams['recon_penalty_anneal_iters'] else
#                           1.0)

#         nll = 0.
#         irm_penalty = 0.
#         ib_penalty = 0.
#         mmd_penalty = 0.

#         all_x = torch.cat([x for x, _ in minibatches])
#         all_features = self.featurizer(all_x)
#         ###Thuan added - normalize the features
#         all_features = F.normalize(all_features, dim=-1)
#         all_logits = self.classifier(all_features)
#         all_logits_idx = 0
#         for i, (x, y) in enumerate(minibatches):
#             features = all_features[all_logits_idx:all_logits_idx + x.shape[0]]# features is all_features which is inter_logits
#             logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
#             all_logits_idx += x.shape[0]
#             nll += F.cross_entropy(logits, y)
#             irm_penalty += self._irm_penalty(logits, y)
#             #### THUAN add
#             class_loss = F.cross_entropy(logits,y).div(math.log(2)) ##H(Y|Z)
#             ib_penalty += features.var(dim=0).mean() ##H(Z)
#             ib_penalty += class_loss 

#         mmd_penalty = 0
#         nmb = len(minibatches)

#         features = [self.featurizer(xi) for xi, _ in minibatches]
       
#         for i in range(nmb):
#             for j in range(i + 1, nmb):
#                 mmd_penalty += self.mmd(features[i], features[j])

#         if nmb > 1:
#             mmd_penalty /= (nmb * (nmb - 1) / 2)*0.01
      
#         ##### Reconstruction loss
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         d_model = Decoder().to(device)
#         latent_code = all_features ### same variable 
#         batch_data =  all_x[:, :, :-1, :-1] ### same variabel but since 27x27 so just delete the last dimension
#         recon_loss = nn.MSELoss().to(device)
#         loss_recon = recon_loss(batch_data,d_model(latent_code))

        
#         ## final loss
#         nll /= len(minibatches) ### risk
#         irm_penalty /= len(minibatches) ### L_IRM
#         ib_penalty /= len(minibatches)

          ###Note: We utilized the code from other papers and set irm_penalty_weight= 0 and ib_penalty_weight = 0, so the loss has only mmd_penalty * mmd_penalty_weight and loss_recon* recon_weight 
#         irm_penalty_weight= 0 
#         ib_penalty_weight = 0 
#         loss = nll ### emperical risk
#         loss += irm_penalty_weight * irm_penalty *0  
#         loss += ib_penalty_weight * ib_penalty*0  
#         loss += mmd_penalty * mmd_penalty_weight ### mmd now is coral
#         loss += loss_recon* recon_weight 

#         if self.update_count == self.hparams['irm_penalty_anneal_iters'] or self.update_count == self.hparams['ib_penalty_anneal_iters']:
#             # Reset Adam, because it doesn't like the sharp jump in gradient
#             # magnitudes that happens at this step.
#             self.optimizer = torch.optim.Adam(
#                 list(self.featurizer.parameters()) + list(self.classifier.parameters()),
#                 lr=self.hparams["lr"],
#                 weight_decay=self.hparams['weight_decay'])

#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         self.update_count += 1
#         return {'loss': loss.item(), 
#                 'nll': nll.item(),
#                 'IRM_penalty': irm_penalty.item(), 
#                 'IB_penalty': ib_penalty.item()}

# #### 02 versions of ERM + reconstruction loss, we prefer version 4-b!
# #### 4-a - ERM + reconstruction loss !
# class ERM(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(ERM, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         self.featurizer = networks.Featurizer(input_shape, self.hparams)
#         self.classifier = networks.Classifier(
#             self.featurizer.n_outputs,
#             num_classes,
#             self.hparams['nonlinear_classifier'])

#         self.network = nn.Sequential(self.featurizer, self.classifier)
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )

#     def update(self, minibatches, unlabeled=None):
#         all_x = torch.cat([x for x,y in minibatches])
#         all_y = torch.cat([y for x,y in minibatches])
#         all_features = self.featurizer(all_x)
#         loss = F.cross_entropy(self.predict(all_x), all_y)
        
#         ##### Reconstruction loss
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         d_model = Decoder().to(device)
#         latent_code = all_features ### same variable 
#         batch_data =  all_x[:, :, :-1, :-1] ### same variabel but since 27x27 so just delete the last dimension
#         recon_loss = nn.MSELoss().to(device)
#         loss_recon = recon_loss(batch_data,d_model(latent_code))
        
#         loss += loss_recon *0.5  ### hand tunining - using 1 first -> acc 51.9, 10-> acc =51.4, 0.2 -> acc = 51.5, 50-> acc =51.1, 0.5->71.1? run again!
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return {'loss': loss.item()}
    
#     def predict(self, x):
#         return self.network(x)

# #### 4-b ERM + reconstruction  loss
# class ERM(Algorithm):
#     """
#     Empirical Risk Minimization (ERM)
#     """

#     def __init__(self, input_shape, num_classes, num_domains, hparams):
#         super(ERM, self).__init__(input_shape, num_classes, num_domains,
#                                   hparams)
#         self.featurizer = networks.Featurizer(input_shape, self.hparams)
#         self.classifier = networks.Classifier(
#             self.featurizer.n_outputs,
#             num_classes,
#             self.hparams['nonlinear_classifier'])

#         self.network = nn.Sequential(self.featurizer, self.classifier)
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=self.hparams["lr"],
#             weight_decay=self.hparams['weight_decay']
#         )

#     def update(self, minibatches, unlabeled=None):
#         recon_weight = (self.hparams['recon_lambda'] if self.update_count
#                           >= self.hparams['recon_penalty_anneal_iters'] else
#                           1.0)
                          
                          
#         all_x = torch.cat([x for x,y in minibatches])
#         all_features = self.featurizer(all_x)
#         all_y = torch.cat([y for x,y in minibatches])
#         loss = F.cross_entropy(self.predict(all_x), all_y)
        
#         #### Reconstruction loss
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         d_model = Decoder().to(device)
#         latent_code = all_features 
#         batch_data =  all_x[:, :, :-1, :-1] ### same variabel but since 27x27 so just delete the last dimension
#         recon_loss = nn.MSELoss().to(device)
#         loss_recon = recon_loss(batch_data,d_model(latent_code))
        
        
#         loss += loss_recon*recon_weight
        
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()

#         return {'loss': loss.item()}

#     def predict(self, x):
#         return self.network(x)
