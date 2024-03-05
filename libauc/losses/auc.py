"""
If you find our library helpful, please acknowledge our library and cite the following papers:
 @misc{libauc2022,
      title={LibAUC: A Deep Learning Library for X-Risk Optimization.},
      author={Zhuoning Yuan, Zi-Hao Qiu, Gang Li, Dixian Zhu, Zhishuai Guo, Quanqi Hu, Bokun Wang, Qi Qi, Yongjian Zhong, Tianbao Yang},
      year={2022}
	}

"""

import torch
import torch.nn.functional as F
from .surrogate import squared_loss, squared_hinge_loss, logistic_loss


def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss


class AUCMLoss_V2(torch.nn.Module):
    """AUC-Margin Loss: a novel loss function to optimize AUROC

    Args:
        margin: margin for AUCM loss, e.g., m in [0, 1]

    Return:
        loss value (scalar)

    Reference:
            Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification,
            Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao,
            Proceedings of the IEEE/CVF International Conference on Computer Vision 2021.

    """
    def __init__(self, margin=1.0, device=None):
        super(AUCMLoss_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device,  requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        pos_mask = (1==y_true).float()
        neg_mask = (0==y_true).float()
        loss = self.mean((y_pred - self.a)**2*pos_mask) + \
               self.mean((y_pred - self.b)**2*neg_mask) + \
               2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask)) - \
               self.alpha**2
        return loss

class AUCM_MultiLabel_V2(torch.nn.Module):
    """AUC-Margin Loss (Multi-Task): a novel loss function to optimize AUROC

    Args:
        margin: margin for AUCM loss, e.g., m in [0, 1]

    Return:
        loss value (scalar)

    Reference:
            Large-scale robust deep auc maximization: A new surrogate loss and empirical studies on medical image classification,
            Yuan, Zhuoning and Yan, Yan and Sonka, Milan and Yang, Tianbao,
            Proceedings of the IEEE/CVF International Conference on Computer Vision 2021.

    """
    def __init__(self, margin=1.0, num_classes=10, device=None):
        super(AUCM_MultiLabel_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.num_classes = num_classes
        self.a = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(num_classes, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)

    @property
    def get_a(self):
        return self.a.mean()

    @property
    def get_b(self):
        return self.b.mean()

    @property
    def get_alpha(self):
        return self.alpha.mean()

    # this function (torch.count_nonzero(tensor)) only works for torch > 1.10.0
    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        total_loss = 0
        for idx in range(self.num_classes):
            y_pred_i = _check_tensor_shape(y_pred[:, idx], (-1,1))
            y_true_i = _check_tensor_shape(y_true[:, idx], (-1,1))
            pos_mask = (1==y_true_i).float()
            neg_mask = (0==y_true_i).float()
            loss = self.mean((y_pred_i - self.a[idx])**2*pos_mask) + \
                   self.mean((y_pred_i - self.b[idx])**2*neg_mask) + \
                   2*self.alpha[idx]*(self.margin + self.mean(y_pred_i*neg_mask) - self.mean(y_pred_i*pos_mask)) - \
                   self.alpha[idx]**2
            total_loss += loss
        return total_loss

class CompositionalAUCLoss_V2(torch.nn.Module):
    """Compositional AUC Loss: a novel loss function to directly optimize AUROC

        Args:
            margin: margin term for AUCM loss, e.g., m in [0, 1]
            backend: which loss to optimize in the backend, e.g., AUCM loss, CrossEntropy loss

        Return:
            loss value (scalar)

        Reference:
                Compositional Training for End-to-End Deep AUC Maximization,
                Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang,
                International Conference on Learning Representations 2022

    """
    def __init__(self, margin=1.0, backend='ce', device=None):
        super(CompositionalAUCLoss_V2, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.margin = margin
        self.a = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.b = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.alpha = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True).to(self.device)
        self.L_AVG = F.binary_cross_entropy_with_logits  # with sigmoid
        self.backend = 'ce'

    def mean(self, tensor):
        return torch.sum(tensor)/torch.count_nonzero(tensor)

    def forward(self, y_pred, y_true):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        if self.backend == 'ce':
           self.backend = 'auc'
           return self.L_AVG(y_pred, y_true)
        else:
           self.backend = 'ce'
           y_pred = torch.sigmoid(y_pred)
           pos_mask = (1==y_true).float()
           neg_mask = (0==y_true).float()
           self.L_AUC = self.mean((y_pred - self.a)**2*pos_mask) + \
                        self.mean((y_pred - self.b)**2*neg_mask) + \
                        2*self.alpha*(self.margin + self.mean(y_pred*neg_mask) - self.mean(y_pred*pos_mask) ) - \
                        self.alpha**2
           return self.L_AUC


class APLoss(torch.nn.Module):
    """AP Loss with squared-hinge function: a novel loss function to directly optimize AUPRC.

        Args:
            margin: margin for squred hinge loss, e.g., m in [0, 1]
            gamma: factors for moving average

        Return:
            loss value (scalar)

        Reference:
                Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence},
                Qi, Qi and Luo, Youzhi and Xu, Zhao and Ji, Shuiwang and Yang, Tianbao,
                Advances in Neural Information Processing Systems 2021.

        Notes: This version of AP loss reduces redundant computation for the original implementation by Qi Qi.
        In addition, it fixed a potential memory leaking issue related to 'u_all' and 'u_pos'. This version is contributed by Gang Li.
    """

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge', device=None):
        super(APLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cpu')

        else:
            self.device = device
        self.u_all = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        CHECKMEORY=False
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p>=0] # only need indices from positive samples

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))#.detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        pos_sur_loss = sur_loss * pos_mask
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_all[index_p] = (1 - self.gamma) * self.u_all[index_p] + self.gamma * (sur_loss.mean(1, keepdim=True)).detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        p = (self.u_pos[index_p] - (self.u_all[index_p]) * pos_mask) / (self.u_all[index_p] ** 2) # size of p: len(f_ps)* len(y_pred)
        # p.detach_()
        loss = torch.mean(p * sur_loss)
        # loss = (p * sur_loss).mean()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        # del sur_loss, pos_sur_loss, p
        return loss


class pAUC_CVaR_Loss(torch.nn.Module):
    """Partial AUC Loss: a stochastic one-way partial AUC based on DRO-CVaR

      Args:
        pos_len: number of positive examples for the training data
        num_neg: number of negative samples for each mini-batch
        margin: margin for basic AUC loss
        beta: FPR upper bound for pAUC used for SOTA
        eta: stepsize for CVaR regularization term
        loss type: basic AUC loss to apply.

        Return:
            loss value (scalar)

        Reference:
            Zhu, D., Li, G., Wang, B., Wu, X., and Yang, T., 2022.
            When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee.
            arXiv preprint arXiv:2203.00176.

    """
    def __init__(self,
                 pos_len,
                 num_neg=None,
                 margin=1.0,
                 alpha=0,
                 beta=0.2,
                 surrogate_loss='squared_hinge'):
        super(pAUC_CVaR_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eta = 1.0 # change this from optimizer
        self.num_neg = num_neg
        self.pos_len = pos_len
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).cuda()
        self.margin = margin
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def set_coef(self, eta):
        self.eta = eta

    def update_coef(self, decay_factor):
        self.eta = self.eta/decay_factor

    @property
    def get_coef(self):
        return self.eta

    def forward(self, y_pred, y_true, index_p, auto=True):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))

        f_ps = y_pred[y_true == 1].reshape(-1, 1)
        f_ns = y_pred[y_true == 0].reshape(-1, 1)
        f_ps = f_ps.repeat(1, len(f_ns))
        f_ns = f_ns.repeat(1, len(f_ps))
        index_p = index_p[index_p>=0]
        if auto:
           self.num_neg = (y_true == 0).float().sum()
           assert self.num_neg > 0
        loss = self.surrogate_loss(self.margin, f_ps - f_ns.transpose(0,1)) # return element-wsie loss
        p = loss > self.u_pos[index_p]
        self.u_pos[index_p] = self.u_pos[index_p]-self.eta/self.pos_len*(1 - p.sum(dim=1, keepdim=True)/(self.beta*self.num_neg))
        p.detach_()
        loss = torch.mean(p * loss) / self.beta
        return loss


class pAUC_DRO_Loss(torch.nn.Module):
    """Partial AUC Loss: a stochastic one-way partial AUC based KLDRO-based
        Args:
            pos_len (int): the size of positive samples in training dataset
            margin (float): margin for squred hinge loss, e.g., m in [0, 1]
            beta (float): factor for moving average
            lambda : the weight for KL divergence

        Return:
            loss value (scalar)

        Reference:
            Zhu, D., Li, G., Wang, B., Wu, X., and Yang, T., 2022.
            When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee.
            arXiv preprint arXiv:2203.00176.
    """
    def __init__(self,
                 pos_len,
                 gamma=0.1,
		 margin=1.0,
                 Lambda=1.0,
                 surrogate_loss='squared_hinge',
                 device=None):
        super(pAUC_DRO_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.u_pos = torch.tensor([0.0]*pos_len).view(-1, 1).to(self.device)
        self.margin = margin
        self.gamma = gamma # moving average
        self.Lambda = Lambda
        # surrogate
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def set_coef(self, gamma):
        self.gamma = gamma

    def update_coef(self, decay_factor):
        self.gamma /= decay_factor

    @property
    def get_coef(self):
        return self.gamma

    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))

        f_ps = y_pred[y_true == 1].reshape(-1,1)
        f_ns = y_pred[y_true == 0].reshape(-1)
        index_p = index_p[index_p>=0]
        mat_data = f_ns.repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))
        exp_loss = torch.exp(sur_loss/self.Lambda)
        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (exp_loss.mean(1, keepdim=True).detach())
        p = exp_loss/self.u_pos[index_p] # (len(f_ps), len(f_ns))
        p.detach_()
        loss = torch.mean(p * sur_loss)
        return loss


class tpAUC_KL_Loss(torch.nn.Module):
    """Partial AUC Loss: a stochastic two-way partial AUC based on DRO-KL. In this formulation, we implicitly handle the \alpha and \beta range of PAUC          by tuning \lambda and \tau.

      Args:
        pos_len: number of positive examples for the training data
        num_neg: number of negative samples for each mini-batch
        margin: margin for basic AUC loss
        Lambda: KL regularization for negative samples
        tau: KL regularization for positive samples
        beta_0: stepsize for negative sample KL regularization term (implemented in optimizer)
        beta_1: stepsize for positive sample KL regularization term  (implemented in optimizer)
      Return:
            loss value (scalar)

      Reference:
            Zhu, D., Li, G., Wang, B., Wu, X., and Yang, T., 2022.
            When AUC meets DRO: Optimizing Partial AUC for Deep Learning with Non-Convex Convergence Guarantee.
            arXiv preprint arXiv:2203.00176.

    """
    def __init__(self,
                 pos_len,
		 tau=1.0,
                 Lambda=1.0,
                 margin=1.0,
                 surrogate_loss='squared_hinge',
		 device=None):
        super(tpAUC_KL_Loss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.gamma0 = 0.9 # change this from optimizer
        self.gamma1 = 0.9 # change this from optimizer
        self.Lambda = Lambda
        self.tau = tau
        self.u_pos = torch.tensor([0.0]*pos_len).view(-1, 1).to(self.device)
        self.w = 0.0
        self.margin = margin
        # surrogate
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def set_coef(self, gamma0, gamma1):
        self.gamma0 = gamma0
        self.gamma1 = gamma1

    def update_coef(self, decay_factor):
        self.gamma0 = self.gamma0/decay_factor
        self.gamma1 = self.gamma1/decay_factor

    @property
    def get_coef(self):
        return (self.gamma0, self.gamma1)

    def forward(self, y_pred, y_true, index_p):
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))

        f_ps = y_pred[y_true == 1].reshape(-1, 1)
        f_ns = y_pred[y_true == 0].reshape(-1, 1)
        f_ps = f_ps.repeat(1, len(f_ns))
        f_ns = f_ns.repeat(1, len(f_ps))
        index_p = index_p[index_p>=0]
        loss = self.surrogate_loss(self.margin, f_ps - f_ns.transpose(0,1)) # return element-wsie loss

        exp_loss = torch.exp(loss/self.Lambda).detach_()
        self.u_pos[index_p] = (1 - self.gamma0) * self.u_pos[index_p] + self.gamma0 * (exp_loss.mean(1, keepdim=True))
        self.w = (1 - self.gamma1) * self.w + self.gamma1 * (torch.pow(self.u_pos[index_p], self.Lambda/self.tau).mean())
        p = torch.pow(self.u_pos[index_p], self.Lambda/self.tau - 1) * exp_loss/self.w
        p.detach_()
        loss = torch.mean(p * loss)
        return loss

class pAUCLoss(torch.nn.Module):
    """A wrapper of calling different partial AUC losses. The list of supported functions includes:
         - SOPA: pAUC_CVaR_Loss (one-way)
         - SOPAs: pAUC_KLDRO_Loss (one-way)
         - SOTAs: tpAUC_KL_Loss (two-way)
    """
    def __init__(self, backend='SOPA', **kwargs):
        super(pAUCLoss, self).__init__()
        assert backend in ['SOPA', 'SOPAs', 'SOTA'], 'Keyword is not found!'
        print ('Backend loss: %s'%backend)
        self.backend = backend
        self.loss_fn = self.get_loss(backend, **kwargs)

    def get_loss(self, backend='SOPA', **kwargs):
        if backend == 'SOPA':
           loss = pAUC_CVaR_Loss(**kwargs)
        elif backend == 'SOPAs':
           loss = pAUC_KLDRO_Loss(**kwargs)
        elif backend == 'SOTA':
           loss = tpAUC_KL_Loss(**kwargs)
        else:
            raise ValueError('Out of options!')
        return loss

    def forward(self, y_pred, y_true, index_p):
        return self.loss_fn(y_pred, y_true, index_p)




# alias
pAUC_Loss = pAUCLoss

# experimental functions
AUCM_MultiLabel_V2
AUCMLoss_V2
CompositionalAUCLoss_V2


