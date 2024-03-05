import math
import torch

class SOTAs(torch.optim.Optimizer):
    r"""Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    The implementation of the L2 penalty follows changes proposed in
    `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, model, loss_fn, 
                 mode = 'adam',
                 lr=1e-3, weight_decay=0,
                 gammas=(0.9, 0.9), 
                 betas=(0.9, 0.999), eps=1e-8,  amsgrad=False, # adam
                 momentum=0.9, nesterov=False, dampening=0,  # sgd
                 ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
            
        try: 
            params = model.parameters() 
        except:
            params = model # if model is already params 
            
        self.params = params
        self.lr = lr
        self.mode = mode.lower()
        self.loss_fn = loss_fn
        self.loss_fn.set_coef(gamma0=gammas[0], gamma1=gammas[1])
        self.steps = 0
        
        defaults = dict(lr=lr, betas=betas, eps=eps, momentum=momentum, nesterov=nesterov, dampening=dampening, 
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(SOTAs, self).__init__(self.params, defaults)

        
    def __setstate__(self, state):
      r"""
      # Set default options for sgd mode and adam mode
      """
      super(SOTAs, self).__setstate__(state)
      for group in self.param_groups:
          if self.mode == 'sgd':
             group.setdefault('nesterov', False)
          elif self.mode == 'adam':
             group.setdefault('amsgrad', False)
          else:
             NotImplementedError

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self.mode == 'adam':
              self.lr = group['lr']
              for i, p in enumerate(group['params']):
                  if p.grad is None:
                      continue
                  grad = p.grad
                  if grad.is_sparse:
                      raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                  amsgrad = group['amsgrad']
                  state = self.state[p]
                  # State initialization
                  if len(state) == 0:
                      state['step'] = 0
                      # Exponential moving average of gradient values
                      state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                      # Exponential moving average of squared gradient values
                      state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                      if amsgrad:
                          # Maintains max of all exp. moving avg. of sq. grad. values
                          state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                  exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                  if amsgrad:
                      max_exp_avg_sq = state['max_exp_avg_sq']
                  beta1, beta2 = group['betas']
                  state['step'] += 1
                  bias_correction1 = 1 - beta1 ** state['step']
                  bias_correction2 = 1 - beta2 ** state['step']
                  if group['weight_decay'] != 0:
                      grad = grad.add(p, alpha=group['weight_decay'])
                  exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                  exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                  if amsgrad:
                      # Maintains the maximum of all 2nd moment running avg. till now
                      torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                      # Use the max. for normalizing running avg. of gradient
                      denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                  else:
                      denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                  step_size = group['lr'] / bias_correction1
                  p.addcdiv_(exp_avg, denom, value=-step_size)
            elif self.mode == 'sgd':
              weight_decay = group['weight_decay']
              momentum = group['momentum']
              dampening = group['dampening']
              nesterov = group['nesterov']
              self.lr = group['lr']
              for p in group['params']:
                  if p.grad is None:
                      continue
                  d_p = p.grad
                  if weight_decay != 0:
                      d_p = d_p.add(p, alpha=weight_decay) # d_p = (d_p + p*weight_decy)
                  if momentum != 0:
                      param_state = self.state[p]
                      if 'momentum_buffer' not in param_state:
                          buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                      else:
                          buf = param_state['momentum_buffer']
                          buf.mul_(momentum).add_(d_p, alpha=1 - dampening) # [v = v*beta + d_p ] --> new d_p
                      if nesterov:
                          d_p = d_p.add(buf, alpha=momentum)
                      else:
                          d_p = buf
                  p.add_(d_p, alpha=-group['lr'])
        self.steps += 1  
        return loss
    
    def update_lr(self, decay_factor=None, coef_decay_factor=None ):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if coef_decay_factor != None:
            self.loss_fn.update_coef(coef_decay_factor)
            coefs = self.loss_fn.get_coef
            coefs = '(%.4f, %.4f)'%(coefs[0], coefs[1])
            print ('Reducing eta/gamma to %s @ T=%s!' % (coefs, self.steps))

