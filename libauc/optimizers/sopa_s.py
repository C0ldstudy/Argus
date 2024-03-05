import torch
import math

class SOPAs(torch.optim.Optimizer):
    r"""A wrapper class for different optimizing methods.

        Arguments:
            params (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
            lr (float): learning rate
            loss_fn: the instance of loss class
            method (str): optimization method
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

            Arguments for SGD optimization method:
                momentum (float, optional): momentum factor (default: 0.9)
                dampening (float, optional): dampening for momentum (default: 0.1)
                nesterov (bool, optional): enables Nesterov momentum (default: False)
            Arguments for ADAM optimization method:
                betas (Tuple[float, float], optional): coefficients used for computing
                    running averages of gradient and its square (default: (0.9, 0.999))
                eps (float, optional): term added to the denominator to improve
                    numerical stability (default: 1e-8)
                amsgrad (boolean, optional): whether to use the AMSGrad variant of this
                    algorithm from the paper `On the Convergence of Adam and Beyond`_
                    (default: False)
    """
    
    def __init__(self, model, loss_fn,
                 mode = 'adam',
                 lr=1e-4, weight_decay=0, 
                 momentum=0.0, nesterov=False, dampening=0, # sgd
                 betas=(0.9, 0.999), eps=1e-8, amsgrad=False  # adam
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
        self.loss_fn = loss_fn
        self.lr = lr
        self.steps = 0
        self.mode = mode.lower()
        
        defaults = dict(lr=lr, weight_decay=weight_decay,
                        momentum=momentum, dampening=dampening, nesterov=nesterov,
                        betas=betas, eps=eps, amsgrad=amsgrad)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super(SOPAs, self).__init__(self.params, defaults)
        

    def __setstate__(self, state):
      r"""
      # Set default options for sgd mode and adam mode
      """
      super(SOPAs, self).__setstate__(state)
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
            if self.mode == 'sgd':
               weight_decay = group['weight_decay']
               momentum = group['momentum']
               dampening = group['dampening']
               nesterov = group['nesterov']
               self.lr = group['lr']  
               for p in group['params']:
                  if p.grad is None:
                      print(p.shape)
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

            elif self.mode == 'adam':
                self.lr = group['lr']
                for p in group['params']:
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

                    # Decay the first and second moment running average coefficient
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
                    
        self.steps += 1
        return loss

    def update_lr(self, decay_factor=None, coef_decay_factor=None):
        if decay_factor != None:
           self.param_groups[0]['lr'] =  self.param_groups[0]['lr']/decay_factor
           print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if coef_decay_factor != None:
            self.loss_fn.update_coef(coef_decay_factor)
            print ('Reducing eta/gamma to %.4f @ T=%s!' % (self.loss_fn.get_coef, self.steps))

    def update_regularizer(self, decay_factor=None): 
        pass

