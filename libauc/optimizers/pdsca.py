import torch 
import copy

class PDSCA(torch.optim.Optimizer):
    """
    Reference:
    @inproceedings{
                    yuan2022compositional,
                    title={Compositional Training for End-to-End Deep AUC Maximization},
                    author={Zhuoning Yuan and Zhishuai Guo and Nitesh Chawla and Tianbao Yang},
                    booktitle={International Conference on Learning Representations},
                    year={2022},
                    url={https://openreview.net/forum?id=gPvB4pdu_Z}
                    }
    """
    def __init__(self, 
                 model, 
                 loss_fn=None,
                 a=None,        # to be deprecated
                 b=None,        # to be deprecated
                 alpha=None,    # to be deprecated 
                 margin=1.0, 
                 lr=0.1, 
                 lr0=None,
                 gamma=None,    # to be deprecated 
                 beta1=0.99,
                 beta2=0.999,
                 clip_value=1.0, 
                 weight_decay=1e-5, 
                 epoch_decay=2e-3, #gamma=500
                 verbose=True,
                 device='cuda',
                 **kwargs):
        
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:  
            self.device = device      
        assert (gamma is None) or (epoch_decay is None), 'You can only use one of gamma and epoch_decay!'
        if gamma is not None:
           assert gamma > 0
           epoch_decay = 1/gamma
        
        self.margin = margin
        self.model = model
        if lr0 is None:
           lr0 = lr
        self.lr = lr
        self.lr0 = lr0
        self.gamma = gamma
        self.clip_value = clip_value
        self.weight_decay = weight_decay
        self.epoch_decay = epoch_decay
        
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.loss_fn = loss_fn
        if loss_fn != None:
            try:
                self.a = loss_fn.a 
                self.b = loss_fn.b 
                self.alpha = loss_fn.alpha 
            except:
                print('AUCLoss is not found!')
        else:
            self.a = a 
            self.b = b 
            self.alpha = alpha     
            
        self.model_ref = self.init_model_ref()
        self.model_acc = self.init_model_acc()

        self.T = 0                # for epoch_decay
        self.steps = 0            # total optim steps
        self.verbose = verbose    # print updates for lr/regularizer

        def get_parameters(params):
            for p in params:
                yield p
        if self.a is not None or self.b is not None:
           self.params = get_parameters(list(model.parameters())+[self.a, self.b])
        else:
           self.params = get_parameters(list(model.parameters()))
        self.defaults = dict(lr=self.lr, 
                             lr0=self.lr0,
                             margin=margin, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=self.clip_value,
                             weight_decay=self.weight_decay,
                             epoch_decay=self.epoch_decay,
                             beta1=self.beta1,
                             beta2=self.beta2,
                             model_ref=self.model_ref,
                             model_acc=self.model_acc)
        
        super(PDSCA, self).__init__(self.params, self.defaults)

    def __setstate__(self, state):
        super(PDSCA, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def init_model_ref(self):
         self.model_ref = []
         for var in list(self.model.parameters())+[self.a, self.b]: 
            if var is not None:
               self.model_ref.append(torch.empty(var.shape).normal_(mean=0, std=0.01).to(self.device))
         return self.model_ref
     
    def init_model_acc(self):
        self.model_acc = []
        for var in list(self.model.parameters())+[self.a, self.b]: 
            if var is not None:
               self.model_acc.append(torch.zeros(var.shape, dtype=torch.float32,  device=self.device, requires_grad=False).to(self.device)) 
        return self.model_acc
    
    @property    
    def optim_steps(self):
        return self.steps
    
    @property
    def get_params(self):
        return list(self.model.parameters())

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
 
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            clip_value = group['clip_value']
            self.lr =  group['lr']
            self.lr0 = group['lr0']
            
            epoch_decay = group['epoch_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            
            m = group['margin']
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            for i, p in enumerate(group['params']):
                if p.grad is None: 
                   continue
                d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                if alpha.grad is None: # sgd + moving p. # TODO: alpha=None mode
                    p.data = p.data - group['lr0']*d_p 
                    if beta1!= 0: 
                        param_state = self.state[p]
                        if 'weight_buffer' not in param_state:
                            buf = param_state['weight_buffer'] = torch.clone(p).detach()
                        else:
                            buf = param_state['weight_buffer']
                            buf.mul_(1-beta1).add_(p, alpha=beta1)
                        p.data =  buf.data # Note: use buf(s) to compute the gradients w.r.t AUC loss can lead to a slight worse performance 
                elif alpha.grad is not None: # auc + moving g. # TODO: alpha=None mode
                   if beta2!= 0: 
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(1-beta2).add_(d_p, alpha=beta2)
                        d_p =  buf
                   p.data = p.data - group['lr']*d_p 
                else:
                    NotImplementedError 
                model_acc[i].data = model_acc[i].data + p.data
                
            if alpha is not None: 
               if alpha.grad is not None: 
                  alpha.data = alpha.data + group['lr']*(2*(m + b.data - a.data)-2*alpha.data)
                  alpha.data  = torch.clamp(alpha.data,  0, 999)
              
        self.T += 1        
        self.steps += 1
        return loss

    def zero_grad(self):
        self.model.zero_grad()
        if self.a is not None and self.b is not None:
           self.a.grad = None
           self.b.grad = None
        if self.alpha is not None:
           self.alpha.grad = None
        
    def update_lr(self, decay_factor=None, decay_factor0=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'],  self.steps))
        if decay_factor0 != None:
            self.param_groups[0]['lr0'] = self.param_groups[0]['lr0']/decay_factor0
            if self.verbose:
               print ('Reducing learning rate (inner) to %.5f @ T=%s!'%(self.param_groups[0]['lr0'], self.steps))
            
    def update_regularizer(self, decay_factor=None, decay_factor0=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if decay_factor0 != None:
            self.param_groups[0]['lr0'] = self.param_groups[0]['lr0']/decay_factor0
            if self.verbose:
               print ('Reducing learning rate (inner) to %.5f @ T=%s!'%(self.param_groups[0]['lr0'], self.steps))
        if self.verbose:
           print ('Updating regularizer @ T=%s!'%(self.steps))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0
        
        
