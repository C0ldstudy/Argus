import torch 

class PESG(torch.optim.Optimizer):
    """Proximal Epoch Stochastic Gradient (PESG) 
    
    Reference: 
        Yuan, Z., Yan, Y., Sonka, M. and Yang, T., 
        Large-scale Robust Deep AUC Maximization: A New Surrogate Loss and Empirical Studies on Medical Image Classification. 
        International Conference on Computer Vision (ICCV 2021)
    Link:
        https://arxiv.org/abs/2012.03173
    """
    def __init__(self, 
                 model, 
                 loss_fn=None,
                 a=None,          # to be deprecated
                 b=None,          # to be deprecated
                 alpha=None,      # to be deprecated 
                 margin=1.0, 
                 lr=0.1, 
                 gamma=None,      # to be deprecated 
                 clip_value=1.0, 
                 weight_decay=1e-5, 
                 epoch_decay=2e-3, # default: gamma=500
                 momentum=0, 
                 verbose=True,
                 device=None,
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
        self.lr = lr
        self.gamma = gamma  # to be deprecated
        self.clip_value = clip_value
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.epoch_decay = epoch_decay
               
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
        if self.a is not None and self.b is not None:
           self.params = get_parameters(list(model.parameters())+[self.a, self.b])
        else:
           self.params = get_parameters(list(model.parameters()))
        self.defaults = dict(lr=self.lr, 
                             margin=margin, 
                             a=self.a, 
                             b=self.b,
                             alpha=self.alpha,
                             clip_value=clip_value,
                             momentum=momentum,
                             weight_decay=weight_decay,
                             epoch_decay=epoch_decay,
                             model_ref=self.model_ref,
                             model_acc=self.model_acc
                             )
        
        super(PESG, self).__init__(self.params, self.defaults)
         
    def __setstate__(self, state):
        super(PESG, self).__setstate__(state)
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
            momentum = group['momentum']
            self.lr =  group['lr']
            
            epoch_decay = group['epoch_decay']
            model_ref = group['model_ref']
            model_acc = group['model_acc']
            
            m = group['margin'] 
            a = group['a']
            b = group['b']
            alpha = group['alpha']
            
            # updates
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = torch.clamp(p.grad.data , -clip_value, clip_value) + epoch_decay*(p.data - model_ref[i].data) + weight_decay*p.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(1-momentum).add_(d_p, alpha=momentum)
                    d_p =  buf
                p.data = p.data - group['lr']*d_p
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
        
    def update_lr(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
            
    def update_regularizer(self, decay_factor=None):
        if decay_factor != None:
            self.param_groups[0]['lr'] = self.param_groups[0]['lr']/decay_factor
            if self.verbose:
               print ('Reducing learning rate to %.5f @ T=%s!'%(self.param_groups[0]['lr'], self.steps))
        if self.verbose:    
           print ('Updating regularizer @ T=%s!'%(self.steps))
        for i, param in enumerate(self.model_ref):
            self.model_ref[i].data = self.model_acc[i].data/self.T
        for i, param in enumerate(self.model_acc):
            self.model_acc[i].data = torch.zeros(param.shape, dtype=torch.float32, device=self.device,  requires_grad=False).to(self.device)
        self.T = 0
        
  
