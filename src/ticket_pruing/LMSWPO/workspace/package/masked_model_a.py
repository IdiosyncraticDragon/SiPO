def make_trainable(self):
      self.maksed_model = self.masked_model.train()
      self.maksed_model.generator = self.masked_model.generator.train()
      self.masked_model.zero_grad()
      '''
      for name, module_tensor in self.masked_model.named_parameters():
         if self.pretrained_model_dict[name].is_leaf and (not module_tensor.is_leaf):
            #module_tensor.detach().requires_grad_()
            module_tensor.retain_grad()
            module_tensor.requires_grad = True
            # module_tensor.requires_grad_()
            module_tensor.data = Variable(module_tensor.data, requires_grad = True)
         print(module_tensor.is_leaf)
      '''
      # adjusting operation for training
      if self.pre_forward_fn is None:
        self.pre_forward_fn = self.masked_model.register_forward_pre_hook(self._apply_mask)
      # updating the pruning connection weights
      #self.forward_fn = self.masked_model.register_forward_hook(self._updateweights_ignore_mask)


1.AttributeError: 'Variable' object has no attribute 'requires_grad_'
2.
3.RuntimeError: you can only change requires_grad flags of leaf variables.
4. AttributeError: 'Variable' object has no attribute 'requires_grad_'
5.RuntimeError: Variable data has to be a tensor, but got Variable

# try
      # print("model id", id(self.masked_model))
      # for name, module_tensor in self.masked_model.named_parameters():
      #    print("name ", name)
         
      #    # print(self.pretrained_model_dict[name].is_leaf)
      #    # print(self.pretrained_model_dict[name].requires_grad)
      #    # print(module_tensor.is_leaf)
      #    # print(module_tensor.requires_grad)
      #    # 从输出看， is_leaf = False
      #    # requires_grad = True
      #    if self.pretrained_model_dict[name].is_leaf and (not module_tensor.is_leaf):
      #       # AttributeError: 'Variable' object has no attribute 'requires_grad_'
      #       #module_tensor.detach().requires_grad_(), XXX
      #       # 对修复 is_leaf 无用
      #       # module_tensor.retain_grad()
      #       # RuntimeError: you can only change requires_grad flags of leaf variables.
      #       # module_tensor.requires_grad = True
      #       # AttributeError: 'Variable' object has no attribute 'requires_grad_'
      #       #module_tensor.requires_grad_(),  XXXX
      #       # RuntimeError: Variable data has to be a tensor, but got Variable
      #       print("perform repair")
      #   这个有用
      #       module_tensor = Variable(module_tensor.data, requires_grad = True)
      #    # print("after repair:")
      #    # print(module_tensor.is_leaf)
      #    # print(module_tensor.requires_grad)
      #    print(id(module_tensor))
      #    if self.pretrained_model_dict[name].is_leaf and (not module_tensor.is_leaf):
      #       print("Still need to repair")
      #    if self.pretrained_model_dict[name].is_leaf != module_tensor.is_leaf:
      #       print("Problem Here")