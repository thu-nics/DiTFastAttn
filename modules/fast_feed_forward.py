import torch.nn as nn
import torch

class FastFeedForward(nn.Module):
    def __init__(self,net,steps_method):
        super().__init__()
        self.net=net
        self.steps_method=steps_method
        self.stepi=None
        self.cache_output=None
    
    def forward(self,hidden_states):
        out=hidden_states
        method=self.steps_method[self.stepi]
        if method=="output_share":
            out=self.cache_output
        elif "cfg_attn_share" in method:
            batch_size=hidden_states.shape[0]
            out=out[:batch_size//2]
            for module in self.net:
                out=module(out)
            out=torch.cat([out, out], dim=0)
            self.cache_output=out
        else:
            for module in self.net:
                out=module(out)
            self.cache_output=out
        self.stepi+=1
        return out
