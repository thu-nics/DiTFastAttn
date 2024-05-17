import torch
from dit_fast_attention import FastAttnProcessor,Attention
import numpy as np
import thop
from thop.profile import *


global_attn_count=0
def count_flops_attn(m:Attention, i, o):
    hidden_states=i[0]
    encoder_hidden_states=None
    if len(i)>1:
        encoder_hidden_states=i[1]
    if len(i)>2:
        attention_mask=i[2]
        assert attention_mask is None
    
    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    batch_size, q_seq_len, dim = hidden_states.size()
    
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif m.norm_cross:
        encoder_hidden_states = m.norm_encoder_hidden_states(encoder_hidden_states)
    batch_size,kv_seq_len,dim=encoder_hidden_states.size()

    ops_qk=q_seq_len*kv_seq_len*m.heads*batch_size*dim//m.heads
    ops_kv=q_seq_len*dim*batch_size*kv_seq_len

    if isinstance(m.processor, FastAttnProcessor):
        processor:FastAttnProcessor=m.processor
        method=processor.steps_method[m.stepi-1]
        
        ws=processor.window_size[1]-processor.window_size[0]
        
        if method=="full_attn":
            if processor.need_compute_residual[m.stepi-1]:
                ops_qk*=(1+ws/kv_seq_len)
                ops_kv*=(1+ws/kv_seq_len)
        elif method=="full_attn+cfg_attn_share":
            ops_qk/=2
            if processor.need_compute_residual[m.stepi-1]:
                ops_qk*=(1+ws/kv_seq_len)
                ops_kv*=(1+ws/kv_seq_len)
        elif method=="residual_window_attn":
            ops_qk*=(ws/kv_seq_len)
            ops_kv*=(ws/kv_seq_len)
        elif method=="residual_window_attn+cfg_attn_share":
            ops_qk*=(ws/kv_seq_len/2)
            ops_kv*=(ws/kv_seq_len)
        elif method=="output_share":
            ops_qk=0
            ops_kv=0
        else:
            raise NotImplementedError

    
    matmul_ops=ops_qk+ops_kv
    assert matmul_ops>=0
    if not hasattr(m, "total_ops"):
        m.total_ops = torch.DoubleTensor([matmul_ops])
    else:
        m.total_ops += torch.DoubleTensor([matmul_ops])
    global global_attn_count
    global_attn_count+=matmul_ops

def profile_pipe_transformer(
    pipe,
    inputs,
    kwargs,
    custom_ops=None,
    verbose=True,
    ret_layer_info=False,
    report_missing=False,
):
    model: nn.Module=pipe.transformer
    handler_collection = {}
    types_collection = set()
    if custom_ops is None:
        custom_ops = {}
    if report_missing:
        # overwrite `verbose` option when enable report_missing
        verbose = True

    def add_hooks(m: nn.Module):
        m.register_buffer("total_ops", torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        # for p in m.parameters():
        #     m.total_params += torch.DoubleTensor([p.numel()])

        m_type = type(m)

        fn = None
        if m_type in custom_ops:
            # if defined both op maps, use custom_ops to overwrite.
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Customize rule %s() %s." % (fn.__qualname__, m_type))
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print("[INFO] Register %s() for %s." % (fn.__qualname__, m_type))
        else:
            if m_type not in types_collection and report_missing:
                prRed(
                    "[WARN] Cannot find rule for %s. Treat it as zero Macs and zero Params."
                    % m_type
                )

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )
        types_collection.add(m_type)

    prev_training_status = model.training

    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        pipe(*inputs,**kwargs)

    def dfs_count(module: nn.Module, prefix="\t"):
        total_ops, total_params = module.total_ops.item(), 0
        ret_dict = {}
        for n, m in module.named_children():
            # if not hasattr(m, "total_ops") and not hasattr(m, "total_params"):  # and len(list(m.children())) > 0:
            #     m_ops, m_params = dfs_count(m, prefix=prefix + "\t")
            # else:
            #     m_ops, m_params = m.total_ops, m.total_params
            next_dict = {}
            if m in handler_collection and not isinstance(
                m, (nn.Sequential, nn.ModuleList, Attention)
            ):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params
        # print(prefix, module._get_name(), (total_ops, total_params))
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model)

    # reset model to original status
    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    if ret_layer_info:
        return total_ops, total_params, ret_dict
    return total_ops, total_params


def calculate_flops(pipe,x, n_steps):
    macs_without_attn, params, ret_dict = profile_pipe_transformer(pipe, inputs=(x, ), kwargs={"num_inference_steps": n_steps},verbose=0,ret_layer_info=True)
    macs, params,ret_dict2 = profile_pipe_transformer(pipe, inputs=(x, ), kwargs={"num_inference_steps": n_steps},
                        custom_ops={Attention: count_flops_attn},verbose=0,ret_layer_info=True)
    print(f"macs is {macs/1e9}G, macs_without_attn is {macs_without_attn/1e9}G, attn is {(macs-macs_without_attn)/1e9}G")
    print(f"global_attn_count is {global_attn_count/1e9}G")
    attn_mac=(macs-macs_without_attn)
    return macs/1e9,attn_mac/1e9