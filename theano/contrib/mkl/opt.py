from __future__ import absolute_import, print_function, division
from six import integer_types
import theano
from theano import gof, tensor, scalar
from theano.compile import optdb
from theano.gof import (local_optimizer, Optimizer, toolbox)
from theano.contrib.mkl import mkl_optimizer, register_opt, mkl_seqopt, mkl_available
from theano.contrib.mkl.basic_ops import (MKLToNdarray, I2UGrad, U2IConv,
                                          U2IPool, U2IRelu, U2ILRN, U2IElemwiseSum, U2IBatchNormalization)
from theano.contrib.mkl import (mkl_conv, mkl_pool, mkl_relu, mkl_elemwise, mkl_lrn,  mkl_bn)
from theano.tensor.basic import Join, Split
from theano.tensor.signal import pool
from theano.tensor.nnet.abstract_conv import (AbstractConv2d,
                                              AbstractConv2d_gradWeights,
                                              AbstractConv2d_gradInputs)

import logging

_logger = logging.getLogger('theano.contrib.mkl.opt')

# global OPT
optdb.register('mkl_opt', mkl_seqopt, 0.09, 'mkl')

# local OPT
mkl_seqopt.register('mkl_local_optimizations', mkl_optimizer, 20,
                    'fast_run', 'fast_compile', 'mkl')


class CutMKLDataConversionChain(Optimizer):
    """
    This global optimizer used to cut the unless data layout transfer from user layout to MKL-DNN internal layout
    from function graph, such as MKLToNdarray + U2IConv.
    """
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def apply(self, fgraph):
        list_forward = ['Relu',
                        'Pool',
                        'Conv2D',
                        'LRN',
                        'ElemwiseSum',
                        'BatchNormalization',
                        'Concatenate']
        list_i2u = ['MKLToNdarray']
        list_u2i = ['U2IPool',
                    'U2IRelu',
                    'U2IConv',
                    'U2IElemwiseSum',
                    'U2ILRN',
                    'U2IBatchNormalization',
                    'U2IConcatenate']
        list_backward = ['ReluGrad',
                         'PoolGrad',
                         'ConvGradInputs',
                         'ConvGradWeights',
                         'LRNGrad',
                         'ElemwiseSum',
                         'BatchNormalizationGrad',
                         'ConcatenateGrad']
        list_i2u_back = ['I2UGrad', 'U2IElemwiseSum']
        list_u2i_back = ['MKLToNdarray']
        try:
            for node in fgraph.toposort():
                # backward
                if node.op.__class__.__name__ in list_i2u_back:
                    out = node.outputs[0]
                    for inp in node.inputs:
                        if isinstance(inp.owner, gof.Apply) and inp.owner.op.__class__.__name__ in list_u2i_back:
                            for inpOP in list(inp.owner.inputs):
                                if isinstance(inpOP.owner, gof.Apply) and inpOP.owner.op.__class__.__name__ in list_backward:
                                    # FIXME, concateGrad Op, has multiple ouputs, so we can't hard code outputs[0] here
                                    # otherwise, it will always reuse outputs[0], while keep others others dangling!!!
                                    # fgraph.replace_validate(out, inpOP.owner.outputs[0])
                                    fgraph.replace_validate(out, inpOP)
                # forward
                if node.op.__class__.__name__ in list_u2i:
                    # FIXME, concate Op hase multiple inputs, U2IConcate Op is a multiple inputs/outputs
                    # So, can't hard code outputs[0] here.
                    out = node.outputs[0]
                    for inp in node.inputs:
                        if isinstance(inp.owner, gof.Apply) and inp.owner.op.__class__.__name__ in list_i2u:
                            for inpOP in list(inp.owner.inputs):
                                if isinstance(inpOP.owner, gof.Apply) and inpOP.owner.op.__class__.__name__ in list_forward:
                                    fgraph.replace_validate(out, inpOP.owner.outputs[0])
        except Exception as e:
            msg = ('Failed to apply global Cut.'
                   'Exception message: %s\n') % str(e)
            _logger.warning(msg)
            return

mkl_seqopt.register('CutMKLDataConversionChain', CutMKLDataConversionChain(),
                    40,
                    'fast_run',
                    'fast_compile',
                    'mkl')


class ReplaceConvBias(Optimizer):
    """
    This global optimizer looks for the pattern AbstractConv2d + Bias in funtion graph.
    Replace AbstractConv2d and Elemwise{Add} OPs with MKLDNN Conv2D OP which can accept
    3 inputs (image, weights, bias) to improve performance.

    The Bias variable is always followed a DimShuffle OP in function graph. This case is
    processed in this optimizer.

    If the pattern AbstractConv2d + Bias is found and replaced in function graph. The
    backward OPs need be optimized respectively. We use ConvGradWeights and ConvGradInputs
    to replace AbstractConv2d_gradWeights and AbstractConv2d_gradInputs. ConvGradWeights
    gives two outputs: gradWeigths and gradBias.

    For the non-bias AbstractConv2d scenario, it will be handled by local optimizations.
    """
    def __init__(self):
        super(ReplaceConvBias, self).__init__()

    def add_requirements(self, fgraph):
        fgraph.attach_feature(toolbox.ReplaceValidate())

    def _check_add_bias_(self, node):
        out = node.outputs
        if (isinstance(out[0].clients[0][0].op, tensor.Elemwise) and
                isinstance(out[0].clients[0][0].op.scalar_op, scalar.Add)):
            if len(out[0].clients[0][0].inputs) == 2:
                if out[0].clients[0][0].inputs[0] is out[0]:
                    bias = out[0].clients[0][0].inputs[1]
                else:
                    bias = out[0].clients[0][0].inputs[0]
                # Get DimShuffle node
                bias_owner = bias.owner
                if bias_owner is None:
                    return bias
                elif isinstance(bias_owner.op, tensor.DimShuffle) and (bias_owner.inputs[0].owner is None):
                    return bias_owner.inputs[0]
                else:
                    return None

        return None

    def _check_grad_bias_(self, node, i):
        assert len(node.outputs[i].clients) >= 2
        op = []
        pre_op = [tensor.DimShuffle, tensor.Elemwise, tensor.DimShuffle]
        for c in node.outputs[i].clients:
            if isinstance(c[0].op, tensor.Sum):
                c_ = c[0]
                for i in range(3):
                    if hasattr(c_.outputs[0], 'clients'):
                        op.append(getattr(c_.outputs[0].clients[0][0], 'op', None))
                        c_ = c_.outputs[0].clients[0][0]
                    else:
                        op.append(None)

                if all([isinstance(op[i], pre_op[i]) for i in range(3)]):
                    return c_.outputs[0]

        return None

    def _check_attributes_(self, node1, node2):
        attr = ['imshp', 'kshp', 'border_mode', 'subsample', 'filter_dilation']

        v1 = [getattr(node1.op, a, None) for a in attr]
        v2 = [getattr(node2.op, a, None) for a in attr]
        if v1 == v2:
            return True
        else:
            return False

    def apply(self, fgraph):
        if not mkl_available():
            return

        did_something = True
        while did_something:
            did_something = False
            topo = fgraph.toposort()
            for node in topo:
                if (node in fgraph.apply_nodes) and isinstance(node.op, AbstractConv2d):
                    inp = node.inputs
                    out = node.outputs
                    imshp = getattr(node.op, 'imshp', None)
                    kshp = getattr(node.op, 'kshp', None)
                    border_mode = getattr(node.op, 'border_mode', 'valid')
                    subsample = getattr(node.op, 'subsample', (1, 1))
                    filter_flip = getattr(node.op, 'filter_flip', False)
                    filter_dilation = getattr(node.op, 'filter_dilation', (1, 1))

                    # Get Elemwise node
                    if (len(out) == 1 and (not out[0] in fgraph.outputs) and
                            isinstance(out[0].clients[0][0].op, tensor.Elemwise) and
                            isinstance(out[0].clients[0][0].op.scalar_op, scalar.Add)):
                        if len(out[0].clients[0][0].inputs) == 2:
                            if out[0].clients[0][0].inputs[0] is out[0]:
                                bias = out[0].clients[0][0].inputs[1]
                            else:
                                bias = out[0].clients[0][0].inputs[0]
                            # Get DimShuffle node
                            bias_owner = bias.owner
                            if (bias_owner is None):
                                try:
                                    inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                                    filter_dilation=filter_dilation)(inp[0])
                                    out_0 = mkl_conv.Conv2D(imshp=imshp,
                                                            kshp=kshp,
                                                            border_mode=border_mode,
                                                            subsample=subsample,
                                                            filter_flip=filter_flip,
                                                            filter_dilation=filter_dilation)(image=inp_0, weight=inp[1], bias=bias)
                                    fgraph.repalce_validate(out[0].clients[0][0].outputs[0],
                                                            out_0,
                                                            'ReplaceConvBias')
                                    did_something = True
                                except Exception as e:
                                    raise
                            elif isinstance(bias_owner.op, tensor.DimShuffle) and (bias_owner.inputs[0].owner is None):
                                try:
                                    inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                                    filter_dilation=filter_dilation)(inp[0])
                                    out_0 = mkl_conv.Conv2D(imshp=imshp,
                                                            kshp=kshp,
                                                            border_mode=border_mode,
                                                            subsample=subsample,
                                                            filter_flip=filter_flip,
                                                            filter_dilation=filter_dilation)(image=inp_0, weight=inp[1], bias=bias_owner.inputs[0])
                                    out_1 = MKLToNdarray()(out_0)
                                    fgraph.replace_validate(out[0].clients[0][0].outputs[0],
                                                            out_1,
                                                            'ReplaceConvBias')
                                    did_something = True
                                except Exception as e:
                                    raise
                            else:
                                pass
                elif (node in fgraph.apply_nodes) and isinstance(node.op, AbstractConv2d_gradWeights):
                    inp = node.inputs  # 0-image, 1-gz, 2-shape
                    out = node.outputs
                    imshp = getattr(node.op, 'imshp', None)
                    kshp = getattr(node.op, 'kshp', None)
                    border_mode = getattr(node.op, 'border_mode', 'valid')
                    subsample = getattr(node.op, 'subsample', (1, 1))
                    filter_flip = getattr(node.op, 'filter_flip', False)
                    filter_dilation = getattr(node.op, 'filter_dilation', (1, 1))

                    assert len(inp) == 3 and len(out) == 1
                    for i, c in enumerate(inp[0].clients):
                        if hasattr(c[0], 'op') and isinstance(c[0].op, U2IConv) and self._check_attributes_(c[0], node):
                            for cc in c[0].outputs[0].clients:
                                if isinstance(cc[0].op, mkl_conv.Conv2D) and len(cc[0].inputs) == 3:
                                    weight, bias = cc[0].inputs[1:3]
                                    try:
                                        inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                                        filter_dilation=filter_dilation)(inp[0])
                                        conv_fw = mkl_conv.Conv2D(imshp=imshp,
                                                                  kshp=kshp,
                                                                  border_mode=border_mode,
                                                                  subsample=subsample,
                                                                  filter_flip=filter_flip,
                                                                  filter_dilation=filter_dilation)(inp_0, weight, bias)
                                        gz = I2UGrad()(conv_fw, inp[1])
                                        out_0, out_1 = mkl_conv.ConvGradWeights(imshp=imshp,
                                                                                kshp=kshp,
                                                                                border_mode=border_mode,
                                                                                subsample=subsample,
                                                                                filter_flip=filter_flip,
                                                                                filter_dilation=filter_dilation)(image=inp_0, weight=weight, gradz=gz, bias=bias)
                                        # Get BiasGrad
                                        oriBiasGrad = None  # BiasGrad in original function graph
                                        gz_node = inp[1].owner
                                        for i, o in enumerate(gz_node.outputs):
                                            if inp[1] is o and len(o.clients) >= 2:
                                                oriBiasGrad = self._check_grad_bias_(gz_node, i)

                                        fgraph.replace_validate(out[0], out_0, 'ReplaceConvBias')
                                        if oriBiasGrad:
                                            fgraph.replace_validate(oriBiasGrad, out_1, 'ReplaceConvBias')
                                        did_something = True
                                    except Exception as e:
                                        raise
                elif (node in fgraph.apply_nodes) and isinstance(node.op, AbstractConv2d_gradInputs):
                    inp = node.inputs  # 0-weight, 1-gz, 2-shape
                    out = node.outputs
                    imshp = getattr(node.op, 'imshp', None)
                    kshp = getattr(node.op, 'kshp', None)
                    border_mode = getattr(node.op, 'border_mode', 'valid')
                    subsample = getattr(node.op, 'subsample', (1, 1))
                    filter_flip = getattr(node.op, 'filter_flip', False)
                    filter_dilation = getattr(node.op, 'filter_dilation', (1, 1))

                    assert len(inp) == 3 and len(out) == 1
                    list_Conv2D = [c[0] for c in inp[0].clients if (hasattr(c[0], 'op') and
                                                                    isinstance(c[0].op, mkl_conv.Conv2D) and
                                                                    len(c[0].inputs) == 3 and
                                                                    self._check_attributes_(c[0], node))]
                    if 3 > len(list_Conv2D) > 0:
                        x = list_Conv2D[0].inputs[0].owner.inputs[0]
                        bias = list_Conv2D[0].inputs[2]
                        inp_0 = list_Conv2D[0].inputs[0]
                        try:
                            inp_0 = U2IConv(imshp=imshp, kshp=kshp, border_mode=border_mode, subsample=subsample,
                                            filter_dilation=filter_dilation)(x)
                            conv_fw = mkl_conv.Conv2D(imshp=imshp,
                                                      kshp=kshp,
                                                      border_mode=border_mode,
                                                      subsample=subsample,
                                                      filter_flip=filter_flip,
                                                      filter_dilation=filter_dilation)(inp_0, inp[0], bias)
                            gz = I2UGrad()(conv_fw, inp[1])
                            out_0 = mkl_conv.ConvGradInputs(imshp=imshp,
                                                            kshp=kshp,
                                                            border_mode=border_mode,
                                                            subsample=subsample,
                                                            filter_flip=filter_flip,
                                                            filter_dilation=filter_dilation)(inp_0, inp[0], gz)
                            inp_grad = MKLToNdarray()(out_0)
                            fgraph.replace_validate(out[0], inp_grad, 'ReplaceConvBias')
                            did_something = True
                        except Exception as e:
                            raise e
                else:
                    pass


# Register the instance of global OPT ReplaceConvBias into mkl_seqopt.
mkl_seqopt.register('MKL_CONV_REPLACE', ReplaceConvBias(), 0.095, 'fast_run', 'fast_compile', 'mkl')


@local_optimizer([AbstractConv2d])
def local_Conv2D_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d):
        return

    if node.op.filter_dilation != (1, 1):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if None in node.op.kshp:
        return

    try:
        image, weight = node.inputs
        image_internal = U2IConv(imshp=node.op.imshp,
                                 kshp=node.op.kshp,
                                 subsample=node.op.subsample,
                                 border_mode=node.op.border_mode,
                                 filter_dilation=node.op.filter_dilation)(image)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation)(image_internal, weight)
        z_user = MKLToNdarray()(convOut)
        reval = z_user
        return [reval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@local_optimizer([AbstractConv2d_gradInputs])
def local_ConvGradInputs_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d_gradInputs):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if node.op.filter_dilation != (1, 1):
        return

    if None in node.op.kshp:
        return

    try:
        weight, gz, zshp = node.inputs
        image = node.inputs[2].owner.inputs[0].owner.inputs[0]
        image_internal = U2IConv(imshp=node.op.imshp,
                                 kshp=node.op.kshp,
                                 subsample=node.op.subsample,
                                 border_mode=node.op.border_mode,
                                 filter_dilation=node.op.filter_dilation)(image)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation)(image_internal, weight)
        gz_internal = I2UGrad()(convOut, gz)
        gradImage = mkl_conv.ConvGradInputs(border_mode=node.op.border_mode,
                                            subsample=node.op.subsample,
                                            imshp=node.op.imshp,
                                            kshp=node.op.kshp)(image_internal, weight, gz_internal)
        gradImage_user = MKLToNdarray()(gradImage)
        rval = gradImage_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@local_optimizer([AbstractConv2d_gradWeights])
def local_ConvGradWeights_mkl(node):

    if not mkl_available():
        return

    if not isinstance(node.op, AbstractConv2d_gradWeights):
        return

    if node.inputs[1].type.ndim != 4 and node.inputs[1].type.ndim != 5:
        return

    if node.op.filter_dilation != (1, 1):
        return

    if None in node.op.kshp:
        return

    try:
        image, gz, zshp = node.inputs
        weight = node.inputs[2].owner.inputs[0].owner.inputs[0]
        image_internal = U2IConv(imshp=node.op.imshp,
                                 kshp=node.op.kshp,
                                 subsample=node.op.subsample,
                                 border_mode=node.op.border_mode,
                                 filter_dilation=node.op.filter_dilation)(image)
        convOut = mkl_conv.Conv2D(imshp=node.op.imshp,
                                  kshp=node.op.kshp,
                                  border_mode=node.op.border_mode,
                                  subsample=node.op.subsample,
                                  filter_flip=node.op.filter_flip,
                                  filter_dilation=node.op.filter_dilation)(image_internal, weight)
        gz_internal = I2UGrad()(convOut, gz)
        gradWeight = mkl_conv.ConvGradWeights(border_mode=node.op.border_mode,
                                              subsample=node.op.subsample,
                                              imshp=node.op.imshp,
                                              kshp=node.op.kshp)(image_internal, weight, gz_internal)
        rval = gradWeight
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


conv_groupopt = theano.gof.optdb.LocalGroupDB()
conv_groupopt.__name__ = "mkl_conv_opts"
register_opt()(conv_groupopt)

# MKL-based convolution, using the same group with theano.tensor.nnet.opt to avoid dumlicating GEMM functions
# It can be disabled by excluding 'conv_mkl'.
conv_groupopt.register('local_Conv2D_mkl', local_Conv2D_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_ConvGradInputs_mkl', local_ConvGradInputs_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')
conv_groupopt.register('local_ConvGradWeights_mkl', local_ConvGradWeights_mkl, 20,
                       'conv_mkl', 'fast_compile', 'fast_run')


@register_opt()
@local_optimizer([pool.Pool])
def local_pool_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, pool.Pool):
        return

    if node.inputs[0].type.ndim != 4:
        return

    mkl_ver = theano.contrib.mkl.mkl_version()

    mkl_pool_modes = ['min', 'max', 'average_exc_pad']
    mkl_ignore_border = [False]
    if isinstance(mkl_ver, integer_types) and (mkl_ver >= 20170206):
        mkl_pool_modes.append('average_inc_pad')
        mkl_ignore_border.append(True)

    if node.op.mode not in mkl_pool_modes:
        return

    if node.op.ignore_border not in mkl_ignore_border:
        return

    x, ws, stride, pad = node.inputs
    if stride is None:
        stride = ws

    try:
        x_internal = U2IPool(ignore_border=node.op.ignore_border,
                             mode=node.op.mode)(x, ws, stride, pad)

        poolOut = mkl_pool.Pool(ignore_border=node.op.ignore_border,
                                mode=node.op.mode)(x_internal, ws, stride, pad)

        z_user = MKLToNdarray()(poolOut)

        rval = z_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([pool.MaxPoolGrad, pool.AveragePoolGrad])
def local_poolGrad_mkl(node):
    if not mkl_available():
        return

    if node.inputs[0].type.ndim != 4:
        return

    mkl_ver = theano.contrib.mkl.mkl_version()

    mkl_pool_modes = ['min', 'max', 'average_exc_pad']
    mkl_ignore_border = [False]
    if isinstance(mkl_ver, integer_types) and (mkl_ver >= 20170206):
        mkl_pool_modes.append('average_inc_pad')
        mkl_ignore_border.append(True)

    if node.op.mode not in mkl_pool_modes:
        return

    if node.op.ignore_border not in mkl_ignore_border:
        return

    if isinstance(node.op, pool.MaxPoolGrad):
        x, maxout, gz, ws, stride, pad = node.inputs
    elif isinstance(node.op, pool.AveragePoolGrad):
        x, gz, ws, stride, pad = node.inputs
    else:
        # Other pool mode is not supported
        return

    if stride is None:
        stride = ws

    try:
        x_internal = U2IPool(ignore_border=node.op.ignore_border,
                             mode=node.op.mode)(x, ws, stride, pad)

        poolOut = mkl_pool.Pool(ignore_border=node.op.ignore_border,
                                mode=node.op.mode)(x_internal, ws, stride, pad)

        gz_internal = I2UGrad()(poolOut, gz)

        poolGradOut = mkl_pool.PoolGrad(ignore_border=node.op.ignore_border,
                                        mode=node.op.mode)(x_internal, gz_internal, ws, stride, pad)

        gx_user = MKLToNdarray()(poolGradOut)

        rval = gx_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_relu.AbstractRelu])
def local_relu_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, mkl_relu.AbstractRelu):
        return

    if node.inputs[0].type.ndim != 4:
        return

    x, = node.inputs

    try:
        x_internal = U2IRelu(slope=node.op.slope)(x)
        reluOut = mkl_relu.Relu(slope=node.op.slope)(x_internal)
        z_user = MKLToNdarray()(reluOut)

        rval = z_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_relu.AbstractReluGrad])
def local_reluGrad_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, mkl_relu.AbstractReluGrad):
        return

    if node.inputs[0].type.ndim != 4:
        return

    x, gz = node.inputs

    try:
        x_internal = U2IRelu(slope=node.op.slope)(x)
        reluOut = mkl_relu.Relu(slope=node.op.slope)(x_internal)
        gz_internal = I2UGrad()(reluOut, gz)

        reluGradOut = mkl_relu.ReluGrad(slope=node.op.slope)(x_internal, gz_internal)

        gx_user = MKLToNdarray()(reluGradOut)

        rval = gx_user
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_lrn.AbstractLRN])
def local_lrn_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, mkl_lrn.AbstractLRN):
        return

    if node.inputs[0].type.ndim != 4:
        return

    try:
        x, = node.inputs
        x_u2i = U2ILRN(alpha=node.op.alpha,
                       beta=node.op.beta,
                       k=node.op.k,
                       n=node.op.n)(x)
        lrnout = mkl_lrn.LRN(alpha=node.op.alpha,
                             beta=node.op.beta,
                             k=node.op.k,
                             n=node.op.n)(x_u2i)
        z_i2u = MKLToNdarray()(lrnout)
        rval = z_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_lrn.AbstractLRNGrad])
def local_lrnGrad_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, mkl_lrn.AbstractLRNGrad):
        return

    if node.inputs[0].type.ndim != 4:
        return

    try:
        x, gz, = node.inputs
        x_u2i = U2ILRN(alpha=node.op.alpha,
                       beta=node.op.beta,
                       k=node.op.k,
                       n=node.op.n)(x)
        lrnOut = mkl_lrn.LRN(alpha=node.op.alpha,
                             beta=node.op.beta,
                             k=node.op.k,
                             n=node.op.n)(x_u2i)
        gz_u2i = I2UGrad()(lrnOut, gz)
        lrnGradOut = mkl_lrn.LRNGrad(alpha=node.op.alpha,
                                     beta=node.op.beta,
                                     k=node.op.k,
                                     n=node.op.n)(x_u2i, gz_u2i)
        gx_i2u = MKLToNdarray()(lrnGradOut)
        rval = gx_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_bn.AbstractBatchNormalization])
def local_bn_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, mkl_bn.AbstractBatchNormalization):
        return

    if node.inputs[0].type.ndim != 4:
        return

    try:
        x, scale, shift, = node.inputs[0:3]
        x_u2i = U2IBatchNormalization(eps=node.op.eps)(x)
        bn_out = mkl_bn.BatchNormalization(eps=node.op.eps,
                                           bias=node.op.bias,
                                           term=node.op.term)(x_u2i, scale, shift)
        z_i2u = MKLToNdarray()(bn_out)
        rval = z_i2u
        return [rval]
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return


@register_opt()
@local_optimizer([mkl_bn.AbstractBatchNormalizationGrad])
def local_bnGrad_mkl(node):
    if not mkl_available():
        return

    if not isinstance(node.op, mkl_bn.AbstractBatchNormalizationGrad):
        return

    if node.inputs[0].type.ndim != 4:
        return

    try:
        x, gz, scale, shift, = node.inputs
        x_u2i = U2IBatchNormalization(eps=node.op.eps)(x)
        bn_out = mkl_bn.BatchNormalization(eps=node.op.eps,
                                           bias=node.op.bias,
                                           term=node.op.term)(x_u2i, scale, shift)
        gz_u2i = I2UGrad()(bn_out, gz)
        bn_GradOut = mkl_bn.BatchNormalizationGrad(eps=node.op.eps,
                                                   bias=node.op.bias,
                                                   term=node.op.term)(x_u2i, gz_u2i, scale, shift)
        gx_i2u = MKLToNdarray()(bn_GradOut[0])
        rval = [gx_i2u, bn_GradOut[1], bn_GradOut[2]]
        return rval
    except Exception as e:
        msg = ('Failed to apply local opt to Op %s. '
               'Exception message: %s\n') % (node.op, str(e))
        _logger.warning(msg)
        return



