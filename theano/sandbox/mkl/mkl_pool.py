from __future__ import absolute_import, print_function, division

import numpy
import warnings

from six import integer_types
from six.moves import xrange

import theano
from theano import tensor, Apply, Variable
from theano.gradient import DisconnectedType
from theano.sandbox.mkl.basic_ops import MKLOp
from theano.sandbox.mkl.mkl_type import MKLNdarrayType


class PoolBase(MKLOp):
    def __init__(self, ignore_border=False, mode='max', ndim=2):
        self.mkl_ver = theano.sandbox.mkl.mkl_version()

        mkl_pool_modes = ['min', 'max', 'average_exc_pad']
        mkl_ignore_border = [False]
        if isinstance(self.mkl_ver, integer_types) and (self.mkl_ver >= 20170206):
            mkl_pool_modes.append('average_inc_pad')
            mkl_ignore_border.append(True)

        if mode not in mkl_pool_modes:
            if 'average_inc_pad' == mode:
                raise ValueError("'average_inc_pad' is supported by MKL newer than 20170206, "
                                 "Current MKL version: %s" % self.mkl_ver)
            else:
                raise ValueError(
                    "Pool mode parameter only support \'%s\' by MKL. Got %s" %
                    (', '.join(mkl_pool_modes), mode))

        if ignore_border not in mkl_ignore_border:
            if ignore_border:
                raise ValueError("'ignore_border=True' is supported by MKL newer than 20170206, "
                                 "Current MKL version: %s" % self.mkl_ver)
            else:
                raise ValueError(
                    "Pool ignore_border only support \'%s\' by MKL. Got %s" %
                    (', '.join(map(str, mkl_ignore_border)), ignore_border))

        self.ndim = ndim
        self.ignore_border = ignore_border
        self.mode = mode

    @staticmethod
    def out_shape(imgshape, ws=None, ignore_border=False, stride=None, pad=None,
                  ndim=2, ds=None, st=None, padding=None):
        """
        Return the shape of the output from this op, for input of given
        shape and flags.

        Parameters
        ----------
        imgshape : tuple, list, or similar of integer or scalar Theano variable
            The shape of a tensor of images. The last N elements are
            interpreted as the number of rows, and the number of cols.
        ws : list or tuple of N ints
            Downsample factor over rows and column.
            ws indicates the pool region size.
        ignore_border : bool
            If ws doesn't divide imgshape, do we include an extra row/col/slice
            of partial downsampling (False) or ignore it (True).
        stride : list or tuple of N ints or None
            Stride size, which is the number of shifts over rows/cols/slices to get the
            next pool region. If stride is None, it is considered equal to ws
            (no overlap on pooling regions).
        pad : tuple of N ints or None
            For each downsampling dimension, this specifies the number of zeros to
            add as padding on both sides. For 2D and (pad_h, pad_w), pad_h specifies the
            size of the top and bottom margins, pad_w specifies the size of the left and
            right margins. No padding is added if pad is None.
        ndim : int
            The number of pooling dimensions N.
            The default is 2.
        ds
            *deprecated*, use parameter ws instead.
        st
            *deprecated*, use parameter st instead.
        padding
            *deprecated*, use parameter pad instead.

        Returns
        -------
        list
            The shape of the output from this op, for input of given shape.
            This will have the same length as imgshape, but with last N
            elements reduced as per the downsampling & ignore_border flags.

        """
        # check for deprecated parameter names
        if ds is not None:
            if ws is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'ws' and 'ds'."
                    " Please provide a value only to 'ws'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'ds' parameter is not going to exist"
                    " anymore as it is going to be replaced by the parameter"
                    " 'ws'.",
                    stacklevel=2
                )
                ws = ds
        elif ds is None and ws is None:
            raise ValueError(
                "You must provide a tuple value for the window size."
            )

        if st is not None:
            if stride is not None:
                raise ValueError(
                    "You can't provide a tuple value to both 'st and 'stride'."
                    " Please provide a value only to 'stride'."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'st' parameter is not going to exist"
                    " anymore as it is going to be replaced by the parameter"
                    " 'stride'.",
                    stacklevel=2
                )
                stride = st

        if padding is not None:
            zero_pad = (0,) * ndim
            if pad not in {None, zero_pad}:
                raise ValueError(
                    "You can't provide a tuple value to both 'padding' and pad."
                    "  Please provide a value only to pad."
                )
            else:
                warnings.warn(
                    "DEPRECATION: the 'padding' parameter is not going to"
                    " exist anymore as it is going to be replaced by the"
                    " parameter 'pad'.",
                    stacklevel=2
                )
                pad = padding

        if ndim is None:
            ndim = 2
        assert ndim > 0
        if len(imgshape) < ndim:
            raise TypeError('imgshape must have at least {} dimensions'.format(ndim))

        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * ndim
        patch_shape = tuple(tensor.extract_constant(imgshape[-ndim + i]) + pad[i] * 2
                            for i in xrange(ndim))

        def compute_out(v, downsample, stride):
            if ignore_border:
                if downsample == stride:
                    return v // stride
                else:
                    out = (v - downsample) // stride + 1
                    if isinstance(out, theano.Variable):
                        return tensor.maximum(out, 0)
                    else:
                        return numpy.maximum(out, 0)
            else:
                if isinstance(v, theano.Variable):
                    return tensor.switch(tensor.ge(stride, downsample),
                                         (v - 1) // stride + 1,
                                         tensor.maximum(0, (v - 1 - downsample) //
                                                        stride + 1) + 1)
                elif stride >= downsample:
                    return (v - 1) // stride + 1
                else:
                    return max(0, (v - 1 - downsample + stride) // stride) + 1

        out_shape = [compute_out(patch_shape[i], ws[i], stride[i]) for i in xrange(ndim)]

        rval = list(imgshape[:-ndim]) + out_shape
        return rval

    def c_headers(self):
        return super(PoolBase, self).c_headers()

    def c_support_code_struct(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        sub = {}
        if dtype == 'float32':
            sub['dtype'] = 'float'
            sub['precision'] = 'F32'
        else:
            sub['dtype'] = 'double'
            sub['precision'] = 'F64'
        sub['name'] = name

        ccode = """
        int first_run;
        size_t inputSize[DIMENSION];
        size_t inputStrides[DIMENSION];
        size_t outputSize[DIMENSION];
        size_t outputStrides[DIMENSION];
        size_t kernelSize[2];
        size_t kernelStride[2];
        int inputOffset[4];

        void *x_internal_buffer;
        void *z_internal_buffer;
        void *gz_internal_buffer;

        dnnError_t err;
        dnnPrimitive_t pool_fwd;
        dnnPrimitive_t pool_bwd;
        void *pool_res[dnnResourceNumber];

        size_t input_bytes;
        size_t output_bytes;

        dnnLayout_t x_internal_layout;
        dnnLayout_t z_internal_layout;
        dnnLayout_t gz_internal_layout;
        dnnPrimitive_t convert_gz_to_internal;
        dnnPrimitive_t convert_x_to_internal;
        """ % sub
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        first_run = 1;
        """
        return ccode

    '''
    def c_cleanup_code_struct(self, node, name):
        if node.inputs[0].type.dtype == "float32":
            precision = "F32"
        elif node.inputs[0].type.dtype == "float64":
            precision = "F64"

        ccode = """
        dnnDelete_%(precision)s(convert_gz_to_internal);
        dnnLayoutDelete_%(precision)s(x_internal_layout);
        dnnLayoutDelete_%(precision)s(z_internal_layout);
        """ % locals()
        return ccode
    '''

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]


class Pool(PoolBase):
    """
    For N-dimensional tensors, consider that the last two dimensions span
    images. This Op downsamples these images by taking the max, sum or average
    over different patch.

    The constructor takes the max, sum or average or different input patches.

    Parameters
    ----------
    ds : list or tuple of two ints
        Downsample factor over rows and column.
        ds indicates the pool region size.
    ignore_border : bool
        If ds doesn't divide imgshape, do we include an extra row/col
        of partial downsampling (False) or ignore it (True).
    st : list or tuple of two ints or None
        Stride size, which is the number of shifts over rows/cols to get the
        next pool region. If st is None, it is considered equal to ds
        (no overlap on pooling regions).
    padding: tuple of two ints
        (pad_h, pad_w), pad zeros to extend beyond four borders of the images,
        pad_h is the size of the top and bottom margins, and pad_w is the size
        of the left and right margins.
    mode : {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}
        ('average_exc_pad' excludes the padding from the count)

    """
    __props__ = ('ignore_border', 'mode')

    '''
    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) == 1:
            # Old interface
            self.ndim = len(node.op.ds)
            self.mode = node.op.mode
            ws = theano.tensor.constant(node.op.ds)
            st = theano.tensor.constant(node.op.st)
            pad = theano.tensor.constant(node.op.padding)
            node.inputs.append(ws)
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]
    '''

    def make_node(self, x, ws, stride=None, pad=None):
        if not isinstance(x.type, MKLNdarrayType):
            raise TypeError('Expected MKLNdarrayType for x, '
                            'but got type %s.' % str(x.type))

        if x.type.ndim != 4:
            raise TypeError('Expected a 4 dims varialbe for x, '
                            'but got %d dims.' % x.type.ndim)

        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        elif isinstance(pad, (tuple, list)):
            if isinstance(ws, (tuple, list)):
                if any(pad[i] >= ws[i] for i in range(nd)):
                    raise NotImplementedError(
                        'padding must be smaller than strides')
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert ws.ndim == 1
        assert stride.ndim == 1
        assert pad.ndim == 1
        if x.type.ndim < nd:
            raise TypeError()
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')
        # If the input shape are broadcastable we can have 0 in the output shape
        broad = x.broadcastable[:-nd] + (False,) * nd
        out = MKLNdarrayType(x.dtype, broad)

        return Apply(self, [x, ws, stride, pad], [out()])

    def infer_shape(self, node, in_shapes):
        ws, stride, pad = [node.inputs[1], node.inputs[2], node.inputs[3]]
        shp = self.out_shape(in_shapes[0], ws, self.ignore_border, stride,
                             pad, self.ndim)
        return [shp]

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]

        return [PoolGrad(ignore_border=self.ignore_border,
                         mode=self.mode)(x, gz, ws, stride, pad)] + disc

    def c_code(self, node, name, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out

        if 'max' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            sub['algo'] = 'dnnAlgorithmPoolingMin'
        elif 'average_exc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgExcludePadding"
        elif 'average_inc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgIncludePadding"
        else:
            raise ValueError("mode must be one of 'max', 'min', "
                             "'average_exc_pad', and 'average_inc_pad'")

        if self.ignore_border:
            sub['borderType'] = 'dnnBorderZerosAsymm'
            sub['ignore_border'] = 1
        else:
            sub['borderType'] = 'dnnBorderZeros'
            sub['ignore_border'] = 0

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #ifdef _MKL_DEBUG_
            std::cout<<"pool start"<<std::endl;
        #endif

        int ret = 0;
        int ndim = MKLNdarray_NDIM(%(x)s);
        size_t user_dims[DIMENSION] = {0};
        int typenum = MKLNdarray_TYPE((MKLNdarray*)%(x)s);

        if (1 == first_run) {
            size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
            size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
            size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
            size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
            size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
            size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

            kernelSize[0] = kernel_w;
            kernelSize[1] = kernel_h;
            kernelStride[0] = stride_w;
            kernelStride[1] = stride_h;
            if (%(ignore_border)s) {
                inputOffset[0] = -pad_w;
                inputOffset[1] = -pad_h;
                inputOffset[2] = -pad_w;
                inputOffset[3] = -pad_h;
            } else {
                inputOffset[0] = -pad_w;
                inputOffset[1] = -pad_h;
            }

            int out_h, out_w; // shape of the output
            int in_h, in_w; // shape of the padded_input
            in_h = MKLNdarray_DIMS(%(x)s)[2];
            in_w = MKLNdarray_DIMS(%(x)s)[3];

            if (%(ignore_border)s) {
                out_h = floor((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
                out_w = floor((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            } else {
                out_h = ceil((float)(in_h + 2 * pad_h - kernel_h)/stride_h) + 1;
                out_w = ceil((float)(in_w + 2 * pad_w - kernel_w)/stride_w) + 1;
            }
            if (pad_h || pad_w) {
                if ((out_h - 1) * stride_h >= (in_h + pad_h)) {
                    --out_h;
                }
                if ((out_w - 1) * stride_w >= (in_w + pad_w)) {
                    --out_w;
                }
                assert((out_h - 1) * stride_h < in_h + pad_h);
                assert((out_w - 1) * stride_w < in_w + pad_w);
            }

            inputSize[0] = MKLNdarray_DIMS(%(x)s)[3];  //w
            inputSize[1] = MKLNdarray_DIMS(%(x)s)[2];  //h
            inputSize[2] = MKLNdarray_DIMS(%(x)s)[1];  //c
            inputSize[3] = MKLNdarray_DIMS(%(x)s)[0];  //n
            inputStrides[0] = 1;
            inputStrides[1] = inputSize[0];
            inputStrides[2] = inputSize[0] * inputSize[1];
            inputStrides[3] = inputSize[0] * inputSize[1] * inputSize[2];

            outputSize[0] = out_w;
            outputSize[1] = out_h;
            outputSize[2] = inputSize[2];
            outputSize[3] = inputSize[3];
            outputStrides[0] = 1;
            outputStrides[1] = outputSize[0];
            outputStrides[2] = outputSize[0] * outputSize[1];
            outputStrides[3] = outputSize[0] * outputSize[1] * outputSize[2];

            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pool_fwd, NULL,
                       %(algo)s, MKLNdarray_LAYOUT(%(x)s), kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );

            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &x_internal_layout, pool_fwd, dnnResourceSrc), err );

            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &z_internal_layout, pool_fwd, dnnResourceDst), err );

            if (NULL == MKLNdarray_WORKSPACE(%(x)s)) {
                ret = MKLNdarray_create_buffer_from_primitive(%(x)s, &pool_fwd, dnnResourceWorkspace);
                if (0 != ret) {
                    std::cout<< "MKLNdarray_create_buffer_from_primitive failed, return: "<< ret <<", line: "<<__LINE__<<std::endl;
                    %(fail)s
                }
            }
        }

        #ifdef _MKL_DEBUG_
            std::cout << "inputSize     : " << inputSize[3] << " x " << inputSize[2] << " x " << inputSize[1] << " x " << inputSize[0] << std::endl;
            std::cout << "outputSize    : " << outputSize[3] << " x " << outputSize[2] << " x " << outputSize[1] << " x " << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[1] << " x " << kernelSize[0] << std::endl;
            std::cout << "pooling stride: " << kernelStride[1] << " x " << kernelStride[0] << std::endl;
            std::cout << "padding       : " << inputOffset[3] << " x " << inputOffset[2] << " x " << inputOffset[1] << " x " << inputOffset[0] << std::endl;
            std::cout << "ignore_border : " << %(ignore_border)s << std::endl;
        #endif

        user_dims[0] = outputSize[3];
        user_dims[1] = outputSize[2];
        user_dims[2] = outputSize[1];
        user_dims[3] = outputSize[0];
        if ( !(%(z)s
               && MKLNdarray_Check((PyObject *)%(z)s)
               && MKLNdarray_NDIM(%(z)s) == ndim
               && MKLNdarray_DIMS(%(z)s)[0] == outputSize[0]
               && MKLNdarray_DIMS(%(z)s)[1] == outputSize[1]
               && MKLNdarray_DIMS(%(z)s)[2] == outputSize[2]
               && MKLNdarray_DIMS(%(z)s)[3] == outputSize[3] )) {
            if (%(z)s) Py_XDECREF(%(z)s);

            %(z)s = (MKLNdarray *)MKLNdarray_New(ndim, typenum);
            if (NULL == %(z)s) {
                %(fail)s
            }

            ret = MKLNdarray_set_structure(%(z)s, ndim, user_dims);
            if (ret != 0) {
                %(fail)s;
            }

            ret = MKLNdarray_create_buffer_from_primitive(%(z)s, &pool_fwd, dnnResourceDst);
            if (ret != 0) {
                %(fail)s;
            }
        }

        if (1 == first_run) {
            if (! dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(x)s), x_internal_layout)) {
                if (NULL == convert_x_to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_x_to_internal, MKLNdarray_LAYOUT(%(x)s), x_internal_layout), err );
                }
            }
        }
        if (convert_x_to_internal) {
            if (NULL == x_internal_buffer) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&x_internal_buffer, x_internal_layout), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_x_to_internal, MKLNdarray_DATA(%(x)s), x_internal_buffer), err );
        } else {
            x_internal_buffer = MKLNdarray_DATA(%(x)s);
        }

        pool_res[dnnResourceSrc] = x_internal_buffer;
        pool_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
        pool_res[dnnResourceDst] = MKLNdarray_DATA(%(z)s);

        #ifdef _MKL_DEBUG_
            input_bytes = dnnLayoutGetMemorySize_%(precision)s(MKLNdarray_LAYOUT(%(x)s));
            output_bytes = dnnLayoutGetMemorySize_%(precision)s(MKLNdarray_LAYOUT(%(z)s));
            std::cout << " input_bytes = " << input_bytes << std::endl;
            std::cout << " output_bytes = " << output_bytes << std::endl;
            std::cout << "pool_res[dnnResourceSrc] = @" << pool_res[dnnResourceSrc] << std::endl;
            std::cout << "pool_res[dnnResourceDst] = @" << pool_res[dnnResourceDst] << std::endl;
            std::cout << "pool_res[dnnResourceWorkspace] = @" << pool_res[dnnResourceWorkspace] << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pool_fwd, (void**)pool_res), err );

        first_run = 0;
        #ifdef _MKL_DEBUG_
            std::cout<<"pool forward, z_internal_buffer: @"<<z_internal_buffer<<", output layout: @"<<z_internal_layout<<std::endl;
            std::cout<<"z.shape: "<<MKLNdarray_DIMS(%(z)s)[3]<<" x "<<MKLNdarray_DIMS(%(z)s)[2]<<" x "<<MKLNdarray_DIMS(%(z)s)[1]<<" x "<<MKLNdarray_DIMS(%(z)s)[0]<<std::endl;
            std::cout<<"pool end\\n"<<std::endl;
        #endif
        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (1, 0)


class PoolGrad(PoolBase):
    __props__ = ('ignore_border', 'mode')

    '''
    def prepare_node(self, node, storage_map, compute_map, impl):
        if len(node.inputs) < 5:  # 5 for AveragePoolGrad, 6 for MaxPoolGrad
            # Old interface
            self.ndim = len(node.op.ds)
            self.mode = node.op.mode
            ws = theano.tensor.constant(node.op.ds)
            st = theano.tensor.constant(node.op.st)
            pad = theano.tensor.constant(node.op.padding)
            node.inputs.append(ws)
            node.inputs.append(st)
            node.inputs.append(pad)
            if isinstance(ws, theano.Constant):
                storage_map[ws] = [ws.data]
                compute_map[ws] = [True]
            else:
                storage_map[ws] = [None]
                compute_map[ws] = [False]
            if isinstance(st, theano.Constant):
                storage_map[st] = [st.data]
                compute_map[st] = [True]
            else:
                storage_map[st] = [None]
                compute_map[st] = [False]
            if isinstance(pad, theano.Constant):
                storage_map[pad] = [pad.data]
                compute_map[pad] = [True]
            else:
                storage_map[pad] = [None]
                compute_map[pad] = [False]
    '''

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, x, gz, ws, stride=None, pad=None):
        if not isinstance(x.type, MKLNdarrayType):
            raise TypeError('Expected MKLNdarrayType for x, '
                            'but got type %s.' % str(x.type))

        if not isinstance(gz.type, MKLNdarrayType):
            raise TypeError('Expected MKLNdarrayType for gz, '
                            'but got type %s.' % str(gz.type))

        if x.type.ndim != 4:
            raise TypeError('Expected a 4 dims varialbe for x, '
                            'but got %d dims.' % x.type.ndim)

        if gz.type.ndim != 4:
            raise TypeError('Expected a 4 dims varialbe for gz, '
                            'but got %d dims.' % gz.type.ndim)

        nd = self.ndim
        if stride is None:
            stride = ws
        if pad is None:
            pad = (0,) * nd
        ws = tensor.as_tensor_variable(ws)
        stride = tensor.as_tensor_variable(stride)
        pad = tensor.as_tensor_variable(pad)
        assert isinstance(ws, Variable) and ws.ndim == 1
        assert isinstance(stride, Variable) and stride.ndim == 1
        assert isinstance(pad, Variable) and pad.ndim == 1
        if ws.dtype not in tensor.int_dtypes:
            raise TypeError('Pool downsample parameters must be ints.')
        if stride.dtype not in tensor.int_dtypes:
            raise TypeError('Stride parameters must be ints.')
        if pad.dtype not in tensor.int_dtypes:
            raise TypeError('Padding parameters must be ints.')

        return Apply(self, [x, gz, ws, stride, pad], [x.type()])

    def c_code(self, node, name, inp, out, sub):
        x, gz, ws, stride, pad = inp
        gx, = out

        if 'max' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            sub['algo'] = 'dnnAlgorithmPoolingMin'
        elif 'average_exc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgExcludePadding"
        elif 'average_inc_pad' == self.mode:
            sub['algo'] = "dnnAlgorithmPoolingAvgIncludePadding"
        else:
            raise ValueError("mode must be one of 'max', 'min', "
                             "'average_exc_pad', and 'average_inc_pad'")

        if self.ignore_border:
            sub['borderType'] = 'dnnBorderZerosAsymm'
            sub['ignore_border'] = 1
        else:
            sub['borderType'] = 'dnnBorderZeros'
            sub['ignore_border'] = 0

        if node.inputs[0].type.dtype == "float32":
            sub['precision'] = 'F32'
            sub['dtype'] = 'float'
        elif node.inputs[0].type.dtype == "float64":
            sub['precision'] = 'F64'
            sub['dtype'] = 'double'

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        #ifdef _MKL_DEBUG_
            std::cout<<"poolgrad start"<<std::endl;
        #endif
        int ret = 0;
        int ndim = MKLNdarray_NDIM(%(x)s);
        int typenum = MKLNdarray_TYPE((MKLNdarray*)%(x)s);

        if (1 == first_run) {
            size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
            size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
            size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
            size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
            size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
            size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

            kernelSize[0] = kernel_w;
            kernelSize[1] = kernel_h;
            kernelStride[0] = stride_w;
            kernelStride[1] = stride_h;
            if (%(ignore_border)s) {
                inputOffset[0] = -pad_w;
                inputOffset[1] = -pad_h;
                inputOffset[2] = -pad_w;
                inputOffset[3] = -pad_h;
            } else {
                inputOffset[0] = -pad_w;
                inputOffset[1] = -pad_h;
            }

            inputSize[0] = MKLNdarray_DIMS(%(x)s)[3];  //w
            inputSize[1] = MKLNdarray_DIMS(%(x)s)[2];  //h
            inputSize[2] = MKLNdarray_DIMS(%(x)s)[1];  //c
            inputSize[3] = MKLNdarray_DIMS(%(x)s)[0];  //n
            inputStrides[0] = 1;
            inputStrides[1] = inputSize[0];
            inputStrides[2] = inputSize[0] * inputSize[1];
            inputStrides[3] = inputSize[0] * inputSize[1] * inputSize[2];

            outputSize[0] = MKLNdarray_DIMS(%(gz)s)[3];  //w
            outputSize[1] = MKLNdarray_DIMS(%(gz)s)[2];  //h
            outputSize[2] = MKLNdarray_DIMS(%(gz)s)[1];  //c
            outputSize[3] = MKLNdarray_DIMS(%(gz)s)[0];  //n
            outputStrides[0] = 1;
            outputStrides[1] = outputSize[0];
            outputStrides[2] = outputSize[0] * outputSize[1];
            outputStrides[3] = outputSize[0] * outputSize[1] * outputSize[2];

            CHECK_ERR( dnnPoolingCreateBackward_%(precision)s(&pool_bwd, NULL,
                       %(algo)s, MKLNdarray_LAYOUT(%(x)s), kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );

            CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&pool_fwd, NULL,
                       %(algo)s, MKLNdarray_LAYOUT(%(x)s), kernelSize,
                       kernelStride, inputOffset, %(borderType)s), err );

            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &x_internal_layout, pool_fwd, dnnResourceSrc), err );

            CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&x_internal_buffer, x_internal_layout) , err );

            CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(
                       &gz_internal_layout, pool_fwd, dnnResourceDst), err );
        }

        #ifdef _MKL_DEBUG_
            std::cout << "inputSize     : " << inputSize[3] << " x " << inputSize[2] << " x " << inputSize[1] << " x " << inputSize[0] << std::endl;
            std::cout << "outputSize    : " << outputSize[3] << " x " << outputSize[2] << " x " << outputSize[1] << " x " << outputSize[0] << std::endl;
            std::cout << "pooling region: " << kernelSize[1] << " x " << kernelSize[0] << std::endl;
            std::cout << "pooling stride: " << kernelStride[1] << " x " << kernelStride[0] << std::endl;
            std::cout << "padding       : " << inputOffset[3] << " x " << inputOffset[2] << " x " << inputOffset[1] << " x " << inputOffset[0] << std::endl;
            std::cout << "ignore_border : " << %(ignore_border)s << std::endl;
        #endif

        if ( !(%(gx)s
               && MKLNdarray_Check((PyObject *)%(gx)s)
               && MKLNdarray_NDIM(%(gx)s) == ndim
               && MKLNdarray_DIMS(%(gx)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
               && MKLNdarray_DIMS(%(gx)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
               && MKLNdarray_DIMS(%(gx)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
               && MKLNdarray_DIMS(%(gx)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {
            if (%(gx)s) Py_XDECREF(%(gx)s);

            %(gx)s = (MKLNdarray *)MKLNdarray_New(ndim, typenum);
            if (NULL == %(gx)s) {
                %(fail)s
            }

            ret = MKLNdarray_set_structure(%(gx)s, ndim, MKLNdarray_DIMS(%(x)s));
            if (ret != 0) {
                %(fail)s;
            }

            ret = MKLNdarray_create_buffer_from_primitive(%(gx)s, &pool_bwd, dnnResourceDiffSrc);
            if (ret != 0) {
                %(fail)s;
            }
        }

        #pragma omp parallel for
        #pragma ivdep
        for(int i = 0 ; i < (MKLNdarray_DIMS(%(gx)s)[0] * MKLNdarray_STRIDES(%(gx)s)[0]); ++i) {
             ((unsigned int *)MKLNdarray_DATA(%(gx)s))[i] = 0;
        }

        if (1 == first_run) {
            if (!dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(gz)s), gz_internal_layout)) {
            #ifdef _MKL_DEBUG_
                std::cout<<"pool backward, gz layout is not equal" <<std::endl;
            #endif
                if (NULL == convert_gz_to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_gz_to_internal, MKLNdarray_LAYOUT(%(gz)s),  gz_internal_layout), err );
                 }
            }
        }

        if (convert_gz_to_internal) {
            if (NULL == gz_internal_buffer) {
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)&gz_internal_buffer, gz_internal_layout), err );
            }
            CHECK_ERR( dnnConversionExecute_%(precision)s(convert_gz_to_internal, MKLNdarray_DATA(%(gz)s), gz_internal_buffer), err );
        } else {
             gz_internal_buffer = MKLNdarray_DATA(%(gz)s);
        }

        pool_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
        pool_res[dnnResourceDiffDst] = gz_internal_buffer;
        pool_res[dnnResourceDiffSrc] = x_internal_buffer;

        #ifdef _MKL_DEBUG_
            input_bytes = dnnLayoutGetMemorySize_%(precision)s(x_internal_layout);
            output_bytes = dnnLayoutGetMemorySize_%(precision)s(gz_internal_layout);
            std::cout << " input_bytes = " << input_bytes << std::endl;
            std::cout << " output_bytes = " << output_bytes << std::endl;
        #endif

        CHECK_ERR( dnnExecute_%(precision)s(pool_bwd, (void**)pool_res), err );

        if (!dnnLayoutCompare_%(precision)s(x_internal_layout, MKLNdarray_LAYOUT(%(x)s))) {
            #ifdef _MKL_DEBUG_
                std::cout<<"pool backward, x layout is not equal" <<std::endl;
            #endif
            if (NULL == convert_x_to_internal) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_x_to_internal, x_internal_layout, MKLNdarray_LAYOUT(%(gx)s)), err );
            }
        }

        CHECK_ERR( dnnConversionCreate_%(precision)s(&convert_x_to_internal, x_internal_layout, MKLNdarray_LAYOUT(%(gx)s)), err );
        CHECK_ERR( dnnConversionExecute_%(precision)s(convert_x_to_internal, x_internal_buffer, MKLNdarray_DATA(%(gx)s)), err );

        first_run = 0;
        """ % sub

        return ccode

    def c_code_cache_version(self):
        return (1, 0)
