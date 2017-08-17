from theano import gof, tensor, Variable
from theano.tensor import as_tensor_variable
from theano.contrib.mkl import basic_ops
from theano.gradient import DisconnectedType
from theano.contrib.mkl import mkl_type


class AbstractBatchNormalization(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term')

    def __init__(self, eps=1e-5, bias=1, term=1):
        self.eps = eps
        self.bias = bias
        self.term = term

    def make_node(self, x, scale, shift, mean, std):
        x = tensor.as_tensor_variable(x)
        assert x.ndim == 4
        scale = tensor.as_tensor_variable(scale)
        shift = tensor.as_tensor_variable(shift)
        mean = tensor.as_tensor_variable(mean)
        std = tensor.as_tensor_variable(std)

        return gof.Apply(self, [x, scale, shift, mean, std], [x.type()])

    def grad(self, inp, grads):
        x, scale, shift, mean, std = inp
        gz, = grads

        disc = [DisconnectedType()() for i in inp[3:]]
        AbstractBN = AbstractBatchNormalizationGrad(eps=self.eps,
                                                    bias=self.bias,
                                                    term=self.term)
        [gx, g_scale, g_shift] = AbstractBN(x, gz, scale, shift)
        return [gx, g_scale, g_shift] + disc

    def perform(self, node, inp, out):
        x, scale, shift, mean, std = inp
        z, = out

    def connection_pattern(self, node):
        return [[1], [1], [1], [0], [0]]


class AbstractBatchNormalizationGrad(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term')

    def __init__(self, eps=1e-5, bias=1, term=1):
        self.eps = eps
        self.bias = bias
        self.term = term

    def make_node(self, x, gz, scale, shift):
        assert isinstance(x, Variable) and x.ndim == 4
        assert isinstance(gz, Variable) and gz.ndim == 4
        scale = as_tensor_variable(scale)
        shift = as_tensor_variable(shift)
        return gof.Apply(self, [x, gz, scale, shift], [x.type(), scale.type(), shift.type()])

    def perform(self, node, inp, out):
        x, gz, scale, shift = inp
        gx, g_scale, g_shift = out


class BatchNormalization(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term')

    def __init__(self, eps=1e-5, bias=1, term=1):
        super(BatchNormalization, self).__init__()
        self.eps = eps
        self.bias = bias
        self.term = term

    def make_node(self, x, scale, shift):
        if not isinstance(x.type, mkl_type.MKLNdarrayType):
            raise TypeError('BatchNormalization: input x should be an instance of MKLNdarrayType')

        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        # x = tensor.as_tensor_variable(x)
        scale = tensor.as_tensor_variable(scale)
        shift = tensor.as_tensor_variable(shift)

        assert x.dtype == scale.dtype
        assert x.dtype == shift.dtype
        return gof.Apply(self, [x, scale, shift], [x.type()])

    def grad(self, inp, grads):
        x, scale, shift, = inp
        gz, = grads
        gx, g_scale, g_shift = BatchNormalizationGrad(eps=self.eps,
                                                      bias=self.bias,
                                                      term=self.term)(x, gz, scale, shift)
        return gx, g_scale, g_shift

    def c_support_code_struct(self, node, name):
        support_code = """
            dnnLayout_t layout_x_internal;
            dnnLayout_t layout_scaleshift;
            void *scaleShift_buffer;
            dnnPrimitive_t bnFwd;
            dnnPrimitive_t prim_to_internal;

            int first_run;
            int typenum;
            int x_bs;
            int x_channels;
            int x_row;
            int x_col;
            size_t sizes[DIMENSION];
            size_t strides[DIMENSION];
            dnnError_t err;
            void* x_internal_buffer;
            void* bn_res[dnnResourceNumber];
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
            layout_x_internal = NULL;
            layout_scaleshift = NULL;

            scaleShift_buffer = NULL;
            bnFwd = NULL;
            prim_to_internal = NULL;
            x_internal_buffer = NULL;

            first_run = 1;
            typenum = 0;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')

        if dtype is 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        return """
        // release layout
        if (NULL != layout_x_internal) {
            dnnLayoutDelete_%(precision)s(layout_x_internal);
            layout_x_internal = NULL;
        }
        if (NULL != layout_scaleshift) {
            dnnLayoutDelete_%(precision)s(layout_scaleshift);
            layout_scaleshift = NULL;
        }
        // release primitive
        if (NULL != bnFwd) {
            dnnDelete_%(precision)s(bnFwd);
            bnFwd = NULL;
        }
        if (NULL != prim_to_internal) {
            dnnDelete_%(precision)s(prim_to_internal);
            prim_to_internal = NULL;
        }
        // release buffer
        if (NULL != x_internal_buffer) {
            dnnReleaseBuffer_%(precision)s(x_internal_buffer);
            x_internal_buffer = NULL;
        }
        if (NULL != scaleShift_buffer) {
            dnnReleaseBuffer_%(precision)s(scaleShift_buffer);
            scaleShift_buffer = NULL;
        }
        """ % locals()

    def c_code(self, node, name, inp, out, sub):
        x, scale, shift, = inp
        z, = out
        eps = self.eps
        bias = self.bias
        term = self.term

        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')

        if dtype is 'float32':
            precision = 'F32'
            t = 'float'
        else:
            precision = 'F64'
            t = 'double'

        fail = sub['fail']

        ret = """
        {
            %(t)s* scale_buffer_ptr = NULL;
            %(t)s* shift_buffer_ptr = NULL;
            int ret = 0;
            if (first_run) {
                typenum    = MKLNdarray_TYPE((MKLNdarray*)%(x)s);
                x_bs       = MKLNdarray_DIMS(%(x)s)[0];
                x_channels = MKLNdarray_DIMS(%(x)s)[1];
                x_row      = MKLNdarray_DIMS(%(x)s)[2];
                x_col      = MKLNdarray_DIMS(%(x)s)[3];
                sizes[0]   = x_col;
                sizes[1]   = x_row;
                sizes[2]   = x_channels;
                sizes[3]   = x_bs;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = strides[1] * sizes[1];
                strides[3] = strides[2] * sizes[2];
            }

            if (%(bias)s) {
                scale_buffer_ptr = (%(t)s*)PyArray_DATA(%(scale)s);
                shift_buffer_ptr = (%(t)s*)PyArray_DATA(%(shift)s);
            }

            if(first_run) {
                // create bn fwd primitive
                CHECK_ERR( dnnBatchNormalizationCreateForward_%(precision)s(&bnFwd, NULL, MKLNdarray_LAYOUT(%(x)s), %(eps)s), err);

                // create internal layout for input x
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_x_internal, bnFwd, dnnResourceSrc), err);

                // layout for scaleshift
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_scaleshift, bnFwd, dnnResourceScaleShift), err);

                // buffer for scaleshift
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)(&scaleShift_buffer), layout_scaleshift), err);

                if (!%(bias)s) {
                    for (int i = 0; i < x_channels; i++) {
                        if (((%(t)s*)scaleShift_buffer)[i] != 1.0){
                            std::cout<<"scale init failed! "<<((%(t)s*)scaleShift_buffer)[i]<<std::endl;
                            exit(1);
                        }
                        if(((%(t)s*)scaleShift_buffer)[x_channels + i] != 0) {
                            std::cout<<"shift init failed!"<<std::endl;
                            exit(1);
                        }
                    }
                }
            }

            // create workspace buffer in x
            if (MKLNdarray_WORKSPACE(%(x)s) == NULL) {
                ret = MKLNdarray_create_buffer_from_primitive(%(x)s, &bnFwd, dnnResourceWorkspace);
                if (0 != ret) {
                    std::cout<<"MKLNdarray_create_buffer_from_primitive return: "<<ret<<", line: "<<__LINE__<<std::endl;
                    exit(1);
                }
            }

            if (! (%(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == MKLNdarray_NDIM(%(x)s)
                && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
                && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
                && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
                && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {

                if (%(z)s) Py_XDECREF(%(z)s);

                %(z)s = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(%(x)s), typenum);
                if (! %(z)s) {
                    %(fail)s;
                }

                int status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(x)s), MKLNdarray_DIMS(%(x)s));
                if (status != 0) {
                    %(fail)s;
                }

                // create dst layout and buffer in z
                ret = MKLNdarray_create_buffer_from_primitive(%(z)s, &bnFwd, dnnResourceDst);
                if (0 != ret) {
                    std::cout<<"MKLNdarray_createt_buffer return: "<<ret<<", line: "<<__LINE__<<std::endl;
                    exit(1);
                }
            }  // else reuse %(z)s

            // compare input layout and internal layout, do internal to internal conversion
            if (! dnnLayoutCompare_%(precision)s(layout_x_internal, MKLNdarray_LAYOUT(%(x)s))) {
                if (NULL == prim_to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&prim_to_internal, MKLNdarray_LAYOUT(%(x)s), layout_x_internal), err);
                }
                if (NULL == x_internal_buffer) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s(&x_internal_buffer, layout_x_internal), err);
                }

                if (prim_to_internal) {
                    CHECK_ERR( dnnConversionExecute_%(precision)s(prim_to_internal, MKLNdarray_DATA(%(x)s), x_internal_buffer), err);
                }
                bn_res[dnnResourceSrc] = x_internal_buffer;
            } else {
                bn_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
            }

            if (%(bias)s) {
                // Read data from bias weight and bias term buffern to ScaleShift buffer
                for (int i = 0; i < x_channels; i++) {
                    ((%(t)s*)scaleShift_buffer)[i] = scale_buffer_ptr[i];
                    ((%(t)s*)scaleShift_buffer)[x_channels + i] = 0;
                    if (%(term)s) {
                        ((%(t)s*)scaleShift_buffer)[x_channels + i] = shift_buffer_ptr[i];
                    }
                }
            }

            bn_res[dnnResourceDst] = MKLNdarray_DATA(%(z)s);
            bn_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
            bn_res[dnnResourceScaleShift] = scaleShift_buffer;

            CHECK_ERR( dnnExecute_%(precision)s(bnFwd, bn_res), err);
            first_run = 0;
        }
        """ % locals()
        return ret

    def c_code_cache_version(self):
        return (0, 1, 1)


class BatchNormalizationGrad(basic_ops.MKLOp):
    __props__ = ('eps', 'bias', 'term')

    def __init__(self, eps=1e-5, bias=1, term=1):
        super(BatchNormalizationGrad, self).__init__()
        self.eps = eps
        self.bias = bias
        self.term = term

    def c_support_code_struct(self, node, name):
        support_code = """
        int first_run;
        int typenum;
        int x_bs;
        int x_channels;
        int x_row;
        int x_col;
        size_t sizes[DIMENSION];
        size_t strides[DIMENSION];
        dnnError_t err;

        // for backward wrt data
        void* bn_res[dnnResourceNumber];
        dnnPrimitive_t bnBwd;
        dnnPrimitive_t prim_to_internal_gz;
        dnnLayout_t layout_gz_internal;
        dnnLayout_t layout_scaleshift;
        void* gz_internal_buffer;
        void* scaleShift_buffer;

        // for backward wrt scaleshift
        void* BatchNormBwdScaleShift_res[dnnResourceNumber];
        dnnPrimitive_t bnBwdScaleShift;
        dnnPrimitive_t prim_to_internal_gz_gs;  // gs means gradient wrt scaleshift
        dnnPrimitive_t prim_to_internal_x_gs;   // gs means gradient wrt scaleshift
        dnnLayout_t layout_gz_internal_gs;      // gs means gradient wrt scaleshift, dnnResourceDiffDst
        dnnLayout_t layout_x_internal_gs;       // gs means gradient wrt scaleshift, dnnResourceSrc
        dnnLayout_t layout_gs_internal_gs;      // gs means gradient wrt scaleshift, dnnResourceDiffScaleShift
        void* gz_internal_buffer_gs;            // gs means gradient wrt scaleshift
        void* x_internal_buffer_gs;             // gs means gradient wrt scaleshift
        void* gs_internal_buffer_gs;
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
            // primitive
            bnBwd = NULL;
            bnBwdScaleShift = NULL;
            prim_to_internal_gz = NULL;
            prim_to_internal_gz_gs = NULL;
            prim_to_internal_x_gs = NULL;

            // layout
            layout_gz_internal = NULL;
            layout_x_internal_gs = NULL;
            layout_gz_internal_gs = NULL;
            layout_gs_internal_gs = NULL;
            layout_scaleshift = NULL;

            // buffer
            gz_internal_buffer = NULL;
            gz_internal_buffer_gs = NULL;
            x_internal_buffer_gs = NULL;
            scaleShift_buffer = NULL;
            gs_internal_buffer_gs = NULL;

            first_run = 1;
            typenum = 0;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')

        if dtype is 'float32':
            precision = 'F32'
        else:
            precision = 'F64'

        return """
        // release layout
        if (NULL != layout_gz_internal) {
            dnnLayoutDelete_%(precision)s(layout_gz_internal);
            layout_gz_internal = NULL;
        }
        if ( NULL != layout_x_internal_gs) {
            dnnLayoutDelete_%(precision)s(layout_x_internal_gs);
            layout_x_internal_gs = NULL;
        }
        if (NULL != layout_gz_internal_gs) {
            dnnLayoutDelete_%(precision)s(layout_gz_internal_gs);
            layout_gz_internal_gs = NULL;
        }
        if (NULL != layout_gs_internal_gs) {
            dnnLayoutDelete_%(precision)s(layout_gs_internal_gs);
            layout_gs_internal_gs = NULL;
        }
        if (NULL != layout_scaleshift) {
            dnnLayoutDelete_%(precision)s(layout_scaleshift);
            layout_scaleshift = NULL;
        }

        // release primitive
        if (NULL != bnBwd) {
            dnnDelete_%(precision)s(bnBwd);
            bnBwd = NULL;
        }
        if (NULL != bnBwdScaleShift) {
            dnnDelete_%(precision)s(bnBwdScaleShift);
            bnBwdScaleShift = NULL;
        }
        if (NULL != prim_to_internal_gz) {
            dnnDelete_%(precision)s(prim_to_internal_gz);
            prim_to_internal_gz = NULL;
        }
        if (NULL != prim_to_internal_gz_gs) {
            dnnDelete_%(precision)s(prim_to_internal_gz_gs);
            prim_to_internal_gz_gs = NULL;
        }
        if (NULL != prim_to_internal_x_gs) {
            dnnDelete_%(precision)s(prim_to_internal_x_gs);
            prim_to_internal_x_gs = NULL;
        }

        // release buffer
        if (NULL != gz_internal_buffer) {
            dnnReleaseBuffer_%(precision)s(gz_internal_buffer);
            gz_internal_buffer = NULL;
        }
        if (NULL != gz_internal_buffer_gs) {
            dnnReleaseBuffer_%(precision)s(gz_internal_buffer_gs);
            gz_internal_buffer_gs = NULL;
        }
        if (NULL != x_internal_buffer_gs) {
            dnnReleaseBuffer_%(precision)s(x_internal_buffer_gs);
            x_internal_buffer_gs = NULL;
        }
        if (NULL != scaleShift_buffer) {
            dnnReleaseBuffer_%(precision)s(scaleShift_buffer);
            scaleShift_buffer = NULL;
        }
        if (NULL != gs_internal_buffer_gs) {
            dnnReleaseBuffer_%(precision)s(gs_internal_buffer_gs);
            gs_internal_buffer_gs = NULL;
        }
        """ % locals()

    def make_node(self, x, gz, scale, shift):
        scale = as_tensor_variable(scale)
        shift = as_tensor_variable(shift)

        assert isinstance(x.type, mkl_type.MKLNdarrayType) and x.ndim == 4
        assert isinstance(gz.type, mkl_type.MKLNdarrayType) and gz.ndim == 4
        return gof.Apply(self, [x, gz, scale, shift], [x.type(), scale.type(), shift.type()])

    def c_code(self, node, name, inp, out, sub):
        x, gz, scale, shift, = inp
        z, g_scale, g_shift, = out
        eps = self.eps
        bias = self.bias
        term = self.term

        dtype = node.inputs[0].type.dtype
        assert dtype in ('float32', 'float64')
        if dtype is 'float32':
            precision = 'F32'
            t = 'float'
        else:
            precision = 'F64'
            t = 'double'

        fail = sub['fail']

        ret = """
        {
            %(t)s *g_scale_buffer_ptr = NULL;
            %(t)s *g_shift_buffer_ptr = NULL;
            %(t)s *scale_buffer_ptr = NULL;
            %(t)s *shift_buffer_ptr = NULL;

            if (first_run) {
                typenum    = MKLNdarray_TYPE((MKLNdarray*)%(x)s);
                x_bs       = MKLNdarray_DIMS(%(x)s)[0];
                x_channels = MKLNdarray_DIMS(%(x)s)[1];
                x_row      = MKLNdarray_DIMS(%(x)s)[2];
                x_col      = MKLNdarray_DIMS(%(x)s)[3];
                sizes[0]   = x_col;
                sizes[1]   = x_row;
                sizes[2]   = x_channels;
                sizes[3]   = x_bs;
                strides[0] = 1;
                strides[1] = sizes[0];
                strides[2] = strides[1] * sizes[1];
                strides[3] = strides[2] * sizes[2];
            }

            if (first_run) {
                // create bn bwd primitive
                CHECK_ERR( dnnBatchNormalizationCreateBackwardData_%(precision)s(&bnBwd, NULL,
                                                                                 MKLNdarray_LAYOUT(%(x)s), %(eps)s), err);

                // layout for internal gz layout
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_gz_internal, bnBwd, dnnResourceDiffDst), err);

                // layout for scaleshift
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_scaleshift, bnBwd, dnnResourceScaleShift), err);

                // buffer for scaleshift
                CHECK_ERR( dnnAllocateBuffer_%(precision)s((void**)(&scaleShift_buffer), layout_scaleshift), err);
                dnnLayoutDelete_%(precision)s(layout_scaleshift);
                layout_scaleshift = NULL;

                if (!%(bias)s) {
                    for (int i = 0; i < x_channels; i++) {
                        if(((%(t)s*)scaleShift_buffer)[i] != 1.0){
                            std::cout<<"scale init failed! "<<((%(t)s*)scaleShift_buffer)[i]<<std::endl;
                            exit(1);
                        }
                        if(((%(t)s*)scaleShift_buffer)[x_channels + i] != 0) {
                            std::cout<<"shift init failed!"<<std::endl;
                            exit(1);
                        }
                    }
                }

                if (%(bias)s) {
                    CHECK_ERR( dnnBatchNormalizationCreateBackwardScaleShift_%(precision)s(&bnBwdScaleShift,
                                                                                           NULL,
                                                                                           MKLNdarray_LAYOUT(%(x)s),
                                                                                           %(eps)s), err);
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_x_internal_gs,
                                                                          bnBwdScaleShift,
                                                                          dnnResourceSrc), err);
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_gz_internal_gs,
                                                                          bnBwdScaleShift,
                                                                          dnnResourceDiffDst), err);
                    CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_gs_internal_gs,
                                                                          bnBwdScaleShift,
                                                                          dnnResourceDiffScaleShift), err);

                    %(g_scale)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(scale)s),
                                                                PyArray_DIMS(%(scale)s),
                                                                PyArray_TYPE(%(scale)s),
                                                                0);
                    %(g_shift)s = (PyArrayObject*)PyArray_ZEROS(PyArray_NDIM(%(shift)s),
                                                                PyArray_DIMS(%(shift)s),
                                                                PyArray_TYPE(%(shift)s),
                                                                0);

                    if ((NULL == %(g_shift)s) || (NULL == %(g_scale)s)) {
                        std::cout<<"Allocate g_scale buffer failed"<<std::endl;
                    }

                    g_scale_buffer_ptr = (%(t)s*)PyArray_DATA(%(g_scale)s);
                    g_shift_buffer_ptr = (%(t)s*)PyArray_DATA(%(g_shift)s);

                    if (NULL == gs_internal_buffer_gs) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s(&gs_internal_buffer_gs, layout_gs_internal_gs), err);
                    }
                }
            }

            if (! (%(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == MKLNdarray_NDIM(%(x)s)
                && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
                && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
                && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
                && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {

                if (%(z)s) Py_XDECREF(%(z)s);
                %(z)s = (MKLNdarray*)MKLNdarray_New(DIMENSION, typenum);
                if (! %(z)s) {
                    %(fail)s;
                }

                int status = MKLNdarray_set_structure(%(z)s, DIMENSION, MKLNdarray_DIMS(%(x)s));
                if (0 != status) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &bnBwd, dnnResourceDiffSrc);
                if (0 != status) {
                    std::cout<<"MKLNdarray_create_buffer_from_primitive failed: "<<status<<", line: "<<__LINE__<<std::endl;
                    exit(1);
                }
            }

            if (%(bias)s) {
                scale_buffer_ptr = (%(t)s*)PyArray_DATA(%(scale)s);
                shift_buffer_ptr = (%(t)s*)PyArray_DATA(%(shift)s);

                // Read data from bias weight and bias term buffern to ScaleShift buffer
                for (int i = 0; i < x_channels; i++) {
                    ((%(t)s*)scaleShift_buffer)[i] = scale_buffer_ptr[i];
                    ((%(t)s*)scaleShift_buffer)[x_channels + i] = 0;
                    if (%(term)s) {
                        ((%(t)s*)scaleShift_buffer)[x_channels + i] = shift_buffer_ptr[i];
                    }
                }
            }

            // compare gz layout with internal layout, do internal to internal conversion
            if (! dnnLayoutCompare_%(precision)s(layout_gz_internal, MKLNdarray_LAYOUT(%(gz)s))) {
                if (NULL == prim_to_internal_gz) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&prim_to_internal_gz,
                                                                 MKLNdarray_LAYOUT(%(gz)s),
                                                                 layout_gz_internal), err);
                }

                if (NULL == gz_internal_buffer) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s(&gz_internal_buffer, layout_gz_internal), err);
                }

                if (prim_to_internal_gz) {
                    CHECK_ERR( dnnConversionExecute_%(precision)s(prim_to_internal_gz,
                                                                  MKLNdarray_DATA(%(gz)s),
                                                                  gz_internal_buffer), err);
                }
                bn_res[dnnResourceDiffDst] = gz_internal_buffer;
            } else {
                bn_res[dnnResourceDiffDst] = MKLNdarray_DATA(%(gz)s);
            }

            // bwd with respect to data
            bn_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
            bn_res[dnnResourceScaleShift] = (void*)scaleShift_buffer;
            bn_res[dnnResourceDiffSrc] = MKLNdarray_DATA(%(z)s);
            bn_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
            CHECK_ERR( dnnExecute_%(precision)s(bnBwd, bn_res), err);

            if (%(bias)s) {
                // compare gz layout with internal layout, do internal to internal conversion
                if (! dnnLayoutCompare_%(precision)s(layout_gz_internal_gs, MKLNdarray_LAYOUT(%(gz)s))) {
                    if (NULL == prim_to_internal_gz_gs) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&prim_to_internal_gz_gs,
                                                                     MKLNdarray_LAYOUT(%(gz)s),
                                                                     layout_gz_internal_gs), err);
                    }

                    if (NULL == gz_internal_buffer_gs) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s(&gz_internal_buffer_gs, layout_gz_internal_gs), err);
                    }

                    if (prim_to_internal_gz_gs) {
                        CHECK_ERR( dnnConversionExecute_%(precision)s(prim_to_internal_gz_gs,
                                                                      MKLNdarray_DATA(%(gz)s),
                                                                      gz_internal_buffer_gs), err);
                    }
                    BatchNormBwdScaleShift_res[dnnResourceDiffDst] = gz_internal_buffer_gs;
                } else {
                    BatchNormBwdScaleShift_res[dnnResourceDiffDst] = MKLNdarray_DATA(%(gz)s);
                }

                //compare x input laout with internal layout, do internal to internal conversion
                if (! dnnLayoutCompare_%(precision)s(layout_x_internal_gs, MKLNdarray_LAYOUT(%(x)s))) {
                    if (NULL == prim_to_internal_x_gs) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&prim_to_internal_x_gs,
                                                                     MKLNdarray_LAYOUT(%(x)s),
                                                                     layout_x_internal_gs), err);
                    }

                    if (NULL == x_internal_buffer_gs) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s(&x_internal_buffer_gs, layout_x_internal_gs), err);
                    }

                    if (prim_to_internal_x_gs) {
                        CHECK_ERR( dnnConversionExecute_%(precision)s(prim_to_internal_x_gs,
                                                                      MKLNdarray_DATA(%(x)s),
                                                                      x_internal_buffer_gs), err);
                    }
                    BatchNormBwdScaleShift_res[dnnResourceSrc] = x_internal_buffer_gs;
                } else {
                    BatchNormBwdScaleShift_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
                }

                BatchNormBwdScaleShift_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
                BatchNormBwdScaleShift_res[dnnResourceDiffScaleShift] = gs_internal_buffer_gs;
                CHECK_ERR( dnnExecute_%(precision)s(bnBwdScaleShift, BatchNormBwdScaleShift_res), err);

                for (int i = 0; i < x_channels; i++) {
                    g_scale_buffer_ptr[i] = ((%(t)s*)gs_internal_buffer_gs)[i];
                    g_shift_buffer_ptr[i] = 0;
                    if (%(term)s) {
                        g_shift_buffer_ptr[i] =  ((%(t)s*)gs_internal_buffer_gs)[x_channels + i];
                    }
                }
            }
            first_run = 0;
        }
        """ % locals()
        return ret

    def c_code_cache_version(self):
        return (0, 1, 1)
