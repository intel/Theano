import theano.tensor as T
from theano.gof import Apply, Op
from theano.tensor.blas import ldflags
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.gradient import DisconnectedType 
from theano.contrib.mkl.mkl_helper import header_text
from theano.contrib.mkl.mkl_type import MKLNdarrayType


class MKLOp(Op):
    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        compile_args = ldflags(libs=False, flags=True)
        compile_args += super(MKLOp, self).c_compile_args()
        return compile_args

    def c_support_code(self):
        ccode = header_text()
        ccode += """
        #define DIMENSION  4

        #define CHECK_ERR(f, err) \\
            do { \\
                (err) = (f); \\
                if ((err) != E_SUCCESS) { \\
                    printf("Error in file [%s:%d], err code (%d)", \\
                           __FILE__, __LINE__, err); \\
                    exit(1); \\
                } \\
            } while(0)
        """

        return ccode


class BaseConvertOp(MKLOp):
    def c_support_code_struct(self, node, name):
        ccode = """
        dnnError_t err;
        int first_run;
        void *internal_buf;
        void *user_buf;
        dnnLayout_t layout_internal;
        dnnLayout_t layout_user;
        dnnPrimitive_t to_internal;
        dnnPrimitive_t from_internal;
        dnnPrimitive_t primitive;
        void *convert_resources[dnnResourceNumber];
        size_t bottomSize[DIMENSION];
        size_t bottomStride[DIMENSION];
        """
        return ccode

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        first_run = 1;
        internal_buf = NULL;
        user_buf = NULL;
        layout_internal = NULL;
        layout_user = NULL;
        to_internal = NULL;
        from_internal = NULL;
        primitive = NULL;
        """
        return ccode

    def c_code_cache_version(self):
        return (1, 0)


class MKLToNdarray(BaseConvertOp):
    __props__ = ()

    def make_node(self, x):
        assert isinstance(x.type, MKLNdarrayType)
        return Apply(self, [x], [T.TensorType(broadcastable=x.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [I2UGrad()(x, gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        fail = sub['fail']

        ccode = """
            if (%(z)s) {
                Py_XDECREF(%(z)s);
            }
            %(z)s = (PyArrayObject*)MKLNdarray_CreateArrayObj(%(x)s);
            if (!%(z)s) {
                %(fail)s;
            }
        """ % locals()

        return ccode


class I2UGrad(BaseConvertOp):
    __props__ = ()

    def make_node(self, x, gz):
        out = x.type()
        return Apply(self, [x, gz], [out])

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        sub['x'] = x
        sub['z'] = z
        sub['gz'] = gz

        if 'float32' == node.inputs[0].type.dtype:
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
            sub['type'] = "float"
        elif "float64" == node.inputs[0].type.dtype:
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
            sub["type"] = "double"
        else:
            raise TypeError("Type %s not implemented" %
                            node.inputs[0].type.dtype)

        ccode = """
        int status = 0;
        if (! (%(z)s
            && MKLNdarray_Check((PyObject*)%(z)s)
            && MKLNdarray_NDIM(%(z)s) == MKLNdarray_NDIM(%(x)s)
            && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
            && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
            && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
            && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {

            if (%(z)s) Py_XDECREF(%(z)s);

            %(z)s = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(%(x)s), MKLNdarray_TYPE(%(x)s));
            if (NULL == %(z)s) {
                %(fail)s;
            }

            status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(x)s), MKLNdarray_DIMS(%(x)s));
            if (0 != status) {
                %(fail)s;
            }

            status = MKLNdarray_copy_layout(%(z)s, %(x)s, MNDA_DATA);
            if (0 != status) {
                %(fail)s;
            }

            status = MKLNdarray_create_buffer_from_layout(%(z)s, MNDA_DATA);
            if (0 != status) {
                %(fail)s;
            }
        }

        //create usr layerout of gz
        if (1 == first_run) {
            size_t gz_size[4] = {0};
            size_t gz_stride[4] = {0};

            gz_size[0] = PyArray_DIMS(%(gz)s)[3];  //w
            gz_size[1] = PyArray_DIMS(%(gz)s)[2];  //h
            gz_size[2] = PyArray_DIMS(%(gz)s)[1];  //c
            gz_size[3] = PyArray_DIMS(%(gz)s)[0];  //n
            gz_stride[0] = 1;
            gz_stride[1] = gz_size[0];
            gz_stride[2] = gz_size[0] * gz_size[1];
            gz_stride[3] = gz_size[0] * gz_size[1] * gz_size[2];

            CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user,
                                                     DIMENSION,
                                                     gz_size,
                                                     gz_stride), err);

            if (!dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(z)s), layout_user)) {
                CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal,
                                                             layout_user,
                                                             MKLNdarray_LAYOUT(%(z)s)), err);
            } else {
                to_internal = NULL;
            }
        }

        if (to_internal) {
            CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                          PyArray_DATA(%(gz)s),
                                                          MKLNdarray_DATA(%(z)s)), err);
        } else {
            memcpy((void*)MKLNdarray_DATA(%(z)s),(void*)PyArray_DATA(%(gz)s), %(z)s->data_size);
        }
        first_run = 0;
        """ % sub
        return ccode


class U2IConv(BaseConvertOp):
    __props__ = ('imshp', 'kshp', 'border_mode', 'subsample', 'filter_dilation')

    def __init__(self, imshp=None, kshp=None, border_mode='valid', subsample=(1, 1), filter_dilation=(1, 1)):
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        self.imshp = imshp
        self.kshp = kshp
        self.filter_dilation = filter_dilation

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim is not 4:
            raise TypeError('U2IConv: input x should be an 4-dim tensor')
        return Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [MKLToNdarray()(gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        dH, dW = self.subsample

        if len(self.kshp) == 5:
            grp, k_n, k_c, k_h, k_w = self.kshp
            gkshp = [grp * k_n, grp * k_c, k_h, k_w]
        else:
            k_n, k_c, k_h, k_w = self.kshp
            grp = 1
            gkshp = self.kshp

        if None in self.imshp:
            i_n, i_c, i_h, i_w = 0, 0, 0, 0
            o_n, o_c, o_h, o_w = 0, 0, 0, 0
        else:
            i_n, i_c, i_h, i_w = self.imshp
            o_n, o_c, o_h, o_w = get_conv_output_shape(image_shape=self.imshp,
                                                       kernel_shape=self.kshp,
                                                       border_mode=self.border_mode,
                                                       filter_dilation=self.filter_dilation,
                                                       subsample=self.subsample)

        if self.border_mode == 'valid':
            padH, padW = (0, 0)
        elif self.border_mode == 'full':
            padH, padW = ((k_h - 1), (k_w - 1))
        elif self.border_mode == 'half':
            padH, padW = ((k_h / 2), (k_w / 2))
        elif isinstance(self.border_mode, tuple):
            padH, padW = self.border_mode
        else:
            raise ValueError("border_mode must have two elements")

        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise TypeError("Type %s is not supported!" %
                            node.inputs[0].type.dtype)
        fail = sub['fail']

        ccode = """
            int status = 0;
            npy_intp* x_dims = PyArray_DIMS(%(x)s);

            if (1 == first_run) {
                int convPadding[2];
                size_t convStride[2], weightSize[5], weightStride[5], imageSize[4], imageStride[4], zSize[4], zStride[4];
                convStride[0] = %(dW)s;
                convStride[1] = %(dH)s;
                convPadding[0] = -%(padW)s;
                convPadding[1] = -%(padH)s;

                imageSize[0] = %(i_w)s;  //w
                imageSize[1] = %(i_h)s;  //h
                imageSize[2] = %(i_c)s;  //c
                imageSize[3] = %(i_n)s;  //n

                if (0 == imageSize[0] || 0 == imageSize[1] || 0 == imageSize[2] || 0 == imageSize[3]) {
                    imageSize[0] = x_dims[3];
                    imageSize[1] = x_dims[2];
                    imageSize[2] = x_dims[1];
                    imageSize[3] = x_dims[0];
                }

                imageStride[0] = 1;
                imageStride[1] = imageSize[0];
                imageStride[2] = imageSize[0] * imageSize[1];
                imageStride[3] = imageSize[0] * imageSize[1] * imageSize[2];

                weightSize[0] = %(k_w)s;
                weightSize[1] = %(k_h)s;
                weightSize[2] = %(k_c)s;
                weightSize[3] = %(k_n)s;
                weightSize[4] = %(grp)s;
                weightStride[0] = 1;
                weightStride[1] = weightSize[0];
                weightStride[2] = weightSize[0] * weightSize[1];
                weightStride[3] = weightSize[0] * weightSize[1] * weightSize[2];
                weightStride[4] = weightSize[0] * weightSize[1] * weightSize[2] * weightSize[3];

                zSize[0] = %(o_w)s;
                zSize[1] = %(o_h)s;
                zSize[2] = %(o_c)s;
                zSize[3] = %(o_n)s;

                if (0 == zSize[0] || 0 == zSize[1] || 0 == zSize[2] || 0== zSize[3]) {
                    zSize[0] = (imageSize[0] - 2 * convPadding[0] - weightSize[0]) / convStride[0] + 1;
                    zSize[1] = (imageSize[1] - 2 * convPadding[1] - weightSize[1]) / convStride[1] + 1;
                    zSize[2] = weightSize[3];
                    zSize[3] = imageSize[3];
                }

                zStride[0] = 1;
                zStride[1] = zSize[0];
                zStride[2] = zSize[0] * zSize[1];
                zStride[3] = zSize[0] * zSize[1] * zSize[2];

                const int group = %(grp)s;
                // create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, imageSize, imageStride), err );
                // create convolution primitive
                CHECK_ERR( dnnGroupsConvolutionCreateForward_%(precision)s(&primitive, NULL,
                           dnnAlgorithmConvolutionDirect, group, DIMENSION, imageSize, zSize,
                           weightSize, convStride, convPadding, dnnBorderZeros), err );
            }

            int ndim = PyArray_NDIM(%(x)s);
            assert(ndim == DIMENSION);
            size_t dims[DIMENSION] = {0};

            if (!( %(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == ndim
                && MKLNdarray_DIMS(%(z)s)[0] == x_dims[0]
                && MKLNdarray_DIMS(%(z)s)[1] == x_dims[1]
                && MKLNdarray_DIMS(%(z)s)[2] == x_dims[2]
                && MKLNdarray_DIMS(%(z)s)[3] == x_dims[3]) ) {

                if (%(z)s) {
                    Py_XDECREF(%(z)s);
                }

                for (int i = 0; i < ndim; i++) {
                    dims[i] = (size_t)x_dims[i];
                }

                %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, PyArray_TYPE(%(x)s));
                if (!%(z)s) {
                    %(fail)s;
                }

                status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceSrc);
                if (status != 0) {
                    %(fail)s;
                }
            }

            if (1 == first_run) {
                if(!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal,
                                                                 layout_user,
                                                                 MKLNdarray_LAYOUT(%(z)s)), err );
                } else {
                    to_internal = NULL;
                }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;
        """ % locals()
        return ccode


class U2IPool(BaseConvertOp):
    __props__ = ('ignore_border', 'mode')

    def __init__(self, ignore_border=False, mode='max', ndim=2):
        self.ignore_border = ignore_border
        self.mode = mode
        self.ndim = ndim

    def make_node(self, x, ws, stride=None, pad=(0, 0)):
        x = T.as_tensor_variable(x)
        if stride is None:
            stride = ws

        ws = T.as_tensor_variable(ws)
        stride = T.as_tensor_variable(stride)
        pad = T.as_tensor_variable(pad)
        nd = self.ndim

        broad = x.broadcastable[:-nd] + (False,) * nd
        out = MKLNdarrayType(x.dtype, broad)

        return Apply(self, [x, ws, stride, pad], [out()])

    def grad(self, inp, grads):
        x, ws, stride, pad = inp
        gz, = grads
        disc = [DisconnectedType()() for i in inp[1:]]

        return [MKLToNdarray()(gz)] + disc

    def c_code(self, node, name, inp, out, sub):
        x, ws, stride, pad = inp
        z, = out

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise TypeError("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        if self.ignore_border:
            borderType = 'dnnBorderZerosAsymm'
            ignore_border = 1
        else:
            borderType = 'dnnBorderZeros'
            ignore_border = 0

        if 'max' == self.mode:
            algo = "dnnAlgorithmPoolingMax"
        elif 'min' == self.mode:
            algo = 'dnnAlgorithmPoolingMin'
        elif 'average_exc_pad' == self.mode:
            algo = "dnnAlgorithmPoolingAvgExcludePadding"
        elif 'average_inc_pad' == self.mode:
            algo = "dnnAlgorithmPoolingAvgIncludePadding"
        else:
            raise ValueError("mode must be one of 'max', 'min', "
                             "'average_exc_pad', and 'average_inc_pad'")

        ccode = """
            int status = 0;
            int typenum = PyArray_TYPE(%(x)s);
            int ndim = PyArray_NDIM(%(x)s);
            size_t dims[MNDA_MAX_NDIM] = {0};
            for (int i = 0; i < ndim; i++) {
                dims[i] = (size_t)PyArray_DIMS(%(x)s)[i];
            }

            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  //n
                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                size_t kernel_h = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 0));
                size_t kernel_w = *((npy_intp*)PyArray_GETPTR1(%(ws)s, 1));
                size_t stride_h = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 0));
                size_t stride_w = *((npy_intp*)PyArray_GETPTR1(%(stride)s, 1));
                size_t pad_h = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 0));
                size_t pad_w = *((npy_intp*)PyArray_GETPTR1(%(pad)s, 1));

                size_t kernelSize[2] = {kernel_w, kernel_h};
                size_t kernelStride[2] = {stride_w, stride_h};
                int inputOffset[4] = {0};

                if (%(ignore_border)s) {
                    inputOffset[0] = -pad_w;
                    inputOffset[1] = -pad_h;
                    inputOffset[2] = -pad_w;
                    inputOffset[3] = -pad_h;
                } else {
                    inputOffset[0] = -pad_w;
                    inputOffset[1] = -pad_h;
                }

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, bottomSize, bottomStride), err );

                CHECK_ERR( dnnPoolingCreateForward_%(precision)s(&primitive, NULL, %(algo)s,
                           layout_user, kernelSize, kernelStride, inputOffset, %(borderType)s), err );
            }

            if ( !(%(z)s
                   && MKLNdarray_Check((PyObject *)%(z)s)
                   && MKLNdarray_NDIM(%(z)s) == PyArray_NDIM(%(x)s)
                   && MKLNdarray_DIMS(%(z)s)[0] == PyArray_DIMS(%(x)s)[0]
                   && MKLNdarray_DIMS(%(z)s)[1] == PyArray_DIMS(%(x)s)[1]
                   && MKLNdarray_DIMS(%(z)s)[2] == PyArray_DIMS(%(x)s)[2]
                   && MKLNdarray_DIMS(%(z)s)[3] == PyArray_DIMS(%(x)s)[3]) ) {
                if (%(z)s) Py_XDECREF(%(z)s);

                %(z)s = (MKLNdarray *)MKLNdarray_New(PyArray_NDIM(%(x)s), typenum);
                if (NULL == %(z)s) {
                    %(fail)s
                }

                status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceSrc);
                if (status != 0) {
                    %(fail)s
                }
            }

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                    if (NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, MKLNdarray_LAYOUT(%(z)s)), err );
                    }
                }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;

            #ifdef _MKL_DEBUG_
                std::cout << "U2IPool: from buffer: " << convert_resources[dnnResourceFrom] << " to buffer: " << convert_resources[dnnResourceTo] << std::endl;
            #endif
        """ % locals()
        return ccode

    def connection_pattern(self, node):
        return [[1], [0], [0], [0]]


class U2IRelu(BaseConvertOp):
    __props__ = ('slope', )

    def __init__(self, slope=0):
        self.slope = slope

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        out = MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()

        return Apply(self, [x], [out])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [MKLToNdarray()(gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        slope = self.slope
        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise TypeError("Type %s is not supported!" %
                            node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            int typenum = PyArray_TYPE(%(x)s);
            int ndim = PyArray_NDIM(%(x)s);
            size_t dims[MNDA_MAX_NDIM] = {0};
            for (int i = 0; i < ndim; i++) {
                dims[i] = (size_t)PyArray_DIMS(%(x)s)[i];
            }

            if (1 == first_run) {
                bottomSize[0] = PyArray_DIMS(%(x)s)[3];  //w
                bottomSize[1] = PyArray_DIMS(%(x)s)[2];  //h
                bottomSize[2] = PyArray_DIMS(%(x)s)[1];  //c
                bottomSize[3] = PyArray_DIMS(%(x)s)[0];  //n
                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnReLUCreateForward_%(precision)s(&primitive, NULL, layout_user, %(slope)s), err );
            }

            if ( !(%(z)s
                   && MKLNdarray_Check((PyObject *)%(z)s)
                   && MKLNdarray_NDIM(%(z)s) == PyArray_NDIM(%(x)s)
                   && MKLNdarray_DIMS(%(z)s)[0] == PyArray_DIMS(%(x)s)[0]
                   && MKLNdarray_DIMS(%(z)s)[1] == PyArray_DIMS(%(x)s)[1]
                   && MKLNdarray_DIMS(%(z)s)[2] == PyArray_DIMS(%(x)s)[2]
                   && MKLNdarray_DIMS(%(z)s)[3] == PyArray_DIMS(%(x)s)[3]) ) {
                if (%(z)s) Py_XDECREF(%(z)s);

                %(z)s = (MKLNdarray *)MKLNdarray_New(PyArray_NDIM(%(x)s), typenum);
                if (NULL == %(z)s) {
                    %(fail)s
                }

                int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceSrc);
                if (status != 0) {
                    %(fail)s
                }
            }

            if (!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                if (NULL == to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, MKLNdarray_LAYOUT(%(z)s)), err );
                }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;
            #ifdef _MKL_DEBUG_
                std::cout << "U2IRelu: from buffer: " << convert_resources[dnnResourceFrom] << " to buffer: " << convert_resources[dnnResourceTo] << std::endl;
            #endif
        """ % locals()
        return ccode

class U2ILRN(BaseConvertOp):
    __props__ = ('alpha', 'beta', 'k', 'size')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.size = n

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim variable.')
        return Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads

        return [MKLToNdarray()(gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        alpha = self.alpha
        beta = self.beta
        k = self.k
        size = self.size

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise TypeError('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            int ndim = PyArray_NDIM(%(x)s);
            int dtype = PyArray_TYPE(%(x)s);
            npy_intp* d = PyArray_DIMS(%(x)s);

            size_t dims[MNDA_MAX_NDIM] = {0};
            for (int i = 0; i < ndim; i++) {
                dims[i] = (size_t)d[i];
            }

            if (1 == first_run) {
                bottomSize[0] = d[3]; //w
                bottomSize[1] = d[2]; //h
                bottomSize[2] = d[1]; //c
                bottomSize[3] = d[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnLRNCreateForward_%(precision)s(&primitive, NULL, layout_user, %(size)s, %(alpha)s, %(beta)s, %(k)s), err );
            }

            if (!( %(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == ndim
                && MKLNdarray_DIMS(%(z)s)[0] == d[0]
                && MKLNdarray_DIMS(%(z)s)[1] == d[1]
                && MKLNdarray_DIMS(%(z)s)[2] == d[2]
                && MKLNdarray_DIMS(%(z)s)[3] == d[3]) ) {

                if (%(z)s) Py_XDECREF(%(z)s);
                %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
                if (!%(z)s) {
                    %(fail)s;
                }

                int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceSrc);
                if (status != 0) {
                    %(fail)s;
                }
            }

            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                    if (NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, MKLNdarray_LAYOUT(%(z)s)), err );
                    }
                }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;
        """ % locals()
        return ccode

class U2IBatchNormalization(BaseConvertOp):
    __props__ = ('eps',)

    def __init__(self, eps=1e-5):
        self.eps = eps

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('The input should be a 4-dim tensor.')
        return Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [MKLToNdarray()(gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        eps = self.eps

        if 'float32' == node.inputs[0].type.dtype:
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            precision = 'F64'
        else:
            raise TypeError('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']

        ccode = """
            int ndim = PyArray_NDIM(%(x)s);
            int dtype = PyArray_TYPE(%(x)s);
            npy_intp* d = PyArray_DIMS(%(x)s);

            size_t dims[MNDA_MAX_NDIM] = {0};
            for (int i = 0; i < ndim; i++) {
                dims[i] = (size_t)d[i];
            }

            if (1 == first_run) {
                bottomSize[0] = d[3]; //w
                bottomSize[1] = d[2]; //h
                bottomSize[2] = d[1]; //c
                bottomSize[3] = d[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnBatchNormalizationCreateForward_%(precision)s(&primitive, NULL, layout_user, %(eps)s), err);
            }

            if (!( %(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == ndim
                && MKLNdarray_DIMS(%(z)s)[0] == d[0]
                && MKLNdarray_DIMS(%(z)s)[1] == d[1]
                && MKLNdarray_DIMS(%(z)s)[2] == d[2]
                && MKLNdarray_DIMS(%(z)s)[3] == d[3]) ) {

                if (%(z)s) Py_XDECREF(%(z)s);
                %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
                if (!%(z)s) {
                    %(fail)s;
                }

                int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceSrc);
                if (status != 0) {
                    %(fail)s;
                }
            }
            
            if (1 == first_run) {
                if (!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                    if (NULL == to_internal) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, MKLNdarray_LAYOUT(%(z)s)), err );
                    }
                }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;
        """ % locals()
        return ccode

class U2IElemwiseSum(BaseConvertOp):
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=(1.0, )):
        self.inp_num = inp_num
        if isinstance(coeff, tuple):
            self.coeff = coeff
        elif isinstance(coeff, list):
            self.coeff = tuple(coeff)
        else:
            raise TypeError('Coeff should be a tuple or list.')
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to number of coefficients.')

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('U2IElemwiseSum inputs should be 4-dim tensor')
        return Apply(self, [x], [MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [MKLToNdarray()(gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        coeff = self.coeff
        inp_num = self.inp_num

        if 'float32' == node.inputs[0].type.dtype:
            sub['type'] = 'float'
            precision = 'F32'
        elif 'float64' == node.inputs[0].type.dtype:
            sub['type'] = 'double'
            precision = 'F64'
        else:
            raise TypeError('Type %s is not supported!' % node.inputs[0].type.dtype)

        fail = sub['fail']
        sub['len'] = inp_num

        ccode = """
            %(type)s coeffs[%(len)s] = {1.0};
        """ % sub

        for i, co in enumerate(coeff):
            ccode = ccode + """
            coeffs[%s] = %s;
            """ % (i, co)

        ccode = ccode + """
            int ndim = PyArray_NDIM(%(x)s);
            int dtype = PyArray_TYPE(%(x)s);
            npy_intp* d = PyArray_DIMS(%(x)s);

            size_t dims[MNDA_MAX_NDIM] = {0};
            for (int i = 0; i < ndim; i++) {
                dims[i] = (size_t)d[i];
            }

           if (1 == first_run) {
                bottomSize[0] = d[3]; //w
                bottomSize[1] = d[2]; //h
                bottomSize[2] = d[1]; //c
                bottomSize[3] = d[0]; //n

                bottomStride[0] = 1;
                bottomStride[1] = bottomStride[0] * bottomSize[0];
                bottomStride[2] = bottomStride[1] * bottomSize[1];
                bottomStride[3] = bottomStride[2] * bottomSize[2];

                //create user layout
                CHECK_ERR( dnnLayoutCreate_%(precision)s(&layout_user, DIMENSION, bottomSize, bottomStride), err );
                CHECK_ERR( dnnSumCreate_%(precision)s(&primitive, NULL, %(inp_num)s, layout_user, coeffs), err);
            }

            if (!( %(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == ndim
                && MKLNdarray_DIMS(%(z)s)[0] == d[0]
                && MKLNdarray_DIMS(%(z)s)[1] == d[1]
                && MKLNdarray_DIMS(%(z)s)[2] == d[2]
                && MKLNdarray_DIMS(%(z)s)[3] == d[3]) ) {

                if (%(z)s) Py_XDECREF(%(z)s);
                %(z)s = (MKLNdarray*)MKLNdarray_New(ndim, dtype);
                if (!%(z)s) {
                    %(fail)s;
                }

                int status = MKLNdarray_set_structure(%(z)s, ndim, dims);
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceMultipleSrc);
                if (status != 0) {
                    %(fail)s;
                }
            }

            if (1 == first_run) {
               if (!dnnLayoutCompare_%(precision)s(layout_user, MKLNdarray_LAYOUT(%(z)s))) {
                   if (NULL == to_internal) {
                       CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, layout_user, MKLNdarray_LAYOUT(%(z)s)), err );
                   }
               }
            }

            if (to_internal) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal,
                                                              PyArray_DATA(%(x)s),
                                                              MKLNdarray_DATA(%(z)s)), err );
            } else {
                memcpy(MKLNdarray_DATA(%(z)s), (void*)PyArray_DATA(%(x)s), %(z)s->data_size);
            }

            first_run = 0;
        """ % locals()
        return ccode

