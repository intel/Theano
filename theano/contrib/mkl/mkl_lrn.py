
from theano import gof, tensor
from theano.contrib.mkl import basic_ops, mkl_type


class AbstractLRN(gof.Op):
    """
    LRN: local response normalization.
    An abstract OP for LRN, called in /tensor/lrn/py.
    This OP will be optimized in local OPT with LRN OP.
    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        super(AbstractLRN, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [AbstractLRNGrad(alpha=self.alpha,
                                beta=self.beta,
                                k=self.k,
                                n=self.n)(x, gz)]

    def perform(self, node, inp, out):
        print('AbstracLRN is a abstract OP, should not exist in graph..')
        x, = inp
        z, = out


class AbstractLRNGrad(gof.Op):
    """
    LRN: local response normalization.
    An abstract OP for LRN gradient. It will be called in AbstractLRN.grad().
    This OP will be optimized in local OPT with LRNGrad OP.
    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        super(AbstractLRNGrad, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x, gz):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        return gof.Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out):
        print('AbstracLRNGrad is a abstract OP, should not exist in graph..')
        x, gz = inp
        gx, = out


class LRN(basic_ops.MKLOp):
    """
    LRN: local response normalization (Across Maps)

    Refer to the below link for the definition of LRN
        https://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_
        response_normalization_layer_(across_maps)

    The activity of a neuron is divided only by the "adjacent" of activities
    which are in the same spatial postion but in different maps (channels).
    'c' stands for current channel index.

        F[c][x,y] = (1 + (alpha*1.0/n) * sum(F[c - n/2][x,y]^2,
                    F[c + n/2][x,y]^2))^beta

    Parameters
    ----------
    alpha: hyper-parameter
    beta : hyper-parameter
    k    : hyper-parameter
    n    : hyper-parameter, indicates how many nearby maps to use for normalization.
    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.k = k

    def make_node(self, x):
        # x = tensor.as_tensor_variable(x)
        if not isinstance(x.type, mkl_type.MKLNdarrayType):
            raise TypeError('LRN: input x should be an instance of MKLNdarrayType')
        if x.type.ndim != 4:
            raise TypeError('Input should be a 4-dim tensor')
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [LRNGrad(alpha=self.alpha, beta=self.beta, k=self.k,
                        n=self.n)(x, gz)]

    def c_support_code_struct(self, node, name):
        support_code = """
            int typenum;
            dnnError_t err;
            int first_run;
            void* buffer_internal;
            dnnLayout_t layout_internal;

            dnnPrimitive_t to_internal;
            dnnPrimitive_t primitive;

            size_t bottomSize[DIMENSION];
            size_t bottomStride[DIMENSION];

            void* lrn_res[dnnResourceNumber];
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
            typenum = 0;
            first_run = 1;
            buffer_internal = NULL;
            layout_internal = NULL;
            to_internal = NULL;
            primitive = NULL;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        ccode = """
            // release layout
            if (NULL != layout_internal) {
                dnnLayoutDelete_%(precision)s(layout_internal);
                layout_internal = NULL;
            }

            // release primitive
            if (NULL != primitive) {
                dnnDelete_%(precision)s(primitive);
                primitive = NULL;
            }

            if (NULL != to_internal) {
                dnnDelete_%(precision)s(to_internal);
                to_internal = NULL;
            }

            // release buffer
            if (NULL != buffer_internal) {
                dnnReleaseBuffer_%(precision)s(buffer_internal);
                buffer_internal = NULL;
            }
        """ % locals()
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        alpha = self.alpha
        beta = self.beta
        size = self.n
        k = self.k

        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        fail = sub['fail']

        ccode = """
        {
            int status = 0;
            if (first_run) {
                typenum = MKLNdarray_TYPE((MKLNdarray*)%(x)s);

                bottomSize[0] = MKLNdarray_DIMS(%(x)s)[3];  // w
                bottomSize[1] = MKLNdarray_DIMS(%(x)s)[2];  // h
                bottomSize[2] = MKLNdarray_DIMS(%(x)s)[1];  // c
                bottomSize[3] = MKLNdarray_DIMS(%(x)s)[0];  // n

                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];
            }

            if (first_run) {
                // primitive for LRN
                CHECK_ERR( dnnLRNCreateForward_%(precision)s(&primitive, NULL, MKLNdarray_LAYOUT(%(x)s),
                                                             %(size)s, %(alpha)s, %(beta)s, %(k)s), err );

                // internal layout for input
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&layout_internal, primitive, dnnResourceSrc), err );

            }

            // create workspace buffer in x
            if (MKLNdarray_WORKSPACE(%(x)s) == NULL) {
                status = MKLNdarray_create_buffer_from_primitive(%(x)s, &primitive, dnnResourceWorkspace);
                if (0 != status) {
                    std::cout<< "MKLNdarray_create_buffer_from_primitive failed, return: "<<status<<", line: "<<__LINE__<<std::endl;
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
                    %(fail)s
                }

                status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(x)s), MKLNdarray_DIMS(%(x)s));
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceDst);
                if (status != 0) {
                    std::cout<<"MKLNdarray_create_buffer_from_primitive fail, return: "<<status<<", line: "<<__LINE__<<std::endl;
                    exit(1);
                }
            }  // else reuse %(z)s

            if (! dnnLayoutCompare_%(precision)s(layout_internal, MKLNdarray_LAYOUT(%(x)s))) {
                if (NULL == to_internal) {
                    CHECK_ERR( dnnConversionCreate_%(precision)s(&to_internal, MKLNdarray_LAYOUT(%(x)s), layout_internal), err);
                }

                if (NULL == buffer_internal) {
                    CHECK_ERR( dnnAllocateBuffer_%(precision)s(&buffer_internal, layout_internal), err);
                }

                if (to_internal) {
                    CHECK_ERR( dnnConversionExecute_%(precision)s(to_internal, MKLNdarray_DATA(%(x)s), buffer_internal), err);
                }

                lrn_res[dnnResourceSrc] = buffer_internal;
            } else {
                lrn_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
            }
            lrn_res[dnnResourceDst] = MKLNdarray_DATA(%(z)s);
            lrn_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
            CHECK_ERR( dnnExecute_%(precision)s(primitive, lrn_res), err );
            first_run = 0;
        }
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (0, 1, 1)


class LRNGrad(basic_ops.MKLOp):
    """
    LRN: local response normalization
    Grad Function of NormAcrossMap
        roOut = gz * f(x)
        f(x) = 1/(1 + (alpha/n)*sum(x*x))**beta - 2*x*alpha*beta*sum(x)/(1+(alpha/n)*sum(x*x))**(beta+1)

    Parameters
    ----------
    alpha:
    beta :
    n    : indicates how many nearby maps to use for normalization.

    """
    __props__ = ('alpha', 'beta', 'k', 'n')

    def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.n = n

    def make_node(self, x, gz):
        if not isinstance(x.type, mkl_type.MKLNdarrayType) or x.type.ndim != 4:
            raise TypeError('LRNGrad: Input x type error or dimension error.')
        if not isinstance(gz.type, mkl_type.MKLNdarrayType) or gz.type.ndim != 4:
            raise TypeError('LRNGrad: Inputs gz type error or dimension error.')

        return gof.Apply(self, [x, gz], [x.type()])

    def c_support_code_struct(self, node, name):
        support_code = """
            int typenum;
            dnnError_t err;
            int first_run;
            dnnPrimitive_t primitive;
            size_t bottomSize[DIMENSION];
            size_t bottomStride[DIMENSION];
            void* lrn_res[dnnResourceNumber];
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
            typenum = 0;
            first_run = 1;
            primitive = NULL;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        ccode = """
            // release primitive
            if (NULL != primitive) {
                dnnDelete_%(precision)s(primitive);
                primitive = NULL;
            }
        """ % locals()
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        z, = out
        alpha = self.alpha
        beta = self.beta
        size = self.n
        k = self.k

        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        fail = sub['fail']

        ccode = """
        {
            int status = 0;
            if (first_run) {
                typenum = MKLNdarray_TYPE(%(x)s);

                bottomSize[0] = MKLNdarray_DIMS(%(x)s)[3];  // w
                bottomSize[1] = MKLNdarray_DIMS(%(x)s)[2];  // h
                bottomSize[2] = MKLNdarray_DIMS(%(x)s)[1];  // c
                bottomSize[3] = MKLNdarray_DIMS(%(x)s)[0];  // n

                bottomStride[0] = 1;
                bottomStride[1] = bottomSize[0];
                bottomStride[2] = bottomSize[0] * bottomSize[1];
                bottomStride[3] = bottomSize[0] * bottomSize[1] * bottomSize[2];

                CHECK_ERR( dnnLRNCreateBackward_%(precision)s(&primitive, NULL, MKLNdarray_LAYOUT(%(gz)s), MKLNdarray_LAYOUT(%(x)s),
                                                              %(size)s, %(alpha)s, %(beta)s, %(k)s), err );
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
                if (NULL == %(z)s) {
                    %(fail)s
                }

                status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(x)s), MKLNdarray_DIMS(%(x)s));
                if (0 != status) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &primitive, dnnResourceDiffSrc);
                if (0 != status) {
                    std::cout<<"MKLNdarray_create_buffer_from_primitive failed, return: "<<status<<", line: "<<__LINE__<<std::endl;
                    exit(1);
                }
            }

            lrn_res[dnnResourceWorkspace] = MKLNdarray_WORKSPACE(%(x)s);
            lrn_res[dnnResourceDiffDst] = MKLNdarray_DATA(%(gz)s);
            lrn_res[dnnResourceDiffSrc] = MKLNdarray_DATA(%(z)s);
            lrn_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
            CHECK_ERR( dnnExecute_%(precision)s(primitive, lrn_res), err );
            first_run = 0;
        }
        """ % locals()
        return ccode

    def c_code_cache_version(self):
        return (0, 1, 1)
