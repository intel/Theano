from theano import gof, Apply
from theano.tensor import as_tensor_variable
from theano.tensor.blas import ldflags
from theano.sandbox.mkl import basic_ops
from theano.sandbox.mkl.mkl_type import MKLNdarrayType


class AbstractRelu(gof.Op):
    __props__ = ('slope',)

    def __init__(self, slope=0):
        self.slope = slope

    def make_node(self, x):
        x = as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Expect a 4D tensor, but actually got %dD tensor' %
                            x.type.ndim)
        x = as_tensor_variable(x)
        return gof.Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [AbstractReluGrad(slope=self.slope)(x, gz)]

    def perform(self, node, inp, out_):
        x, = inp
        z, = out_

        z[0] = x


class AbstractReluGrad(gof.Op):
    __props__ = ('slope',)

    def __init__(self, slope=0):
        self.slope = slope

    def make_node(self, x, gz):
        x = as_tensor_variable(x)
        if x.type.ndim != 4:
            raise TypeError('Expect a 4D tensor, but actually got %dD tensor' %
                            x.type.ndim)
        x = as_tensor_variable(x)
        return gof.Apply(self, [x, gz], [x.type()])

    def perform(self, node, inp, out_):
        x, gz = inp
        gx, = out_

        gx[0] = gz


class Relu(basic_ops.MKLOp):
    """
    Parameters
    ----------
    slope: slope for relu
    """
    __props__ = ('slope',)

    def __init__(self, slope=0):
        self.slope = slope

    def make_node(self, x):
        if not isinstance(x.type, MKLNdarrayType):
            raise TypeError('Expected MKLNdarrayType for x, '
                            'but got type %s.' % str(x.type))

        if x.type.ndim != 4:
            raise TypeError('Expected a 4 dims varialbe for x, '
                            'but got %d dims.' % x.type.ndim)

        return Apply(self, [x], [x.type()])

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [ReluGrad(slope=self.slope)(x, gz)]

    def c_support_code_struct(self, node, name):
        return """
        dnnError_t err;
        dnnPrimitive_t relu_fwd;
        void *relu_res[dnnResourceNumber];
        """

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        relu_fwd = NULL;
        """
        return ccode

    def c_cleanup_code_struct(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        ccode = """
            // release primitive
            if (NULL != relu_fwd) {
                dnnDelete_%(precision)s(relu_fwd);
                relu_fwd = NULL;
            }
        """ % locals()
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        slope = self.slope

        if node.inputs[0].type.dtype == "float32":
            precision = 'F32'
        elif node.inputs[0].type.dtype == "float64":
            precision = 'F64'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        {
            assert (MKLNdarray_Check((PyObject *)%(x)s));
            int typenum = MKLNdarray_TYPE((MKLNdarray*)%(x)s);
            int ndim = MKLNdarray_NDIM(%(x)s);
            int ret = 0;

            if (NULL == relu_fwd) {
                CHECK_ERR( dnnReLUCreateForward_%(precision)s(&relu_fwd,
                                                              NULL,
                                                              MKLNdarray_LAYOUT(%(x)s),
                                                              %(slope)s), err );
            }

            if ( !(%(z)s
                   && MKLNdarray_Check((PyObject *)%(z)s)
                   && MKLNdarray_NDIM(%(z)s) == ndim
                   && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
                   && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
                   && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
                   && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {
                if (%(z)s) Py_XDECREF(%(z)s);

                %(z)s = (MKLNdarray *)MKLNdarray_New(ndim, typenum);
                if (NULL == %(z)s) {
                    %(fail)s
                }

                ret = MKLNdarray_set_structure(%(z)s, ndim, MKLNdarray_DIMS(%(x)s));
                if (ret != 0) {
                    %(fail)s;
                }

                ret = MKLNdarray_create_buffer_from_primitive(%(z)s, &relu_fwd, dnnResourceDst);
                if (ret != 0) {
                    %(fail)s;
                }
            }

            relu_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
            relu_res[dnnResourceDst] = MKLNdarray_DATA(%(z)s);
            CHECK_ERR( dnnExecute_%(precision)s(relu_fwd, relu_res), err );
        }
        """ % sub
        return ccode

    def c_code_cache_version(self):
        return (1, 0)


class ReluGrad(basic_ops.MKLOp):
    __props__ = ('slope',)

    def __init__(self, slope=0,):
        self.slope = slope

    def c_headers(self):
        return ['<math.h>']

    def c_support_code_struct(self, node, name):
        return """
        dnnError_t err;
        dnnPrimitive_t relu_bwd;
        void *relu_res[dnnResourceNumber];
        """

    def c_init_code_struct(self, node, name, sub):
        ccode = """
        relu_bwd = NULL;
        """
        return ccode

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def make_node(self, x, gz):
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

        return Apply(self, [x, gz], [x.type()])

    def c_cleanup_code_struct(self, node, name):
        dtype = str(node.__dict__['inputs'][0].dtype)
        assert dtype in ('float32', 'float64')

        if 'float32' == dtype:
            precision = 'F32'
        else:
            precision = 'F64'

        ccode = """
            // release primitive
            if (NULL != relu_bwd) {
                dnnDelete_%(precision)s(relu_bwd);
                relu_bwd = NULL;
            }
        """ % locals()
        return ccode

    def c_code(self, node, name, inp, out, sub):
        x, gz, = inp
        gx, = out
        slope = self.slope

        if node.inputs[0].type.dtype == "float32":
            precision = 'F32'
        elif node.inputs[0].type.dtype == "float64":
            precision = 'F64'
        else:
            raise TypeError('input must be float32 or float64')

        sub = sub.copy()
        sub.update(locals())

        ccode = """
        {
            assert (MKLNdarray_Check((PyObject *)%(x)s));
            assert (MKLNdarray_Check((PyObject *)%(gz)s));
            int typenum = MKLNdarray_TYPE((MKLNdarray*)%(x)s);
            int ndim = MKLNdarray_NDIM(%(x)s);
            int ret = 0;

            if (NULL == relu_bwd) {
                CHECK_ERR( dnnReLUCreateBackward_%(precision)s(&relu_bwd,
                                                               NULL,
                                                               MKLNdarray_LAYOUT(%(gz)s),
                                                               MKLNdarray_LAYOUT(%(x)s),
                                                               %(slope)s), err );
            }

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

                ret = MKLNdarray_create_buffer_from_primitive(%(gx)s, &relu_bwd, dnnResourceDiffSrc);
                if (status != 0) {
                    %(fail)s;
                }
            }

            relu_res[dnnResourceSrc] = MKLNdarray_DATA(%(x)s);
            relu_res[dnnResourceDiffDst] = MKLNdarray_DATA(%(gz)s);
            relu_res[dnnResourceDiffSrc] = MKLNdarray_DATA(%(gx)s);

            CHECK_ERR( dnnExecute_%(precision)s(relu_bwd, relu_res), err );
        }
        """ % sub
        return ccode

    def c_code_cache_version(self):
        return (1, 0)
