import theano.tensor as T
from theano.gof import Apply, Op, Variable
from theano.tensor.blas import ldflags
from theano.contrib.mkl.mkl_helper import header_text
from theano.contrib.mkl.mkl_type import MKLNdarrayType


def as_mkl_ndarray_variable(x):
    if isinstance(x, Variable):
        while True:
            if (isinstance(x.type, MKLNdarrayType)):
                return x

            # If x is the result of a transfer, try to dig through.
            if getattr(x, 'owner', None):
                if isinstance(x.owner.op, MKLToNdarray):
                    x = x.owner.inputs[0]
                    continue
                if isinstance(x.owner.op, NdarrayToMKL):
                    x = x.owner.inputs[0]
                    continue
            break

        # If we couldn't deal with transfers, then maybe it's a tensor
        if isinstance(x.type, T.TensorType):
            return ndarray_to_mkl(x)


class MKLOp(Op):
    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_libraries(self):
        return ldflags()

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_support_code(self):
        ccode = header_text()
        ccode += """
        #define DIMENSION  4

        #define CHECK_ERR(f, err, fail_code) \\
            do { \\
                (err) = (f); \\
                if ((err) != E_SUCCESS) { \\
                    PyErr_Format(PyExc_RuntimeError, "Error in file [%s:%d], err code (%d)", \\
                           __FILE__, __LINE__, err); \\
                    (fail_code); \\
                } \\
            } while(0)
        """

        return ccode


class NdarrayToMKL(MKLOp):
    __props__ = ()

    def make_node(self, x):
        x = T.as_tensor_variable(x)
        out = MKLNdarrayType(broadcastable=x.type.broadcastable, dtype=x.dtype)()

        return Apply(self, [x], [out])

    def grad(self, inp, grads):
        gz, = grads,
        gz = as_mkl_ndarray_variable(gz)

        return [mkl_to_ndarray(gz)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out

        if node.inputs[0].type.dtype not in ['float32', 'float64']:
            raise TypeError("Type %s is not supported by %s!" %
                            node.inputs[0].type.dtype, self.__class__.__name__)

        fail = sub['fail']

        ccode = """
            int err = 0;

            if (%(z)s) {
                Py_XDECREF(%(z)s);
            }

            %(z)s = (MKLNdarray *)MKLNdarray_New(PyArray_NDIM(%(x)s), PyArray_TYPE(%(x)s));
            if (!%(z)s) {
                %(fail)s
            }

            err = MKLNdarray_ViewFromArray(%(z)s, %(x)s);
            if (err != 0) {
                %(fail)s
            }
        """ % locals()

        return ccode

    def c_code_cache_version(self):
        return (1, 0)


class MKLToNdarray(MKLOp):
    __props__ = ()

    def make_node(self, x):
        x = as_mkl_ndarray_variable(x)
        out = T.TensorType(broadcastable=x.broadcastable, dtype=x.dtype)()

        return Apply(self, [x], [out])

    def grad(self, inp, grads):
        gz, = grads
        gz = T.as_tensor_variable(gz)

        return [ndarray_to_mkl(gz)]

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

    def c_code_cache_version(self):
        return (1, 0)


# alias
mkl_to_ndarray = MKLToNdarray()
ndarray_to_mkl = NdarrayToMKL()
