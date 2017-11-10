from theano import tensor, gof
import numpy
import theano
from theano.tensor.blas import ldflags


class ElementwiseAdd(gof.Op):
    __props__ = ()

    def make_node(self, *inputs):
        # Validate the inputs' type
        inputs = list(map(tensor.as_tensor_variable, inputs))

        if len(inputs) != 2:
            raise TypeError(
                'ElementwiseAdd: 2 arguments required, %d given' % len(inputs))

        if inputs[0].ndim > 5:
            raise TypeError("ElementwiseAdd: A's dimension is to large")
        if inputs[1].ndim > 5:
            raise TypeError("ElementwiseAdd: B's dimension is to large")

        output_var = theano.tensor.TensorType(
                dtype=theano.scalar.upcast(inputs[0].dtype, inputs[1].dtype),
                broadcastable=[False])()

        return gof.Apply(self, inputs, [output_var])

    def c_libraries(self):
        return ldflags()

    def c_headers(self):
        headers = ['<mkl.h>']
        return headers

    def c_code(self, node, name, inputs, outputs, sub):
        x, y = inputs
        z, = outputs

        convert_x = 0
        convert_y = 0

        if node.inputs[0].dtype is 'float64' or node.inputs[1].dtype is 'float64':
            c_type = 'double'
            ccode_addfunc = 'vdAdd'
            if node.inputs[0].dtype is 'float32':
                convert_x = 1
            elif node.inputs[1].dtype is 'float32':
                convert_y = 1
        elif node.inputs[0].dtype is 'float32' and node.inputs[1].dtype is 'float32':
            c_type = 'float'
            ccode_addfunc = 'vsAdd'
        else:
            raise TypeError('ElementwiseAdd: dtype %s is not supported.'
                            % (node.inputs[0].dtype))

        dim_x = node.inputs[0].ndim
        dim_y = node.inputs[1].ndim
        if dim_x > dim_y:
            dim_max = dim_x
            dim_min = dim_y
            max_input = x
            min_input = y
        elif dim_x < dim_y:
            dim_max = dim_y
            dim_min = dim_x
            max_input = y
            min_input = x
        else:
            dim_max = dim_x
            dim_min = dim_x
            max_input = x
            min_input = x

        dtype_x = node.inputs[0].dtype
        dtype_y = node.inputs[1].dtype
        dtype_z = node.outputs[0].dtype

        typenum_z = numpy.dtype(dtype_z).num

        fail = sub['fail']

        c_code = """
        // Validate that the output storage exists.
        if (NULL == %(z)s)
        {
            /* Reference received to invalid output variable.
            Decrease received reference's ref count and allocate new
            output variable */
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*)PyArray_EMPTY(%(dim_max)s,
                                                PyArray_DIMS(%(max_input)s),
                                                %(typenum_z)s,
                                                0);

            if (!%(z)s) {
                %(fail)s;
            }
        }

        //Covert type if two inputs have different types
        if (%(convert_x)s == 1){
            %(x)s = (PyArrayObject*)PyArray_Cast(%(x)s ,PyArray_TYPE(%(y)s));
        }else if (%(convert_y)s == 1){
            %(y)s = (PyArrayObject*)PyArray_Cast(%(y)s ,PyArray_TYPE(%(x)s));
        }

        //Validate that two inputs have same shape
        //or same part of shape with diff dimensions
        if (%(dim_x)s == %(dim_y)s){
            for (int i = 0; i < %(dim_x)s; i++){
                if (PyArray_DIMS(%(x)s)[i] != PyArray_DIMS(%(y)s)[i]){
                    PyErr_Format(PyExc_ValueError, "Shape mismatch : "
                        "x.shape and y.shape should match but "
                        "x.shape[%%d] == %%i and y.shape[%%d] == %%i",
                        i, PyArray_DIMS(%(x)s)[0],
                        i, PyArray_DIMS(%(y)s)[0]);
                    %(fail)s;
                }
            }
        }else {
            for (int i = 0; i < %(dim_min)s; i++){
                if (PyArray_DIMS(%(min_input)s)[i] != PyArray_DIMS
                (%(max_input)s)[i + %(dim_max)s - %(dim_min)s]){
                    PyErr_Format(PyExc_ValueError, "Shape mismatch : "
                        "x.shape and y.shape should match but "
                        "x.shape[%%d] == %%i and y.shape[%%d] == %%i",
                        i, PyArray_DIMS(%(min_input)s)[i],
                        i + %(dim_min)s,
                        PyArray_DIMS(%(max_input)s)
                        [i + %(dim_max)s - %(dim_min)s]);
                    %(fail)s;
                }
            }
        }

        //Computation
        if (%(dim_x)s == %(dim_y)s){
            //Get the number of elements from PyArray_SIZE
            int cal_num = PyArray_SIZE(%(x)s);

            //Perform the MKL Add (elementwise add)
            %(c_type)s *px = (%(c_type)s*)PyArray_DATA(%(x)s);
            %(c_type)s *py = (%(c_type)s*)PyArray_DATA(%(y)s);
            %(c_type)s *pz = (%(c_type)s*)PyArray_DATA(%(z)s);

            %(ccode_addfunc)s(cal_num, px, py, pz);

        }else{
            //calculate the computation times
            int loop_num = 1;
            for (int i = 0; i < %(dim_max)s - %(dim_min)s; i++){
                loop_num *= PyArray_DIMS(%(max_input)s)[i];
            }

            //calculate the number of elements to be computed each time
            int cal_num = 1;
            for (int i = 0; i < %(dim_min)s; i++){
                cal_num *= PyArray_DIMS(%(min_input)s)[i];
            }

            //Perform the MKL Add (elementwise add)
            %(c_type)s *pmax = (%(c_type)s*)PyArray_DATA(%(max_input)s);
            %(c_type)s *pmin = (%(c_type)s*)PyArray_DATA(%(min_input)s);
            %(c_type)s *pz = (%(c_type)s*)PyArray_DATA(%(z)s);

            for (int i = 0; i < loop_num; i++){
                %(ccode_addfunc)s(cal_num, pmax, pmin, pz);
                pmax += cal_num;
                pz += cal_num;
            }
        }
        """

        return c_code % locals()

    def grad(self, inp, grads):
        x, y = inp
        gz, = grads
        return gz, gz

    def c_code_cache_version(self):
        return (1, 0, 0)
