from theano.gof import Apply
from theano.contrib.mkl import basic_ops, mkl_type


class ElemwiseSum(basic_ops.MKLOp):
    """
    ElemwiseSum is used to add inputs with MKL layout.

    inp_num: number of inputs
    coeff: coefficients for all inputs
    """
    __props__ = ('inp_num', 'coeff')

    def __init__(self, inp_num=1, coeff=(1.0, )):
        super(ElemwiseSum, self).__init__()
        self.inp_num = inp_num
        if isinstance(coeff, tuple):
            self.coeff = coeff
        elif isinstance(coeff, list):
            self.coeff = tuple(coeff)
        else:
            raise TypeError('Coeff should be a tuple or list.')
        if self.inp_num != len(self.coeff):
            raise ValueError('Number of ElemwiseSum inputs is not equal to \
                             number of coefficients.')

    def make_node(self, *tensors):
        for x in tensors:
            assert isinstance(x.type, mkl_type.MKLNdarrayType)
            assert x.type.ndim == 4

        def agv(v):
            return v
        return Apply(self, list(map(agv, tensors)), [x.type()])

    def infer_shape(self, node, shapes):
        return list(shapes[-1:])

    def grad(self, inp, grads):
        raise NotImplementedError('Gradient for MKL ElemwiseSum not implemented currently..')
        """
        gz, = grads
        return ElemwiseSumGrad(inp_num=self.inp_num, coeff=self.coeff)(gz, inp)
        """

    def c_code_cache_version(self):
        return (1, 0, 2)

    def c_support_code_struct(self, node, name):
        support_code = """
        dnnError_t err;
        dnnPrimitive_t pSum;
        void** pbuf_inputs;
        void *elemwise_res[dnnResourceNumber];
        dnnLayout_t* layout_internal;
        dnnPrimitive_t* convert_int2int_bottom;
        int first_run;
        """
        return support_code

    def c_init_code_struct(self, node, name, sub):
        init_code = """
        first_run = 1;
        pSum = NULL;
        pbuf_inputs = NULL;
        layout_internal = NULL;
        convert_int2int_bottom = NULL;
        """
        return init_code

    def c_cleanup_code_struct(self, node, name):
        sub = {}
        sub['len'] = len(node.inputs)
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = "float"
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise Exception("Type %s not implemented"
                            % node.inputs[1].type.dtype)

        return """
        dnnError_t err;
        if (NULL != pSum) {
            CHECK_ERR (dnnDelete_%(precision)s(pSum), err);
            pSum = NULL;
        }

        if (NULL != layout_internal) {
            for (int i = 0; i < %(len)s; i++) {
                if (NULL != layout_internal[i]) {
                    CHECK_ERR( dnnLayoutDelete_%(precision)s(layout_internal[i]), err);
                    layout_internal[i] = NULL;
                }
            }
            free(layout_internal);
            layout_internal = NULL;
        }

        if (NULL != pbuf_inputs) {
            for (int i = 0; i < %(len)s; i++) {
                if (NULL != pbuf_inputs[i]) {
                    CHECK_ERR( dnnReleaseBuffer_%(precision)s(pbuf_inputs[i]), err);
                    pbuf_inputs[i] = NULL;
                }
            }
            free(pbuf_inputs);
            pbuf_inputs = NULL;
        }

        if (NULL != convert_int2int_bottom) {
            for (int i = 0; i < %(len)s; i++) {
                if (NULL != convert_int2int_bottom[i]) {
                    CHECK_ERR( dnnDelete_%(precision)s(convert_int2int_bottom[i]), err);
                    convert_int2int_bottom[i] = NULL;
                }
            }
            free(convert_int2int_bottom);
            convert_int2int_bottom = NULL;
        }
        """ % sub

    def c_code(self, node, name, inp, out, sub):
        tensors = inp
        z, = out
        sub['z'] = z
        sub['len'] = self.inp_num
        if 'float32' == node.inputs[1].type.dtype:
            sub['type'] = "float"
            sub['precision'] = "F32"
            sub['x_item_size'] = 4
        elif "float64" == node.inputs[1].type.dtype:
            sub['type'] = 'double'
            sub['precision'] = "F64"
            sub['x_item_size'] = 8
        else:
            raise Exception("Type %s not implemented" % node.inputs[1].type.dtype)
        sub['x'] = tensors[0]
        coeff = self.coeff

        ccode = """
            %(type)s coeffs[%(len)s] = {1.0};
            """ % sub

        for i, co in enumerate(coeff):
            ccode += """
            coeffs[%s] = %s;
            """ % (i, co)

        ccode += """
            int status = 0;
            if (NULL == pSum) {
                CHECK_ERR( dnnSumCreate_%(precision)s(&pSum, NULL, %(len)s, MKLNdarray_LAYOUT(%(x)s), coeffs), err);
            }

            if (NULL == convert_int2int_bottom) {
                convert_int2int_bottom = (dnnPrimitive_t*)malloc(%(len)s * sizeof (dnnPrimitive_t));
                for (int i = 0; i < %(len)s; i++)
                    convert_int2int_bottom[i] = NULL;
            }

            if (NULL == layout_internal) {
                layout_internal = (dnnLayout_t*)malloc(%(len)s * sizeof (dnnLayout_t));
                for (int i =  0; i < %(len)s; i++)
                    layout_internal[i] = NULL;
            }

            if (NULL == pbuf_inputs) {
                pbuf_inputs = (void**)malloc(%(len)s * sizeof (void*));
                for (int i = 0; i < %(len)s; i++)
                    pbuf_inputs[i] = NULL;
            }
            """ % sub

        for i, inp in enumerate(tensors):
            d = {}
            d['i'] = i
            d['inp'] = inp
            d['precision'] = sub['precision']
            ccode += """
            if (NULL == layout_internal[%(i)s]) {
                CHECK_ERR( dnnLayoutCreateFromPrimitive_%(precision)s(&(layout_internal[%(i)s]), pSum,
                                            (dnnResourceType_t)(dnnResourceMultipleSrc + %(i)s)), err);

                //Create I2I primitive
                if (!dnnLayoutCompare_%(precision)s(MKLNdarray_LAYOUT(%(inp)s), layout_internal[%(i)s])) {
                    if (NULL == convert_int2int_bottom[%(i)s]) {
                        CHECK_ERR( dnnConversionCreate_%(precision)s(
                                            &(convert_int2int_bottom[%(i)s]),
                                            MKLNdarray_LAYOUT(%(inp)s), layout_internal[%(i)s]), err);
                    }
                    // Alloc memory for new x layout
                    if (NULL == pbuf_inputs[%(i)s]) {
                        CHECK_ERR( dnnAllocateBuffer_%(precision)s(
                                                (void**)(&(pbuf_inputs[%(i)s])),
                                                layout_internal[%(i)s]), err);
                    }
                }
            }

            if (NULL != convert_int2int_bottom[%(i)s]) {
                CHECK_ERR( dnnConversionExecute_%(precision)s(convert_int2int_bottom[%(i)s],
                                                            MKLNdarray_DATA(%(inp)s),
                                                            pbuf_inputs[%(i)s]), err);
                elemwise_res[dnnResourceMultipleSrc + %(i)s] = (void*)(pbuf_inputs[%(i)s]);
            } else {
                elemwise_res[dnnResourceMultipleSrc + %(i)s] = MKLNdarray_DATA(%(inp)s);
            }
            """ % d

        ccode += """
            if (! (%(z)s
                && MKLNdarray_Check((PyObject*)%(z)s)
                && MKLNdarray_NDIM(%(z)s) == MKLNdarray_NDIM(%(x)s)
                && MKLNdarray_DIMS(%(z)s)[0] == MKLNdarray_DIMS(%(x)s)[0]
                && MKLNdarray_DIMS(%(z)s)[1] == MKLNdarray_DIMS(%(x)s)[1]
                && MKLNdarray_DIMS(%(z)s)[2] == MKLNdarray_DIMS(%(x)s)[2]
                && MKLNdarray_DIMS(%(z)s)[3] == MKLNdarray_DIMS(%(x)s)[3] )) {
                if (%(z)s) Py_XDECREF(%(z)s);
                //create PyArrayObject
                %(z)s = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(%(x)s), MKLNdarray_TYPE(%(x)s));
                if (NULL == %(z)s) {
                    %(fail)s;
                }

                status = MKLNdarray_set_structure(%(z)s, MKLNdarray_NDIM(%(x)s), MKLNdarray_DIMS(%(x)s));
                if (status != 0) {
                    %(fail)s;
                }

                status = MKLNdarray_create_buffer_from_primitive(%(z)s, &pSum, dnnResourceDst);
                if (status != 0) {
                    std::cout<<"MKLNdarray_create_buffer_from_primitive fail, return: "<<status<<", line: "<<__LINE__<<std::endl;
                    exit(1);
                }
            }

            elemwise_res[dnnResourceDst] = MKLNdarray_DATA(%(z)s);
            CHECK_ERR( dnnExecute_%(precision)s(pSum, elemwise_res), err);
            first_run = 0;
            """ % sub
        return ccode
