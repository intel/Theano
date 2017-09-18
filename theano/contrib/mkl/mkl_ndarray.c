#include <Python.h>
#include <structmember.h>

#include "mkl_ndarray.h"


static int
MKLNdarray_uninit(MKLNdarray *self) {
    int rval = 0;

    if (NULL != self->base) {
        Py_DECREF(self->base);
        // if the object is reference to a PyArrayObject
        // should release the layout additionally
        if (self->flag & MNDA_VIEW_FROM_NP) {
            if (self->layout) {
                if (MNDA_FLOAT64 == self->dtype) {
                    rval = dnnLayoutDelete_F64(self->layout);
                } else {
                    rval = dnnLayoutDelete_F32(self->layout);
                }
                if (E_SUCCESS != rval) {
                        PyErr_Format(PyExc_RuntimeError,
                                     "MKLNdarray_uninit: fail to release layout: %d, "
                                     "line: %d, dtype: %d",
                                     rval, __LINE__, self->dtype);
                }
                self->layout = NULL;
            }
        }

        self->base      = NULL;
        self->flag      = 0;
        self->data_size = 0;
        self->nd        = -1;
        self->dtype     = -1;

    } else {
        if (self->data) {
            if (MNDA_FLOAT64 == self->dtype) {
                rval = dnnReleaseBuffer_F64(self->data);
            } else {
                rval = dnnReleaseBuffer_F32(self->data);
            }
            if (E_SUCCESS != rval) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release data: %d, "
                             "line: %d, dtype: %d",
                             rval, __LINE__, self->dtype);
            }
            self->data = NULL;
        }

        if (self->layout) {
            if (MNDA_FLOAT64 == self->dtype) {
                rval = dnnLayoutDelete_F64(self->layout);
            } else {
                rval = dnnLayoutDelete_F32(self->layout);
            }
            if (E_SUCCESS != rval) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_uninit: fail to release layout: %d, "
                             "line: %d, dtype: %d",
                             rval, __LINE__, self->dtype);
            }
            self->layout = NULL;
        }

        self->flag      = 0;
        self->data_size = 0;
        self->nd        = -1;
        self->dtype     = -1;
    }
    return rval;
}


static void
MKLNdarray_dealloc(MKLNdarray *self) {
    if (Py_REFCNT(self) > 1) {
        printf("WARNING: MKLNdarray_dealloc called when there is still active reference to it.\n");
    }
    if (NULL != self) {
        MKLNdarray_uninit(self);
        Py_TYPE(self)->tp_free((PyObject*)self);
    }
}


static PyObject*
MKLNdarray_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    MKLNdarray* self = NULL;
    self = (MKLNdarray*)(type->tp_alloc(type, 0));

    if (NULL != self) {
        self->base      = NULL;
        self->flag      = 0;
        self->nd        = -1;
        self->dtype     = -1;
        self->data      = NULL;
        self->layout    = NULL;
        self->data_size = 0;
        memset((void*)(self->user_structure), 0, 2 * MNDA_MAX_NDIM * sizeof (size_t));
    } else {
        PyErr_SetString(PyExc_MemoryError,
                        "MKLNdarray_new: fail to create a new instance");
        return NULL;
    }
    return (PyObject*)self;
}


static int
MKLNdarray_init(MKLNdarray *self, PyObject *args, PyObject *kwds) {
    PyObject* arr = NULL;

    if (!PyArg_ParseTuple(args, "O", &arr))
        return -1;

    if (!PyArray_Check(arr)) {
        PyErr_SetString(PyExc_TypeError,
                        "MKLNdarray_init: PyArrayObject arg required");
        return -1;
    }

    // do type conversion here. PyArrayObject -> MKLNdarray
    int rval = -1;
    if (NULL != self) {
        rval = MKLNdarray_CopyFromArray(self, (PyArrayObject*)arr);
        return rval;
    } else {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_init: input MKLNdarray* self is NULL");
        return -1;
    }
}


PyObject*
MKLNdarray_repr(PyObject *self) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_repr: input PyObject* self is NULL");
        return NULL;
    }

    MKLNdarray* object = (MKLNdarray*)self;
    char cstr[64]; // 64 chars is enough for a string.
    sprintf(cstr, "ndim=%d, dtype=%s", object->nd, MNDA_TYPE[object->dtype]);
    PyObject* out = PyString_FromFormat("%s%s%s", "MKLNdarray(", cstr, ")");

#if PY_MAJOR_VERSION >= 3
    PyObject* out2 = PyObject_Str(out);
    Py_DECREF(out);
    return out2;
#else
    return out;
#endif
}


const size_t*
MKLNdarray_DIMS(const MKLNdarray *self) {
    if (NULL != self) {
        return self->user_structure;
    } else {
        return NULL;
    }
}


const size_t*
MKLNdarray_STRIDES(const MKLNdarray *self) {
    if (NULL != self) {
        return self->user_structure + self->nd;
    } else {
        return NULL;
    }
}


int MKLNdarray_NDIM(const MKLNdarray *self) {
    if (NULL != self) {
        return self->nd;
    } else {
        return -1;
    }
}


int MKLNdarray_TYPE(const MKLNdarray *self) {
    if (NULL != self) {
        return self->dtype;
    } else {
        return -1;
    }
}


void*
MKLNdarray_DATA(const MKLNdarray *self) {
    if (NULL != self) {
        return self->data;
    } else {
        return NULL;
    }
}

dnnLayout_t
MKLNdarray_LAYOUT(const MKLNdarray *self) {
    if (NULL != self) {
        return self->layout;
    } else {
        return NULL;
    }
}


int MKLNdarray_create_buffer_from_primitive(MKLNdarray           *self,
                                            const dnnPrimitive_t *prim,
                                            dnnResourceType_t    res_type) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: "
                        "input MKLNdarray* self is NULL");
        return -1;
    }

    if (self->nd < 0 ||
        (MNDA_FLOAT32 != self->dtype && MNDA_FLOAT64 != self->dtype) ) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: "
                        "Can't create layout and buffer for an uninitialized MKLNdarray");
        return -1;
    }

    if (NULL == prim) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: "
                        "Can't create layout and buffer with an empty primtive");
        return -1;
    }

    if (dnnResourceWorkspace == res_type) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: "
                        "Please refer to CDataType for workspace buffer");
        return -1;
    }

    if (self->layout || self->data) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_primitive: "
                        "Can't create layout or buffer for MKLNdarray repeatly");
        return -1;
    }

    int status = 0;
    if (MNDA_FLOAT64 == self->dtype) {  // for float64
        status = dnnLayoutCreateFromPrimitive_F64(&(self->layout), *prim, res_type);
        if (E_SUCCESS != status || NULL == self->layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_primitive: "
                         "Create layout failed: %d, line: %d",
                         status, __LINE__);
            return -1;
        }

        status = dnnAllocateBuffer_F64(&(self->data), self->layout);
        if (E_SUCCESS != status || NULL == self->data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_primitive: "
                         "Create private data failed: %d, line: %d",
                         status, __LINE__);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F64(self->layout);
    } else {  // for float32
        status = dnnLayoutCreateFromPrimitive_F32(&(self->layout), *prim, res_type);
        if (E_SUCCESS != status || NULL == self->layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_primitive: "
                         "Create layout failed: %d, line: %d",
                         status, __LINE__);
            return -1;
        }

        status = dnnAllocateBuffer_F32(&(self->data), self->layout);
        if (E_SUCCESS != status || NULL == self->data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_primitive: "
                         "Create private data failed: %d, line: %d",
                         status, __LINE__);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F32(self->layout);
    }

    return 0;
}


int MKLNdarray_create_buffer_from_structure(MKLNdarray *self) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_structure: "
                        "input MKLNdarray* self is NULL");
        return -1;
    }
    if (self->nd <= 0) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_create_buffer_from_structure: "
                     "Can't create mkl layout and allocate buffer "
                     "for a %d dimension MKLNdarray",
                     self->nd);
        return -1;
    }

    size_t ndim = self->nd;

    if (self->layout || self->data) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_create_buffer_from_structure: "
                     "MKL buffer has been allocated for %p \n", self);
        return -1;
    }

    size_t mkl_size[MNDA_MAX_NDIM] = {0};
    size_t mkl_stride[MNDA_MAX_NDIM] = {0};

    // nchw -> whcn
    for (int i = 0; i < self->nd; i++) {
        mkl_size[i] = (MKLNdarray_DIMS(self))[self->nd - i - 1];
        mkl_stride[i] = (MKLNdarray_STRIDES(self))[self->nd - i -1];
    }

    int status = 0;
    // float64
    if (MNDA_FLOAT64 == self->dtype) {
        status = dnnLayoutCreate_F64(&(self->layout), ndim, mkl_size, mkl_stride);
        if (E_SUCCESS != status || NULL == self->layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: "
                         "Call dnnLayoutCreate_F64 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F64(&(self->data), self->layout);
        if (E_SUCCESS != status || NULL == self->data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: "
                         "Call dnnAllocateBuffer_F64 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F64(self->layout);

    } else {  // float32
        status = dnnLayoutCreate_F32(&(self->layout), ndim, mkl_size, mkl_stride);
        if (E_SUCCESS != status || NULL == self->layout) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: "
                         "Call dnnLayoutCreate_F32 failed: %d",
                         status);
            return -1;
        }

        status = dnnAllocateBuffer_F32(&(self->data), self->layout);
        if (E_SUCCESS != status || NULL == self->data) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_create_buffer_from_structure: "
                         "Call dnnAllocateBuffer_F32 failed: %d",
                         status);
            return -1;
        }
        self->data_size = dnnLayoutGetMemorySize_F32(self->layout);
    }

    return 0;
}


int MKLNdarray_create_buffer_from_layout(MKLNdarray *self) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_layout: "
                        "input MKLNdarray* self is NULL");
        return -1;
    }

    if (NULL == self->layout) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_layout: "
                        "layout is NULL");
        return -1;
    }

    if (self->data) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_buffer_from_layout: "
                        "MKL buffer is already allocated");
        return -1;
    }

    int status = 0;
    if (MNDA_FLOAT64 == self->dtype) {
        status = dnnAllocateBuffer_F64(&(self->data), self->layout);
        self->data_size = dnnLayoutGetMemorySize_F64(self->layout);
    } else {  // float32
        status = dnnAllocateBuffer_F32(&(self->data), self->layout);
        self->data_size = dnnLayoutGetMemorySize_F32(self->layout);
    }

    if (E_SUCCESS != status || NULL == self->data) {
        self->data_size = 0;
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_create_buffer_from_layout: "
                     "Call dnnAllocateBuffer failed: %d, dtype: %d",
                     status, self->dtype);
        return -1;
    }
    return 0;
}


int MKLNdarray_copy_layout(MKLNdarray *dst, const MKLNdarray *src) {
    if (NULL == dst || NULL == src) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_copy_layout: "
                        "input source or destination is NULL");
        return -1;
    }

    if (dst == src) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_copy_layout: "
                        "source is same with destination");
        return -1;
    }

    if (NULL == src->layout) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_copy_layout: "
                        "layout of source object is NULL");
        return -1;
    }

    if (dst->layout) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_copy_layout: "
                        "destination layout is already existed");
        return -1;
    }

    assert (src->nd == dst->nd);
    assert (src->dtype == dst->dtype);

    int status = 0;
    void* layout_buf = NULL;
    if (MNDA_FLOAT64 == src->dtype) {
        layout_buf = (void*)mkl_malloc(dnnLayoutSerializationBufferSize_F64(), 64);
        if (NULL == layout_buf) {
            PyErr_SetString(PyExc_MemoryError,
                            "MKLNdarray_copy_layout: "
                            "allocate buffer for layout failed");
            return -1;
        }

        status = dnnLayoutSerialize_F64(src->layout, layout_buf);
        if (E_SUCCESS != status) {
            PyErr_SetString(PyExc_RuntimeError,
                            "MKLNdarray_copy_layout: "
                            "serialize layout failed");
            if (layout_buf) {
                mkl_free (layout_buf);
                layout_buf = NULL;
            }
            return -1;
        }

        status = dnnLayoutDeserialize_F64(&(dst->layout), layout_buf);
        if (E_SUCCESS != status) {
            PyErr_SetString(PyExc_RuntimeError,
                            "MKLNdarray_copy_layout: "
                            "deserialize layout failed");
            if (layout_buf) {
                mkl_free (layout_buf);
                layout_buf = NULL;
            }
            return -1;
        }
    } else {  // MNDA_FLOAT32
        layout_buf = (void*)mkl_malloc(dnnLayoutSerializationBufferSize_F32(), 64);
        if (NULL == layout_buf) {
            PyErr_SetString(PyExc_MemoryError,
                            "MKLNdarray_copy_layout: "
                            "allocate buffer for layout failed");
            return -1;
        }

        status = dnnLayoutSerialize_F32(src->layout, layout_buf);
        if (E_SUCCESS != status) {
            PyErr_SetString(PyExc_RuntimeError,
                            "MKLNdarray_copy_layout: "
                            "serialize layout failed");
            if (layout_buf) {
                mkl_free (layout_buf);
                layout_buf = NULL;
            }
            return -1;
        }

        status = dnnLayoutDeserialize_F32(&(dst->layout), layout_buf);
        if (E_SUCCESS != status) {
            PyErr_SetString(PyExc_RuntimeError,
                            "MKLNdarray_copy_layout: "
                            "deserialize layout failed");
            if (layout_buf) {
                mkl_free (layout_buf);
                layout_buf = NULL;
            }
            return -1;
        }
    }

    if (layout_buf) {
        mkl_free (layout_buf);
        layout_buf = NULL;
    }
    return 0;
}


int MKLNdarray_set_structure(MKLNdarray *self,
                             int nd,
                             const size_t *dims,
                             const size_t *strides) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_set_structure: input MKLNdarray* self is NULL");
        return -1;
    }

    if (NULL == dims) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_set_structure: input dims is NULL");
        return -1;
    }

    if (nd > MNDA_MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray does not support a %d-dim array. "
                     "Try array which ndim is <= %d",
                     nd, MNDA_MAX_NDIM);
        return -1;
    }

    // if argument strides is NULL, strides in user_structure will be computed 
    // from argument dims.
    if (NULL == strides) {
        self->user_structure[0] = dims[0];
        self->user_structure[2 * nd - 1] = 1;
        for (int i = 1; i < nd; i++) {
            self->user_structure[i] = dims[i];
            self->user_structure[2 * nd - 1 - i] = self->user_structure[2 * nd - i] * dims[nd - i];
        }
    } else {
        for (int i = 0; i < nd; i++) {
            self->user_structure[i] = dims[i];
            self->user_structure[nd + i] = strides[i];
        }
    }

    return 0;
}


int MKLNdarray_CopyFromArray(MKLNdarray *self, PyArrayObject *obj) {
    if (NULL == self || NULL == obj) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CopyFromArray: "
                        "input self or obj is NULL");
        return -1;
    }

    if (self->base || self->layout || self->data) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_CopyFromArray: "
                     "MKLNdarray base, layout or buffer have been allocated for %p", self);
        return -1;
    }

    int ndim    = PyArray_NDIM(obj);
    npy_intp* d = PyArray_DIMS(obj);
    int typenum = PyArray_TYPE(obj);

    if (NPY_FLOAT32 != typenum && NPY_FLOAT64 != typenum) {
        PyErr_SetString(PyExc_TypeError,
                        "MKLNdarray_CopyFromArray: "
                        "can only copy from float/double arrays");
        return -1;
    }

    if (ndim < 0 || ndim > MNDA_MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray_CopyFromArray: "
                     "does not support a %d-dim array. "
                     "Try array which ndim is <= %d",
                     ndim, MNDA_MAX_NDIM);
        return -1;
    }

    self->dtype = typenum;
    self->nd    = ndim;
    self->flag  = 0;

    PyArrayObject* py_src = (PyArrayObject*)PyArray_ContiguousFromAny((PyObject*)obj,
                                                                      typenum,
                                                                      self->nd,
                                                                      self->nd);
    if (!py_src) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CopyFromArray: "
                        "fail to cast obj to contiguous array");
        return -1;
    }

    size_t dims[MNDA_MAX_NDIM] = {0};
    size_t user_size = 1;

    for (int i = 0; i < ndim; i++) {
        dims[i] = (size_t)d[i];
        user_size *= dims[i];
    }

    int err = MKLNdarray_set_structure(self, ndim, dims, NULL);
    if (err < 0) {
        Py_DECREF(py_src);
        return err;
    }

    // prepare user layout and mkl buffer
    err = MKLNdarray_create_buffer_from_structure(self);
    if (err < 0) {
        Py_DECREF(py_src);
        return err;
    }

    // copy data to mkl buffer
    size_t element_size = (size_t)PyArray_ITEMSIZE(py_src);
    assert (user_size * element_size <= self->data_size);
    memcpy((void*)self->data, (void*)PyArray_DATA(py_src), user_size * element_size);
    Py_DECREF(py_src);
    return 0;
}


int MKLNdarray_ViewFromArray(MKLNdarray *self, PyArrayObject *obj) {
    if (NULL == self || NULL == obj) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_ViewFromArray: "
                        "input self or obj is NULL");
        return -1;
    }

    if (self->base || self->layout || self->data) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_ViewFromArray: "
                     "MKLNdarray base, layout or buffer have been allocated for %p", self);
        return -1;
    }

    int ndim    = PyArray_NDIM(obj);    // dimension #
    npy_intp* d = PyArray_DIMS(obj);    // dims
    npy_intp* s = PyArray_STRIDES(obj); // strides
    int typenum = PyArray_TYPE(obj);    // type

    if (NPY_FLOAT32 != typenum && NPY_FLOAT64 != typenum) {
        PyErr_SetString(PyExc_TypeError,
                        "MKLNdarray_ViewFromArray: "
                        "can only view from float/double arrays");
        return -1;
    }

    if (ndim < 0 || ndim > MNDA_MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray_ViewFromArray: "
                     "does not support a %d-dim array. "
                     "Try array which ndim is <= %d",
                     ndim, MNDA_MAX_NDIM);
        return -1;
    }

    self->dtype = typenum;
    self->nd    = ndim;

    size_t dims[MNDA_MAX_NDIM] = {0};
    size_t strides[MNDA_MAX_NDIM] = {0};
    size_t user_size = 1;

    for (int i = 0; i < ndim; i++) {
        dims[i] = (size_t)d[i];
        strides[i] = (size_t)s[i] / (typenum == MNDA_FLOAT64 ? 8 : 4);
        user_size *= dims[i];
    }

    int err = MKLNdarray_set_structure(self, ndim, dims, strides);
    if (err < 0) {
        return err;
    }
   
    size_t mkl_size[MNDA_MAX_NDIM] = {0};
    size_t mkl_stride[MNDA_MAX_NDIM] = {0};
    for (int i = 0; i < self->nd; i++) {
        mkl_size[i] = (MKLNdarray_DIMS(self))[self->nd - i - 1];
        mkl_stride[i] = (MKLNdarray_STRIDES(self))[self->nd - i -1];
    }

    // prepare user layout
    int status = 0;
    if (MNDA_FLOAT64 == self->dtype) {
        status = dnnLayoutCreate_F64(&(self->layout), ndim, mkl_size, mkl_stride);
    } else {  // float32
        status = dnnLayoutCreate_F32(&(self->layout), ndim, mkl_size, mkl_stride);
    }

    if (E_SUCCESS != status || NULL == self->layout) {
        PyErr_Format(PyExc_RuntimeError,
                     "MKLNdarray_ViewFromArray: Call dnnLayoutCreate failed: %d, "
                     "line: %d, dtype: %d",
                     status, __LINE__, self->dtype);
        return -1;
    }

    PyObject *orig_base = (PyObject*)obj;
    while (orig_base &&
            PyArray_Check(orig_base) &&
            PyArray_BASE((PyArrayObject*)orig_base)) {
        orig_base = PyArray_BASE((PyArrayObject*)orig_base);
    }

    self->base  = orig_base;
    self->data  = (void*)PyArray_DATA((PyArrayObject*)orig_base);
    self->flag  |= MNDA_VIEW_FROM_NP;

    Py_INCREF(orig_base);
    return 0;
}


PyObject* MKLNdarray_create_with_zeros(int n, const size_t *dims, int typenum) {
    size_t total_elements = 1;
    if (n < 0 || n > MNDA_MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray_create_with_zeros: "
                     "does not support a %d-dim array. "
                     "Try array which ndim is <= %d",
                     n, MNDA_MAX_NDIM);
        return NULL;
    }

    if (NULL == dims) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_with_zeros: "
                        "input dims is NULL");
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        if (dims[i] != 0 && total_elements > (SIZE_MAX / dims[i])) {
            PyErr_Format(PyExc_RuntimeError,
                         "Can't store in size_t for the bytes requested %llu * %llu",
                         (unsigned long long)total_elements,
                         (unsigned long long)dims[i]);
            return NULL;
        }
        total_elements *= dims[i];
    }

    // total_elements now contains the size of the array
    size_t max = 0;
    if (MNDA_FLOAT64 == typenum)
        max = SIZE_MAX / sizeof (double);
    else
        max = SIZE_MAX / sizeof (float);

    if (total_elements > max) {
        PyErr_Format(PyExc_RuntimeError,
                     "Can't store in size_t for the bytes requested %llu",
                     (unsigned long long)total_elements);
        return NULL;
    }

    size_t total_size = 0;
    if (MNDA_FLOAT64 == typenum)
        total_size = total_elements * sizeof (double);
    else
        total_size = total_elements * sizeof (float);

    MKLNdarray* rval = (MKLNdarray*)MKLNdarray_New(n, typenum);
    if (!rval) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_with_zeros: "
                        "fail to create new instance");
        return NULL;
    }

    if (MKLNdarray_set_structure(rval, n, dims, NULL)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_create_with_zeros: "
                        "syncing structure to mkl failed.");
        Py_DECREF(rval);
        return NULL;
    }

    if (MKLNdarray_create_buffer_from_structure(rval)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarrya_create_with_zeros: "
                        "create buffer from structure failed.");
        Py_DECREF(rval);
        return NULL;
    }
    // Fill with zeros
    memset(rval->data, 0, rval->data_size);
    return (PyObject*)rval;
}


static PyObject*
MKLNdarray_get_shape(MKLNdarray *self, void *closure) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_shape: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_shape: "
                        "MKLNdarray is not initialized");
        return NULL;
    }

    PyObject* rval = PyTuple_New(self->nd);
    if (NULL == rval) {
        return NULL;
    }

    for (int i = 0; i < self->nd; i++) {
        if (PyTuple_SetItem(rval, i, PyInt_FromLong(MKLNdarray_DIMS(self)[i]))) {
            Py_XDECREF(rval);
            return NULL;
        }
    }

    return rval;
}


static PyObject*
MKLNdarray_get_strides(MKLNdarray *self, void *closure) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_strides: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_strides: "
                        "MKLNdarray is not initialized");
        return NULL;
    }

    PyObject* rval = PyTuple_New(self->nd);
    if (NULL == rval) {
        return NULL;
    }

    for (int i = 0; i < self->nd; i++) {
        if (PyTuple_SetItem(rval, i, PyInt_FromLong(MKLNdarray_STRIDES(self)[i]))) {
            Py_XDECREF(rval);
            return NULL;
        }
    }

    return rval;
}


static PyObject*
MKLNdarray_get_dtype(MKLNdarray *self, void *closure) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_dtype: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd < 0 || self->dtype < 0) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_dtype: "
                        "MKLNdarray is not initialized");
        return NULL;
    }

    PyObject * rval = PyString_FromFormat("%s", MNDA_TYPE[self->dtype]);
    return rval;
}


static PyObject*
MKLNdarray_get_ndim(MKLNdarray *self, void *closure) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_ndim: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    } else {
        return PyInt_FromLong(self->nd);
    }
}


static PyObject*
MKLNdarray_get_size(MKLNdarray *self, void *closure) {
    size_t total_element = 1;
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_size: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd <= 0) {
        total_element = 0;
    } else {
        for (int i = 0; i < self->nd; i++) {
            total_element *= MKLNdarray_DIMS(self)[i];
        }
    }
    return PyInt_FromLong(total_element);
}


static PyObject*
MKLNdarray_get_base(MKLNdarray *self, void *closure) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_base: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    }

    PyObject * base = self->base;
    if (!base) {
        base = Py_None;
    }

    Py_INCREF(base);
    return base;
}


PyObject* MKLNdarray_CreateArrayObj(const MKLNdarray *self) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CreateArrayObj: "
                        "input MKLNdarray* self is NULL");
        return NULL;
    }

    if (self->nd < 0 ||
        NULL == self->layout ||
        NULL == self->data) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_CreateArrayObj: "
                        "Can't convert from an uninitialized MKLNdarray");
        return NULL;
    }

    npy_intp npydims[MNDA_MAX_NDIM] = {0};
    for (int i = 0; i < self->nd; i++) {
        npydims[i] = (npy_intp)(MKLNdarray_DIMS(self)[i]);
    }

    PyArrayObject* rval = NULL;
    if (MNDA_FLOAT64 == self->dtype) {
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT64);
    } else {
        rval = (PyArrayObject*)PyArray_SimpleNew(self->nd, npydims, NPY_FLOAT32);
    }

    if (!rval) {
        return NULL;
    }

    void* rval_data = PyArray_DATA(rval);
    int status = -1;
    dnnPrimitive_t primitive = NULL;
    dnnLayout_t layout_user = NULL;

    size_t mkl_size[MNDA_MAX_NDIM] = {0};
    size_t mkl_stride[MNDA_MAX_NDIM] = {0};

    // nchw -> whcn
    for (int i = 0; i < self->nd; i++) {
        mkl_size[i] = (MKLNdarray_DIMS(self))[self->nd - i - 1];
        mkl_stride[i] = (MKLNdarray_STRIDES(self))[self->nd - i - 1];
    }

    if (MNDA_FLOAT64 == self->dtype) { // float64
        status = dnnLayoutCreate_F64(&layout_user,
                                     self->nd,
                                     mkl_size,
                                     mkl_stride);

        if (E_SUCCESS != status || NULL == layout_user) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_CreateArrayObj: "
                         "dnnLayoutCreate_F64 failed: %d, line: %d",
                         status, __LINE__);
            Py_DECREF(rval);
            return NULL;
        }

        if (!dnnLayoutCompare_F64(self->layout, layout_user)) {
            status = dnnConversionCreate_F64(&primitive, self->layout, layout_user);
            if (E_SUCCESS != status || NULL == primitive) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: "
                             "dnnConversionCreate_F64 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                if (NULL != layout_user) {
                    dnnLayoutDelete_F64(layout_user);
                    layout_user = NULL;
                }
                return NULL;
            }

            status = dnnConversionExecute_F64(primitive, (void*)self->data, (void*)rval_data);
            if (E_SUCCESS != status) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: "
                             "dnnConversionExecute_F64 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                if (NULL != layout_user) {
                    dnnLayoutDelete_F64(layout_user);
                    layout_user = NULL;
                }
                
                if (NULL != primitive) {
                    dnnDelete_F64(primitive);
                    primitive = NULL;
                }
                return NULL;
            }
        } else {
            memcpy((void*)rval_data, (void*)self->data, PyArray_SIZE(rval) * sizeof (double)); 
        }

        if (NULL != layout_user) {
            dnnLayoutDelete_F64(layout_user);
            layout_user = NULL;
        }

        if (NULL != primitive) {
            dnnDelete_F64(primitive);
            primitive = NULL;
        }

    } else {  // float32
        status = dnnLayoutCreate_F32(&layout_user,
                                     self->nd,
                                     mkl_size,
                                     mkl_stride);

        if (E_SUCCESS != status || NULL == layout_user) {
            PyErr_Format(PyExc_RuntimeError,
                         "MKLNdarray_CreateArrayObj: "
                         "dnnLayoutCreate_F32 failed: %d, line: %d",
                         status, __LINE__);
            Py_DECREF(rval);
            return NULL;
        }

        if (!dnnLayoutCompare_F32(self->layout, layout_user)) {
            status = dnnConversionCreate_F32(&primitive, self->layout, layout_user);
            if (E_SUCCESS != status || NULL == primitive) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: "
                             "dnnConversionCreate_F32 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                if (NULL != layout_user) {
                    dnnLayoutDelete_F32(layout_user);
                    layout_user = NULL;
                }
                return NULL;
            }

            status = dnnConversionExecute_F32(primitive, (void*)self->data, (void*)rval_data);
            if (E_SUCCESS != status) {
                PyErr_Format(PyExc_RuntimeError,
                             "MKLNdarray_CreateArrayObj: "
                             "dnnConversionExecute_F32 failed: %d, line: %d",
                             status, __LINE__);
                Py_DECREF(rval);
                if (NULL != layout_user) {
                    dnnLayoutDelete_F32(layout_user);
                    layout_user = NULL;
                }

                if (NULL != primitive) {
                    dnnDelete_F32(primitive);
                    primitive = NULL;
                }
                return NULL;
            }
        } else {
            memcpy((void*)rval_data, (void*)self->data, PyArray_SIZE(rval) * sizeof (float));
        }

        if (NULL != layout_user) {
            dnnLayoutDelete_F32(layout_user);
            layout_user = NULL;
        }

        if (NULL != primitive) {
            dnnDelete_F32(primitive);
            primitive = NULL;
        }
    }

    return (PyObject*)rval;
}


PyObject* MKLNdarray_Zeros(PyObject *_unused, PyObject *args) {
    if (!args) {
        PyErr_SetString(PyExc_TypeError,
                        "MKLNdarray_Zeros: "
                        "function takes at least 1 argument");
        return NULL;
    }

    PyObject* shape = NULL;
    int typenum = -1;

    if (!PyArg_ParseTuple(args, "Oi", &shape, &typenum)) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_Zeros: PyArg_ParseTuple failed \n");
        return NULL;
    }

    if ((MNDA_FLOAT32 != typenum) && (MNDA_FLOAT64 != typenum)) {
        typenum = MNDA_FLOAT32;
    }

    if (!PySequence_Check(shape)) {
        PyErr_SetString(PyExc_TypeError,
                        "shape argument must be a sequence");
        return NULL;
    }
    
    int shplen = PySequence_Length(shape);
    if (shplen <= 0 || shplen > MNDA_MAX_NDIM) {
        PyErr_Format(PyExc_TypeError,
                     "length of shape argument must be 1 ~ %d",
                     MNDA_MAX_NDIM);
        return NULL;
    }
    
    size_t newdims[MNDA_MAX_NDIM] = {0};
    for (int i = shplen - 1; i >= 0; i--) {
        PyObject* shp_el_obj = PySequence_GetItem(shape, i);
        if (NULL == shp_el_obj) {
            PyErr_SetString(PyExc_RuntimeError,
                            "MKLNdarray_Zeros: "
                            "index out of bound in sequence");
            return NULL;
        }

        int shp_el = PyInt_AsLong(shp_el_obj);
        Py_DECREF(shp_el_obj);

        if (shp_el < 0) {
            PyErr_SetString(PyExc_ValueError,
                            "MKLNdarray_Zeros: "
                            "shape must contain only non-negative values for size of a dimension");
            return NULL;
        }
        newdims[i] = (size_t)shp_el;
    }

    PyObject* rval = MKLNdarray_create_with_zeros(shplen, newdims, typenum);
    return (PyObject*)rval;
}


size_t MKLNdarray_get_memory_size(const MKLNdarray *self) {
    if (!self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_memory_size: input is NULL");
        return 0;
    }

    if (NULL == self->layout || NULL == self->data) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_get_memory_size: input object doesn't have "
                        "layout and data buffer");
        return 0;
    }

    size_t data_size = 0;
    if (MNDA_FLOAT64 == self->dtype) {
        data_size = dnnLayoutGetMemorySize_F64(self->layout);
    } else {
        data_size = dnnLayoutGetMemorySize_F32(self->layout);
    }

    return data_size;
}


MKLNdarray* MKLNdarray_View(const MKLNdarray *self) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_View: input is NULL");
        return NULL;
    }

    MKLNdarray *rval = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(self), 
                                                   MKLNdarray_TYPE(self));
    if (!rval) {
        rval = NULL;
    } else {
        int ret = MKLNdarray_set_structure(rval,
                                           MKLNdarray_NDIM(self),
                                           MKLNdarray_DIMS(self),
                                           MKLNdarray_STRIDES(self));
        if (0 != ret) {
            Py_DECREF(rval);
            rval = NULL;
        } else {
            rval->data_size = 0;

            PyObject *orig_base = (PyObject*)self;
            while (orig_base &&
                   MKLNdarray_Check(orig_base) &&
                   ((MKLNdarray*)orig_base)->base) {
                orig_base = ((MKLNdarray*)orig_base)->base;
            }

            rval->base      = orig_base;
            rval->data_size = ((MKLNdarray*)orig_base)->data_size;
            rval->layout    = ((MKLNdarray*)orig_base)->layout;
            rval->data      = ((MKLNdarray*)orig_base)->data;
            rval->flag      |= MNDA_VIEW_FROM_MKL;

            Py_INCREF(orig_base);
        }
    }
    return (MKLNdarray*)rval;
}


MKLNdarray * MKLNdarray_Copy(MKLNdarray *self) {
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError, "MKLNdarray_Copy: input is NULL");
        return NULL;
    }

    MKLNdarray *rval = (MKLNdarray*)MKLNdarray_New(MKLNdarray_NDIM(self),
                                                   MKLNdarray_TYPE(self));
    if (!rval || (-1 == self->nd)) {
        PyErr_SetString(PyExc_RuntimeError,
                "MKLNdarray_Copy: fail to new MKLNdarray");

        Py_XDECREF(rval);
        return NULL;
    }

    int ret = MKLNdarray_set_structure(rval,
                                       MKLNdarray_NDIM(self),
                                       MKLNdarray_DIMS(self),
                                       MKLNdarray_STRIDES(self));
    if (ret) {
        Py_DECREF(rval);
        return NULL;
    }

    rval->data_size = 0;
    if (self->layout) {
        ret = MKLNdarray_copy_layout(rval, self);
        if (ret) {
            Py_DECREF(rval);
            return NULL;
        }
    }
    
    size_t data_size = 0;
    if (rval->layout && self->data) {
        ret = MKLNdarray_create_buffer_from_layout(rval);
        if (ret) {
            Py_DECREF(rval);
            return NULL;
        }

        data_size = MKLNdarray_get_memory_size(rval);
        memcpy(rval->data, self->data, data_size);
    }
    return rval;
}


PyObject * MKLNdarray_DeepCopy(MKLNdarray *self, PyObject *memo) {
    assert (PyDict_Check(memo));
    PyObject *selfkey = PyInt_FromLong((long)self);

    assert (selfkey);

    if (PyDict_Contains(memo, selfkey)) {
        PyObject *rval = PyDict_GetItem(memo, selfkey);
        Py_DECREF(selfkey);
        Py_XINCREF(rval);
        return rval;
    } else {
        PyObject* rval = (PyObject*)MKLNdarray_Copy(self);
        if (NULL == rval) {
            Py_DECREF(selfkey);
            return NULL;
        }

        if (PyDict_SetItem(memo, selfkey, rval)) {
            Py_DECREF(rval);
            Py_DECREF(selfkey);
            return NULL;
        }

        Py_DECREF(selfkey);
        return rval;
    }
}


static PyMethodDef MKLNdarray_methods[] = {
    {"__array__",
        (PyCFunction)MKLNdarray_CreateArrayObj, METH_NOARGS,
        "Copy from MKL to a numpy ndarray."},

    {"zeros",
        (PyCFunction)MKLNdarray_Zeros, METH_STATIC | METH_VARARGS,
        "Create a new MklNdarray with specified shape, filled ith zeros."},

    {"__copy__",
        (PyCFunction)MKLNdarray_View, METH_NOARGS,
        "Create a shallow copy of this object. used by module copy"},

    {"__deepcopy__",
        (PyCFunction)MKLNdarray_DeepCopy, METH_O,
        "Create a copy of this obejct"},

    {"copy",
        (PyCFunction)MKLNdarray_Copy, METH_NOARGS,
        "Create a copy of this object"},

    {"view",
        (PyCFunction)MKLNdarray_View, METH_NOARGS,
        "Return an alias of this ndarray"},

    {NULL, NULL, 0, NULL}  /* Sentinel */
};


static PyMemberDef MKLNdarray_members[] = {
    {NULL}      /* Sentinel */
};


static PyGetSetDef MKLNdarray_getset[] = {
    {"shape",
        (getter)MKLNdarray_get_shape,
        NULL,
        "shape of this ndarray (tuple)",
        NULL},

    {"strides",
        (getter)MKLNdarray_get_strides,
        NULL,
        "strides of this ndarray (tuple)",
        NULL},

    {"dtype",
        (getter)MKLNdarray_get_dtype,
        NULL,
        "the dtype of the element.",
        NULL},

    {"size",
        (getter)MKLNdarray_get_size,
        NULL,
        "the number of elements in this object.",
        NULL},

    {"ndim",
        (getter)MKLNdarray_get_ndim,
        NULL,
        "the number of dimensions in this objec.",
        NULL},

    {"base",
        (getter)MKLNdarray_get_base,
        NULL,
        "if this ndarray is a view, base is the original ndarray.",
        NULL},

    {NULL, NULL, NULL, NULL}  /* Sentinel*/
};


static PyTypeObject MKLNdarrayType = {
#if PY_MAJOR_VERSION >= 3
    PyVarObject_HEAD_INIT(NULL, 0)
#else
    PyObject_HEAD_INIT(NULL)
    0,                         /*ob_size*/
#endif
    "MKLNdarray",              /*tp_name*/
    sizeof(MKLNdarray),        /*tp_basicsize*/
    0,                         /*tp_itemsize*/
    (destructor)MKLNdarray_dealloc, /*tp_dealloc*/
    0,                         /*tp_print*/
    0,                         /*tp_getattr*/
    0,                         /*tp_setattr*/
    0,                         /*tp_compare*/
    MKLNdarray_repr,           /*tp_repr*/
    0,                         /*tp_as_number*/
    0,                         /*tp_as_sequence*/
    0,                         /*tp_as_mapping*/
    0,                         /*tp_hash */
    0,                         /*tp_call*/
    0,                         /*tp_str*/
    0,                         /*tp_getattro*/
    0,                         /*tp_setattro*/
    0,                         /*tp_as_buffer*/
#if PY_MAJOR_VERSION >= 3
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
#else
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE | Py_TPFLAGS_CHECKTYPES, /*tp_flags*/
#endif
    "MKLNdarray objects",      /*tp_doc */
    0,                         /*tp_traverse*/
    0,                         /*tp_clear*/
    0,                         /*tp_richcompare*/
    0,                         /*tp_weaklistoffset*/
    0,                         /*tp_iter*/
    0,                         /*tp_iternext*/
    MKLNdarray_methods,        /*tp_methods*/
    MKLNdarray_members,        /*tp_members*/
    MKLNdarray_getset,         /*tp_getset*/
    0,                         /*tp_base*/
    0,                         /*tp_dict*/
    0,                         /*tp_descr_get*/
    0,                         /*tp_descr_set*/
    0,                         /*tp_dictoffset*/
    (initproc)MKLNdarray_init, /*tp_init*/
    0,                         /*tp_alloc*/
    MKLNdarray_new,            /*tp_new*/
};


int MKLNdarray_Check(const PyObject *ob) {
    return ((Py_TYPE(ob) == &MKLNdarrayType) ? 1 : 0);
}


PyObject*
MKLNdarray_New(int nd, int typenum) {
    if (nd < 0 || nd > MNDA_MAX_NDIM) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray_New: not support a %d-dim array. "
                     "Try array which ndim is <= %d. line: %d",
                     nd, MNDA_MAX_NDIM, __LINE__);
        return NULL;
    }

    if (typenum != MNDA_FLOAT32 && typenum != MNDA_FLOAT64) {
        PyErr_Format(PyExc_ValueError,
                     "MKLNdarray_New: not support dtype=%d, "
                     "Try MNDA_FLOAT32 or MNDA_FLOAT64.",
                     typenum);
        return NULL;
    }

    MKLNdarray* self = (MKLNdarray*)(MKLNdarrayType.tp_alloc(&MKLNdarrayType, 0));
    if (NULL == self) {
        PyErr_SetString(PyExc_RuntimeError,
                        "MKLNdarray_New: failed to call tp_alloc");
        return NULL;
    }
    self->base      = NULL;
    self->flag      = 0;
    self->nd        = nd;
    self->dtype     = typenum;
    self->layout    = NULL;
    self->data      = NULL;
    self->data_size = 0;
    memset((void*)(self->user_structure), 0, 2 * MNDA_MAX_NDIM * sizeof (size_t));

    return (PyObject*)self;
}


///////////////////////////////////////////////////////////
//
// Module
//
///////////////////////////////////////////////////////////

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}   /* Sentinel */
};


#if PY_MAJOR_VERSION == 3
static struct PyModuleDef mkl_ndarray_moduledef =
{
    PyModuleDef_HEAD_INIT,
    "mkl_ndarray",
    "MKL implementation of a numpy ndarray-like object.",
    -1,
    module_methods
};

PyMODINIT_FUNC
PyInit_mkl_ndarray(void)
#else
PyMODINIT_FUNC
initmkl_ndarray(void)
#endif
{
    import_array();
    PyObject* m = NULL;

    if (PyType_Ready(&MKLNdarrayType) < 0) {
#if PY_MAJOR_VERSION == 3
        return NULL;
#else
        return;
#endif
    }

    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float32", PyInt_FromLong(MNDA_FLOAT32));
    PyDict_SetItemString(MKLNdarrayType.tp_dict, "float64", PyInt_FromLong(MNDA_FLOAT64));
#if PY_MAJOR_VERSION == 3
    m = PyModule_Create(&mkl_ndarray_moduledef);
#else
    m = Py_InitModule3("mkl_ndarray", module_methods, "MKL implementation of a numpy ndarray-like object.");
#endif
    if (NULL == m) {
#if PY_MAJOR_VERSION == 3
        return NULL;
#else
        return;
#endif
    }
    Py_INCREF(&MKLNdarrayType);
    PyModule_AddObject(m, "MKLNdarray",   (PyObject*)&MKLNdarrayType);
#if PY_MAJOR_VERSION == 3
    return m;
#else
    return;
#endif
}
