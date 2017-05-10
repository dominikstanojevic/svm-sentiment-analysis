//
// Created by Dominik on 6.3.2017..
//
#include <Python.h>
#include <numpy/arrayobject.h>
#include "smo.h"
#include <stdio.h>
#include <time.h>

static PyObject *smo_smo(PyObject *self, PyObject *args) {
    PyObject *input = NULL, *response = NULL;
	int iter;
    double C, tol;

    if (!PyArg_ParseTuple(args, "OOidd", &input, &response, &iter, &C,
                          &tol)) {
        return NULL;
    }

    PyArrayObject *arr_in = PyArray_FROM_OTF(input, PyArray_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (arr_in == NULL || arr_in->nd != 2 || arr_in->descr->type_num != PyArray_DOUBLE) {
        PyErr_SetString(PyExc_ValueError, "Input array must be two-dimensional and of type float.");
        return NULL;
    }

    PyArrayObject *arr_res = PyArray_FROM_OTF(response, PyArray_LONG, NPY_ARRAY_IN_ARRAY);
    if (arr_res->nd != 1 || arr_res->descr->type_num != PyArray_LONG) {
        PyErr_SetString(PyExc_ValueError, "Response array must be one dimensional and of type integer.");
        goto fail;
    }

    double *x = (double *) PyArray_DATA(arr_in);
    int *y = (int *) PyArray_DATA(arr_res);


    Parameters parameters;
    parameters.C = C;
    parameters.iter = iter;
    parameters.input = x;
    parameters.response = y;
    parameters.examples = arr_in->dimensions[0];
    parameters.features = arr_in->dimensions[1];
    parameters.tol = tol;

    //Py_BEGIN_ALLOW_THREADS;
    DecisionFunction *function = smo(&parameters);
    //Py_END_ALLOW_THREADS;


    npy_intp *dims = (npy_intp *) malloc(sizeof(npy_intp));
    dims[0] = parameters.features - 1;
    PyObject *multipliers = PyArray_SimpleNewFromData(1, (npy_intp *) dims, NPY_DOUBLE,
                                                      (void *) function->weights);

    PyObject *ret = Py_BuildValue("Nd", multipliers, function->bias);

    free(function);

    Py_DECREF(arr_in);
    Py_DECREF(arr_res);

    return ret;

    fail:
    Py_XDECREF(arr_in);
    PyArray_XDECREF_ERR(arr_res);
    return NULL;

}

static PyMethodDef smo_methods[] = {
        {
                "smo", smo_smo, METH_VARARGS, "SMO algorithm"
        },
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef smo_definition = {
        PyModuleDef_HEAD_INIT,
        "smo",
        "SMO algorithm module",
        -1,
        smo_methods
};

PyMODINIT_FUNC PyInit_smo(void) {
    PyObject *m = PyModule_Create(&smo_definition);
    import_array();

    if (m == NULL) {
        return NULL;
    }

    return m;
}