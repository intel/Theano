#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import timeit
import commands
import numpy
from nose.plugins.skip import SkipTest

import theano
import theano.tensor.signal.conv
from theano import tensor
from theano.contrib import mkl
from theano.contrib.mkl.basic_ops import U2IConv, I2U
from theano.contrib.mkl.mkl_conv import Conv2D

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


def print_help(exit_status):
    if exit_status:
        print('command "%s" not recognized' % (' '.join(sys.argv)))
    print('Type "theano-perf" to print this help')
    print('Type "theano-perf help" to print this help')
    print('Type "theano-perf gem" to test gem')
    print('Type "theano-perf mem" to test memory bandwidth')
    print('Type "theano-perf disk" to test disk write and read speed')
    print('Type "theano-perf op" to test benchmark of op')

    sys.exit(exit_status)


if len(sys.argv) == 1:
    print_help(exit_status=0)

elif len(sys.argv) == 2:
    if sys.argv[1] == 'help':
        print_help(exit_status=0)

    if sys.argv[1] == 'gem':
        (status, output) = commands.getstatusoutput(
            "lscpu | grep -i 'Model name'")
        if (status == 0):
            print ("CPU model: %s" % (output.split(':')[1].strip()))
        (status, output) = commands.getstatusoutput("lscpu | grep -i 'CPU(s)'")
        if (status == 0):
            print ("CPU cores: %s" % (output.split('\n')[0].split(':')[1].strip()))
        (status, output) = commands.getstatusoutput("lscpu | grep -i 'CPU MHz'")
        if (status == 0):
            print ("CPU MHz: %s \n" % (output.split(':')[1].strip()))

        status = os.system('g++ dgemm_with_timing.c -o a.out -lmkl_rt')
        if (status == 0):
            print("Compiled gem test code.\n")
            print("This code measures performance of Intel(R) MKL function dgemm,",
                "computing real matrix C=alpha*A*B+beta*C, where A, B, and C",
                "are matrices and alpha and beta are double precision scalars.\n")
        status = os.system('bash run_gemm.sh')
        if (status == 0):
            print("Ran gem test code.\n")

    elif sys.argv[1] == 'mem':
        (status, output) = commands.getstatusoutput("icpc -mcmodel=medium -O3 -qopenmp bw.cpp gettime.cpp -o bw.out -liomp5 ")
        if (status == 0):
            print("Compiled memeroy test code.\n")
            status = os.system("./bw.out")
            if (status == 0):
                print("Ran memory test code.\n")

    elif sys.argv[1] == 'disk':
        (status, output) = commands.getstatusoutput('dd if=/dev/zero of=ddfile bs=8k count=250000')
        if (status == 0):
            print("Disk write test done, write speed: %s\n" % (output.split('\n')[2].split(',')[2].strip()))
            (status2, output) = commands.getstatusoutput('dd if=ddfile of=/dev/null bs=8k count=250000')
            if (status2 == 0):
                print("Disk read test done, read speed: %s\n" % (output.split('\n')[2].split(',')[2].strip()))
            (status, output) = commands.getstatusoutput('rm -rf ./ddfile')

    elif sys.argv[1] == 'op':
        images = tensor.dtensor4('inputs')
        weights = tensor.dtensor4('weights')

        images_internal = U2IConv(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3))(images)
        convOut_internal = Conv2D(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3), filter_flip=False)(images_internal, weights)
        convOut_user = I2U()(convOut_internal)

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float64)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float64)

        fopt = theano.function(inputs=[images, weights], outputs=convOut_user, mode=mode_with_mkl)
        start_time = timeit.default_timer()
        for i in range(300):
            new_out = fopt(ival, wval)

        end_time = timeit.default_timer()

        print ('Conv image size(12, 3, 256, 256), filter size(12, 3, 3, 3) ran 50 epoches for %is' % ((end_time - start_time)))

    else:
        print_help(exit_status=1)

elif len(sys.argv) == 3 and sys.argv[1] == 'op':
    if sys.argv[2] == 'conv':
        images = tensor.dtensor4('inputs')
        weights = tensor.dtensor4('weights')

        images_internal = U2IConv(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3))(images)
        convOut_internal = Conv2D(imshp=(12, 3, 256, 256), kshp=(12, 3, 3, 3), filter_flip=False)(images_internal, weights)
        convOut_user = I2U()(convOut_internal)

        ival = numpy.random.rand(12, 3, 256, 256).astype(numpy.float64)
        wval = numpy.random.rand(12, 3, 3, 3).astype(numpy.float64)

        fopt = theano.function(inputs=[images, weights], outputs=convOut_user, mode=mode_with_mkl)
        start_time = timeit.default_timer()
        for i in range(300):
            new_out = fopt(ival, wval)

        end_time = timeit.default_timer()

        print ('Conv image size(12, 3, 256, 256), filter size(12, 3, 3, 3) ran 50 epoches for %is' % ((end_time - start_time)))

    elif sys.argv[2] == 'relu':
        print("TBD")
    else:
        print_help(exit_status=1)
else:
    print_help(exit_status=1)
