//==============================================================
//
// SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
// http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
//
// Copyright 2016-2017 Intel Corporation
//
// THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
// NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
//
// =============================================================
/*******************************************************************************
*   This example measures performance of computing the real matrix product 
*   C=alpha*A*B+beta*C using Intel(R) MKL function dgemm, where A, B, and C are 
*   matrices and alpha and beta are double precision scalars. 
*
*   In this simple example, practices such as memory management, data alignment, 
*   and I/O that are necessary for good programming style and high MKL 
*   performance are omitted to improve readability.
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"
#include "helper_string.h"
/* Consider adjusting LOOP_COUNT based on the performance of your computer */
/* to make sure that total run time is at least 1 second */
// #define LOOP_COUNT 1000

int m, p, n;
int loop = 1000;

void parseCmdLine(int argc, char **argv)
{
        if(checkCmdLineFlag( argc, (const char**) argv, "help") || (argc == 1))
        {
                printf("--help:\t\t\t print this menu\n");
                printf("--loop=[int]:\t\t number of kerenl execution times (default: 1000)\n");
                printf("--M=[int]:\t\t height of W \n");
                printf("--P=[int]:\t\t width of W \n");
                printf("--N=[int]:\t\t width of X \n");
                exit(0);
        }

        if(checkCmdLineFlag( argc, (const char**) argv, "loop"))
        {
                loop = getCmdLineArgumentInt( argc, (const char**) argv, "loop");
        }

        if (checkCmdLineFlag( argc, (const char**) argv, "M"))
        {
                m = getCmdLineArgumentInt( argc, (const char**) argv, "M");
        }

        if (checkCmdLineFlag( argc, (const char**) argv, "P"))
        {
                p = getCmdLineArgumentInt( argc, (const char**) argv, "P");
        }

        if (checkCmdLineFlag( argc, (const char**) argv, "N"))
        {
                n = getCmdLineArgumentInt( argc, (const char**) argv, "N");
        }
}


int main(int argc, char** argv)
{
    double *A, *B, *C;
    int i, r;
    double alpha, beta;
    double s_initial, s_elapsed;

    m = 2560, p = 2560, n = 64;

    parseCmdLine(argc, argv);

    printf ("matrixA(%ix%i) \t matrixB(%ix%i) \t", m, p, p, n);
    alpha = 1.0; beta = 0.0;

    
    A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    B = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\nERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    //printf (" Intializing matrix data \n\n");
    for (i = 0; i < (m*p); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (p*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }


    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, p, alpha, A, p, B, n, beta, C, n);


    s_initial = dsecnd();
    for (r = 0; r < loop; r++) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, p, alpha, A, p, B, n, beta, C, n);
    }
    // s_elapsed = (dsecnd() - s_initial) / loop;
    s_elapsed = (dsecnd() - s_initial);

    /*
    printf (" == Matrix multiplication using Intel(R) MKL dgemm completed == \n"
            " == at %.5f milliseconds == \n\n", (s_elapsed * 1000));
    */
    printf (
            "%d iterations \t %.5f ms \n\n",loop ,(s_elapsed * 1000));
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    /*
    if (s_elapsed < 0.9/loop) {
        s_elapsed=1.0/loop/s_elapsed;
        i=(int)(s_elapsed*loop)+1;
        printf(" It is highly recommended to define loop for this example on your \n"
               " computer as %i to have total execution time about 1 second for reliability \n"
               " of measurements\n\n", i);
    }
    */
    return 0;
}
