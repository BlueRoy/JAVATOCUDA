#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <jni.h>
#include "svm.h"
#include "SVM_J2C.h"
#include "kernel_matrix_calculation.c"
#include "cross_validation_with_matrix_precomputation.c"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_mode model;
jint nr_fold;

JNIEXPORT void JNICALL Java_SVM_1J2C_do_1cross_1validation
(JNIEnv *env, jobject obj, jint jl, jint jsvm_type, jint jkernel_type, jint jdegree, jdouble jgamma, jdouble jcoef0, jdouble jcache_size, jdouble jeps, jdouble jC, jint nr_weight, jintArray jweight_lable, jdoubleArray jweight, jdouble jnu, jdouble jp, jint jshrinking, jint jprobablility, jint jnr_fold){

    prob.l = jl;
	param.svm_type = param.kernel_type;
	param.degree = jdegree;
	param.gamma = jgamma;
	param.coef0 = jcoef0;
	param.cache_size = jcache_size;
	param.eps = jeps;
	param.C = jC;
	param.nr_weight = jnr_weight;
	param.weight_label = jweight_label;
	param.weight = jweight;
	param.nu = jnu;
	param.p = jp;
	param.shrinking = jshrinking;
	param.probability = jprobability;
    nr_fold = jnr_fold;
    do_cross_validation_with_KM_precalculated();
    return;
}




