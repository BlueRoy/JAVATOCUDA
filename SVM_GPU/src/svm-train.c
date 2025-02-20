#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <malloc.h>
#include "svm.h"
#include "SVM_J2C.h"
#include "/usr/local/cuda-7.0/include/cuda_runtime.h"
#include "/usr/local/cuda-7.0/include/cublas_v2.h"
//#include "kernel_matrix_calculation.c"
//#include "cross_validation_with_matrix_precomputation.c"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

struct svm_parameter param;		// set by parse_command_line
struct svm_problem prob;        // set by read_problem
struct svm_mode *model;
struct svm_node *x_space;
jint nr_fold;
static char *line = NULL;
static int max_line_len;

JNIEXPORT void JNICALL Java_SVM_1J2C_do_1cross_1validation
(JNIEnv *env, jobject obj, jstring jfilename, jint jsvm_type, jint jkernel_type, jint jdegree, jdouble jgamma, jdouble jcoef0, jdouble jcache_size, jdouble jeps, jdouble jC, jint jnr_weight, jdouble jnu, jdouble jp, jint jshrinking, jint jprobability, jint jnr_fold){
	param.svm_type = param.kernel_type;
	param.degree = jdegree;
	param.gamma = jgamma;
	param.coef0 = jcoef0;
	param.cache_size = jcache_size;
	param.eps = jeps;
	param.C = jC;
	param.nr_weight = jnr_weight;
	//param.weight_label = *jweight_lable;
	//param.weight = *jweight;
	param.nu = jnu;
	param.p = jp;
	param.shrinking = jshrinking;
	param.probability = jprobability;
    	nr_fold = jnr_fold;
    
    	const char* filename = (char*) (*env)->GetStringUTFChars(env,jfilename,JNI_FALSE);
    	printf("%s\n",filename);
	read_problem(filename);
    	(*env)->ReleaseStringUTFChars(env,jfilename,filename);
    	do_cross_validation_with_KM_precalculated();
   	 return;
}


static char* readline(FILE *input)
{
    int len;
    
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;
    
    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line,max_line_len);
        len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    return line;
}


void read_problem(const char *filename)
{
    int elements, max_index, inst_max_index, i, j;
#ifdef _DENSE_REP
    double value;
#endif
    FILE *fp = fopen(filename,"r");
    char *endptr;
    char *idx, *val, *label;
    
    if(fp == NULL)
    {
        fprintf(stderr,"can't open input file %s\n",filename);
        exit(1);
    }
    
    prob.l = 0;
    elements = 0;
    
    max_line_len = 1024;
    line = Malloc(char,max_line_len);
#ifdef _DENSE_REP
    max_index = 1;
    while(readline(fp) != NULL)
    {
        char *p;
        p = strrchr(line, ':');
        if(p != NULL)
        {
            while(*p != ' ' && *p != '\t' && p > line)
                p--;
            if(p > line)
                max_index = (int) strtol(p,&endptr,10) + 1;
        }
        if(max_index > elements)
            elements = max_index;
        ++prob.l;
    }
    
    rewind(fp);
    
    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(struct svm_node,prob.l);
    
    for(i=0;i<prob.l;i++)
    {
        int *d;
        (prob.x+i)->values = Malloc(double,elements);
        (prob.x+i)->dim = 0;
        
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        
        label = strtok(line," \t");
        prob.y[i] = strtod(label,&endptr);
        if(endptr == label)
            exit_input_error(i+1);
        
        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");
            
            if(val == NULL)
                break;
            
            errno = 0;
            j = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || j <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = j;
            
            errno = 0;
            value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);
            
            d = &((prob.x+i)->dim);
            while (*d < j)
                (prob.x+i)->values[(*d)++] = 0.0;
            (prob.x+i)->values[(*d)++] = value;
        }
    }
    max_index = elements-1;
    
#else
    while(readline(fp)!=NULL)
    {
        char *p = strtok(line," \t"); // label
        
        // features
        while(1)
        {
            p = strtok(NULL," \t");
            if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
                break;
            ++elements;
        }
        ++elements;
        ++prob.l;
    }
    rewind(fp);
    
    prob.y = Malloc(double,prob.l);
    prob.x = Malloc(struct svm_node *,prob.l);
    x_space = Malloc(struct svm_node,elements);
    
    max_index = 0;
    j=0;
    for(i=0;i<prob.l;i++)
    {
        inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
        readline(fp);
        prob.x[i] = &x_space[j];
        label = strtok(line," \t\n");
        if(label == NULL) // empty line
            exit_input_error(i+1);
        
        prob.y[i] = strtod(label,&endptr);
        if(endptr == label || *endptr != '\0')
            exit_input_error(i+1);
        
        while(1)
        {
            idx = strtok(NULL,":");
            val = strtok(NULL," \t");
            
            if(val == NULL)
                break;
            
            errno = 0;
            x_space[j].index = (int) strtol(idx,&endptr,10);
            if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
                exit_input_error(i+1);
            else
                inst_max_index = x_space[j].index;
            
            errno = 0;
            x_space[j].value = strtod(val,&endptr);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
                exit_input_error(i+1);
            
            ++j;
        }
        
        if(inst_max_index > max_index)
            max_index = inst_max_index;
        x_space[j++].index = -1;
    }
#endif
    
    if(param.gamma == 0 && max_index > 0)
        param.gamma = 1.0/max_index;
    
    if(param.kernel_type == PRECOMPUTED)
        for(i=0;i<prob.l;i++)
        {
#ifdef _DENSE_REP
            if ((prob.x+i)->dim == 0 || (prob.x+i)->values[0] == 0.0)
            {
                fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                exit(1);
            }
            if ((int)(prob.x+i)->values[0] < 0 || (int)(prob.x+i)->values[0] > max_index)
            {
                fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                exit(1);
            }
#else
            if (prob.x[i][0].index != 0)
            {
                fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
                exit(1);
            }
            if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
            {
                fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
                exit(1);
            }
#endif
        }
    fclose(fp);
}

void setup_pkm(struct svm_problem *p_km)
{

        int i;

        p_km->l = prob.l;
        p_km->x = Malloc(struct svm_node,p_km->l);
        p_km->y = Malloc(double,p_km->l);
        for(i=0;i<prob.l;i++)
        {
                (p_km->x+i)->values = Malloc(double,prob.l+1);
                (p_km->x+i)->dim = prob.l+1;
        }
        for( i=0; i<prob.l; i++) p_km->y[i] = prob.y[i];
}

void free_pkm(struct svm_problem *p_km)
{

        int i;

        for(i=0;i<prob.l;i++)
                free( (p_km->x+i)->values);

        free( p_km->x );
        free( p_km->y );

}

double do_crossvalidation(struct svm_problem * p_km)
{
                        double rate;

                        int i;
                        int total_correct = 0;
                        double total_error = 0;
                        double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
                        double *target = Malloc(double,prob.l);

                        svm_cross_validation(p_km,&param,nr_fold,target);


                        if(param.svm_type == EPSILON_SVR ||
                                param.svm_type == NU_SVR)
                        {
                                for(i=0;i<prob.l;i++)
                                {
                                        double y = prob.y[i];
                                        double v = target[i];
                                        total_error += (v-y)*(v-y);
                                        sumv += v;
                                        sumy += y;
                                        sumvv += v*v;
                                        sumyy += y*y;
                                        sumvy += v*y;
                                }
                                printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
                                printf("Cross Validation Squared correlation coefficient = %g\n",
                                                                ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
                                                                ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
                                                                );
                        }
                        else
                        {
                                for(i=0;i<prob.l;i++)
                                        if(target[i] == prob.y[i])
                                                ++total_correct;

                                rate = (100.0*total_correct)/prob.l;


                        }
                        free(target);


                        return rate;

}

void run_pair(struct svm_problem * p_km)
{

        double rate;
        cal_km(p_km);

        param.kernel_type = PRECOMPUTED;

        rate = do_crossvalidation(p_km);

        printf("Cross Validation = %g%%\n", rate);


}


void do_cross_validation_with_KM_precalculated()
{
        struct svm_problem p_km;
	
        setup_pkm(&p_km);
        run_pair(&p_km);
        free_pkm(&p_km);

}

// Scalars
const float alpha = 1;
const float beta = 0;

void ckm( struct svm_problem *prob, struct svm_problem *pecm, float *gamma  )
{
        cublasStatus_t status;

        double g_val = *gamma;

        long int nfa;

        int len_tv;
        int ntv;
        int i_v;
        int i_el;
        int i_r, i_c;
        int trvei;

        double *tv_sq;
        double *v_f_g;

        float *tr_ar;
        float *tva, *vtm, *DP;
        float *g_tva = 0, *g_vtm = 0, *g_DotProd = 0;

        cudaError_t cudaStat;
        cublasHandle_t handle;
        cudaSetDevice(0);
        status = cublasCreate(&handle);

        len_tv = prob-> x[0].dim;
        ntv   = prob-> l;
	printf("%d,%d\n",len_tv,ntv);
        nfa = len_tv * ntv;
	
        tva = (float*) malloc ( len_tv * ntv* sizeof(float) );
        vtm = (float*) malloc ( len_tv * sizeof(float) );
        DP  = (float*) malloc ( ntv * sizeof(float) );

        tr_ar = (float*) malloc ( len_tv * ntv* sizeof(float) );

        tv_sq = (double*) malloc ( ntv * sizeof(double) );

        v_f_g  = (double*) malloc ( ntv * sizeof(double) );
        for ( i_r = 0; i_r < ntv ; i_r++ )
        {
                for ( i_c = 0; i_c < len_tv; i_c++ )
                        tva[i_r * len_tv + i_c] = (float)prob-> x[i_r].values[i_c];
        }
	cudaStat = cudaMalloc((void**)&g_tva, len_tv * ntv * sizeof(float));
	
        if (cudaStat != cudaSuccess) {
                free( tva );
                free( vtm );
                free( DP  );

                free( v_f_g );
                free( tv_sq );

                cudaFree( g_tva );
                cublasDestroy( handle );

                fprintf (stderr, "!!!! Device memory allocation error (A)\n");
                getchar();
                return;
    }
	printf("**********");
        cudaStat = cudaMalloc((void**)&g_vtm, len_tv * sizeof(float));

        cudaStat = cudaMalloc((void**)&g_DotProd, ntv * sizeof(float));

        for( i_r = 0; i_r < ntv; i_r++ )
                for( i_c = 0; i_c < len_tv; i_c++ )
                        tr_ar[i_c * ntv + i_r] = tva[i_r * len_tv + i_c];

	// Copy cpu vector to gpu vector
        status = cublasSetVector( len_tv * ntv, sizeof(float), tr_ar, 1, g_tva, 1 );
        free( tr_ar );

        for( i_v = 0; i_v < ntv; i_v++ )
        {
                tv_sq[ i_v ] = 0;
                for( i_el = 0; i_el < len_tv; i_el++ )
                        tv_sq[i_v] += pow( tva[i_v*len_tv + i_el], (float)2.0 );
        }



        for ( trvei = 0; trvei < ntv; trvei++ )
        {
                status = cublasSetVector( len_tv, sizeof(float), &tva[trvei * len_tv], 1, g_vtm, 1 );

                status = cublasSgemv( handle, CUBLAS_OP_N, ntv, len_tv, &alpha, g_tva, ntv , g_vtm, 1, &beta, g_DotProd, 1 );

                status = cublasGetVector( ntv, sizeof(float), g_DotProd, 1, DP, 1 );

                for ( i_c = 0; i_c < ntv; i_c++ )
                        v_f_g[i_c] = exp( -g_val * (tv_sq[trvei] + tv_sq[i_c]-((double)2.0)* (double)DP[i_c] ));


                pecm-> x[trvei].values[0] = trvei + 1;

                for ( i_c = 0; i_c < ntv; i_c++ )
                        pecm-> x[trvei].values[i_c + 1] = v_f_g[i_c];


        }

        free( tva );
        free( vtm );
        free( DP  );
        free( v_f_g );
        free( tv_sq );

        cudaFree( g_tva );
        cudaFree( g_vtm );
        cudaFree( g_DotProd );

        cublasDestroy( handle );
}

void cal_km( struct svm_problem * p_km)
{
        float gamma = param.gamma;
        ckm(&prob, p_km, &gamma);
}

