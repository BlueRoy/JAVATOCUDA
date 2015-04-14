
#include <stdio.h>
#include "J2C.h"


JNIEXPORT jint JNICALL Java_J2C_writetocudac(JNIEnv *env, jobject obj, jint anum, jint bnum, jint cnum)
{
    printf("C: fetching numbers from Java\n");
    
    jint sum = cnum + anum + bnum;
   
    printf("C: Going back to Java\n");
    
    return sum;
    
}
