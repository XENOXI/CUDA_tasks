#include <stdio.h>
#include <malloc.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#define N 10000000

int main()
{
    clock_t start = clock();
    float* arr = (float*)malloc(sizeof(float)*N);
    float period = 2*M_PI/N;
    for (int i =0;i<N;i++)
        arr[i]=sin(period*i);
    printf("first cycle time,s: %f\n",(float)(clock()-start)/CLOCKS_PER_SEC);
    start = clock();
    float res = 0;
    for (int i =0; i<N;i++)
        res += arr[i];
    
    printf("second cycle time,s: %f\n",(float)(clock()-start)/CLOCKS_PER_SEC);

    printf("res: %.20f\n",res);

    free(arr);
}
