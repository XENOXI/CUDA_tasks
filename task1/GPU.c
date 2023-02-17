#include <stdio.h>
#include <malloc.h>
#define _USE_MATH_DEFINES
#include <math.h>
#define N 10000000

int main()
{
    double* arr = (double*)malloc(sizeof(double)*N); 
    double period = 2*M_PI/N;
#pragma acc data create(arr[0:N])
  {
#pragma acc parallel loop
    for (int i =0;i<N;i++)
        arr[i]=sin(period*i);
    double res = 0;

#pragma acc parallel loop,copy(res)
    for (int i =0; i<N;i++)
        res += arr[i];

    printf("res: %.20lf\n",res);

  }
}

