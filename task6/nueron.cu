#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "cuda_runtime.h"
#include <cuda.h>
#include "cublas_v2.h"

#define CREATE_DEVICE_ARR(type,arg,size) type* arg; cudaMalloc((void**)&arg, sizeof(type) * size);

typedef struct farray
{
    float* data;
    uint32_t size;
}farray;

template <class T>
std::vector<T> get_array(std::string filepath)
{
    std::ifstream file(filepath);
    std::vector<T> arr;
    T buff;
    while (!file.eof())
    {
        file >> buff;
        arr.push_back(buff);
    }
    arr.pop_back();
    return arr;
}

uint32_t NOD(uint32_t a, uint32_t b)
{
    while(a > 0 && b > 0)
        if(a > b)
            a %= b;
        else
            b %= a;
    return a + b;
}

std::vector<float> transpose(std::vector<float> in,uint32_t a,uint32_t b)
{
    std::vector<float> out(a*b);
    for(uint32_t i = 0; i < a; i++)
        for(uint32_t j = i; j < b; j++)
            out[j*a+i] = in[i*b+j];
    return out;
}

class Layer
{
protected:
    farray in_x;
    farray out_x;
    farray grad;
    farray err_x;
    dim3 threads={1,1,1};
    dim3 blocks={1,1,1};
public:
    virtual farray forward(farray x) = 0;
    virtual farray backward(farray err)=0;
    virtual void read_weights(std::string filepath) {};
};




class Linear : public Layer
{
private:
    farray weights;
    farray buff;
    cublasHandle_t handle;
    uint32_t in_size;
    uint32_t out_size;
    float alpha = 1;
    float beta = 0;
public:
    Linear(uint32_t in, uint32_t out) : in_size(in), out_size(out)
    {
        cudaMalloc((void**)&weights.data,in*out*sizeof(float));
        cudaMalloc((void**)&grad.data,in*out*sizeof(float));
        cudaMalloc((void**)&err_x.data,in*sizeof(float));
        weights.size = in*out;
        cublasCreate(&handle);
    }
    farray forward(farray x)
    {
        in_x = x;
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,out_size,in_size,&alpha,x.data,1,weights.data,in_size,&beta,out_x.data,1);
        return out_x;
    }
    farray backward(farray err)
    {
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,1,out_size,in_size,&alpha,err.data,1,weights.data,in_size,&beta,err_x.data,1);
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,in_size,out_size,1,&alpha,in_x.data,in_size,err.data,1,&beta,grad.data,in_size);
        return err_x;
    }
    void read_weights(std::string filepath)
    {
        cudaMemcpy(weights.data,transpose(get_array<float>(filepath),out_size,in_size).data(),weights.size*sizeof(float),cudaMemcpyHostToDevice);
    }

};

class Sigmoid : public Layer
{
private:
    __global__ void sigm_forward(float* in,float* out)
    {
        uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        out[id] = 1/(1+exp(-in[id]));
    }
    __global__ void sigm_backward(float* err,float* out,float* grad)
    {
        uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
        grad[id] = err[id]*out[id]*(1-out[id]);
    }
public:
    Sigmoid(uint32_t size) {
        cudaMalloc((void**)&out_x.data,size*sizeof(float));
        cudaMalloc((void**)&err_x.data,size*sizeof(float));
        threads.x = NOD(size,1024);
        blocks.x = 1024/threads.x;
        out_x.size = size;
        in_x.size = size;
    }
    farray forward(farray x)
    {
        in_x = x;
        sigm_forward<<<blocks,threads>>>(x.data,out_x.data);
        return out_x;
    }
    farray backward(farray err)
    {
        sigm_backward<<<blocks,threads>>>(err.data,out_x.data,err_x.data);
        return err_x;
    }
    void read_weights(std::string filepath) {}
};

class Model
{
public:
    std::vector<Layer*> layers;
    farray forward(farray x)
    {
        for (auto lay : layers)
            x = lay->forward(x);
        return x;
    }
    void backward(farray out)
    {
        for (auto lay : layers)
            out = lay->backward(out);
    }
   
};

int main()
{
    auto vec = get_array<float>("weight.txt");

    auto net = Model();
    net.layers.push_back(new Linear(32 * 32, 16 * 16));
    net.layers.push_back(new Sigmoid(16*16));
    net.layers.push_back(new Linear(16 * 16,4*4));
    net.layers.push_back(new Sigmoid(4*4));
    net.layers.push_back(new Linear(4*4,1));
    net.layers.push_back(new Sigmoid(1));


    farray in;
    cudaMalloc((void**)&in.data,sizeof(float)*32*32);
    cudaMemset(in.data,0,sizeof(float)*32*32);
    
    
    std::cout << net.forward(in).data[0] << std::endl;
    return 0;
}