#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "cuda_runtime.h"
#include <cuda.h>
#include "cublas_v2.h"
#include "npy.h"
#include <chrono>
#include <iomanip>

#define CREATE_DEVICE_ARR(type,arg,size) type* arg; cudaMalloc((void**)&arg, sizeof(type) * size);
#define CUBLASCHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS failed\n"); throw std::runtime_error("dfg"); }
#define CUDACHECK(name) do {                        \
  cudaError_t err = cudaGetLastError();             \
  if (err != cudaSuccess) {                         \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(err)); \
    throw std::runtime_error(name);                 \
  }                                                 \
} while(0)

template <class T>
class cudarray
{
private:
    T* _data=NULL;
    uint32_t _size=0;
public:
    void allocate(uint32_t size)
    {
        if (_data)
            cudaFree(_data);

        cudaMalloc((void**)&_data,sizeof(T)*size);
        _size = size;
        
    }
    void resize(uint32_t new_size)
    {
        cudaFree(_data);
        cudaMalloc((void**)&_data,sizeof(T)*new_size);
    }
    cudarray& operator=(const std::vector<T> arr)
    {
        if (arr.size()!=_size)
            this->resize(arr.size());
        cudaMemcpy(_data,arr.data(),_size*sizeof(T),cudaMemcpyHostToDevice);
        return *this;
    }
    ~cudarray()
    {
        if (!_data)
            cudaFree(_data);
    }
    uint32_t size()
    {
        return _size;
    }
    T* data()
    {
        return _data;
    }
    void set_size(uint32_t size)
    {
        _size = size;
    }
};

uint32_t NOD(uint32_t a, uint32_t b)
{
    while(a > 0 && b > 0)
        if(a > b)
            a %= b;
        else
            b %= a;
    return a + b;
}

template <class T>
class Layer
{
protected:
    cudarray<T> in_x;
    cudarray<T> out_x;
    cudarray<T> grad;
    cudarray<T> err_x;
    dim3 threads={1,1,1};
    dim3 blocks={1,1,1};
public:
    virtual cudarray<T> forward(cudarray<T> x) = 0;
    virtual cudarray<T> backward(cudarray<T> err) = 0;
    virtual void read_weights(std::string filepath) {};
};


class Linear : public Layer<float>
{
private:
    cudarray<float> weights;
    cudarray<float> buff;
    cudarray<float> bias;
    cublasHandle_t handle;
    bool h_bias;
    float alpha=1,beta;
public:
    
    Linear(uint32_t in, uint32_t out,bool has_bias = true) : h_bias(has_bias)
    {
        weights.allocate(in*out);
        grad.allocate(in*out);
        err_x.allocate(in);
        out_x.allocate(out);
        in_x.set_size(in);
        
        beta = has_bias;
        if (has_bias)
        {
            bias.allocate(out);
        }
            
        threads.x = NOD(out,1024);
        blocks.x = out/threads.x;
        CUBLASCHECK(cublasCreate(&handle));
    }
    ~Linear()
    {
        cublasDestroy(handle);
    }
    cudarray<float> forward(cudarray<float> x)
    {
        if (in_x.size()!=x.size())
            throw std::runtime_error("Not a valid size");
        in_x = x;
        if (h_bias)
            cudaMemcpy(out_x.data(),bias.data(),sizeof(float)*out_x.size(),cudaMemcpyDeviceToDevice);

    
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_N,1,out_x.size(),in_x.size(),&alpha,in_x.data(),1,weights.data(),in_x.size(),&beta,out_x.data(),1);
        return out_x;
    }
    cudarray<float> backward(cudarray<float> err)
    {
        cublasSgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,1,out_x.size(),in_x.size(),&alpha,err.data(),1,weights.data(),in_x.size(),&beta,err_x.data(),1);
        cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,in_x.size(),out_x.size(),1,&alpha,in_x.data(),in_x.size(),err.data(),1,&beta,grad.data(),in_x.size());
        return err_x;
    }
    void read_weights(std::string filepath)
    {
        std::vector<npy::ndarray_len_t> shape,bias_shape;
        std::vector<float> data,bias_data;
        npy::LoadArrayFromNumpy<float>(filepath,shape,data);

        if (shape[1]!=in_x.size() || shape[0]!=out_x.size())
            throw std::runtime_error("Not a valid shape");

        cudaMemcpy(weights.data(), data.data(),weights.size()*sizeof(float),cudaMemcpyHostToDevice);
        if (h_bias)
        {
            npy::LoadArrayFromNumpy<float>("bias_"+filepath,bias_shape,bias_data);
            cudaMemcpy(bias.data(),bias_data.data(),out_x.size()*sizeof(float),cudaMemcpyHostToDevice);
        }
    }

};


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

class Sigmoid : public Layer<float>
{
public:
    Sigmoid(uint32_t size) {
        out_x.allocate(size);
        err_x.allocate(size);
        in_x.set_size(size);
        threads.x = NOD(size,1024);
        blocks.x = size/threads.x;
    }
    cudarray<float> forward(cudarray<float> x)
    {
        if (in_x.size()!=x.size())
            throw std::runtime_error("Not a valid size");
        in_x = x;
        
        sigm_forward<<<blocks,threads>>>(in_x.data(),out_x.data());

        return out_x;
    }
    cudarray<float> backward(cudarray<float> err)
    {
        sigm_backward<<<blocks,threads>>>(err.data(),out_x.data(),err_x.data());
        return err_x;
    }
    void read_weights(std::string filepath) {}
};

template <class T>
class Model
{
private:
    std::vector<Layer<T>*> layers;
public:
    cudarray<T> forward(cudarray<T> x)
    {
        for (auto lay : layers)
            x = lay->forward(x);
        return x;
    }
    cudarray<T> backward(cudarray<T> out)
    {
        for (auto lay : layers)
            out = lay->backward(out);
    }
    void push_back(Layer<T>* lay)
    {
        layers.push_back(lay);
    }
    void load_from_numpy(std::string filepath)
    {
        auto id = filepath.rfind('.');;
        std::string filename,type;
        for (int i =0;i<id;i++)
            filename += filepath[i];
        for (int i =id;i<filepath.size();i++)
            type += filepath[i];

        for (int i =0;i<layers.size();i++)
            layers[i]->read_weights(filename+std::to_string(i) + type);            
    }
    ~Model()
    {
        for (auto lay : layers)
            delete lay;
    }
};

int main()
{
    
    Model<float> net;

    net.push_back(new Linear(32 * 32, 16 * 16));
    net.push_back(new Sigmoid(16*16));
    net.push_back(new Linear(16 * 16,4*4));
    net.push_back(new Sigmoid(4*4));
    net.push_back(new Linear(4*4,1));
    net.push_back(new Sigmoid(1));


    net.load_from_numpy("npy_weights.npy");
    
    cudarray<float> in;
    in.allocate(32*32);

    std::vector<npy::ndarray_len_t> shape;
    std::vector<float> in_data;
    npy::LoadArrayFromNumpy<float>("data.npy",shape,in_data);

    in = in_data;

    auto start = std::chrono::high_resolution_clock::now();
    auto data = net.forward(in);
    

    float v;
    cudaMemcpy(&v,data.data(),sizeof(float),cudaMemcpyDeviceToHost);
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
    std::cout << std::fixed << std::setprecision(16) << v << std::endl;
    return 0;
}