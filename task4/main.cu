#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <exception>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cub/cub.cuh>

#define CREATE_DEVICE_ARR(type,arg,size) type* arg; cudaMalloc((void**)&arg, sizeof(type) * size);
#define CUDACHECK(name) if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error(name);

template <typename Type>
std::istream& operator>>(std::istream& i, const Type& arg) { return i; }

template <typename Type>
void argpars(Type& arg, std::string& str)
{
	std::stringstream buff;
	buff << str;
	buff >> arg;
	std::string buff2;
	buff2 = str;
	str.clear();
	std::getline(buff, str);
	if (str == buff2) //Nothing changed
		throw std::runtime_error("Not a valid argument");
}

inline double average_neighbours(double* arr,int x,int y,int net_len)
{
    int neigh_cnt = 0;
    double sum = 0;
    int id = y*net_len+x;
    if (x > 0)
    {
        sum += arr[id-1];
        neigh_cnt++;
    }
    if (x+1 < net_len)
    {
        sum += arr[id+1];
        neigh_cnt++;
    }
    if (y > 0)
    {
        sum += arr[id-net_len];
        neigh_cnt++;
    }
    if (y+1 < net_len)
    {
        sum += arr[id+net_len];
        neigh_cnt++;
    }
    return sum/neigh_cnt;
}


__global__ void interpolate(double* A,double* Anew,unsigned int size)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x==0 || y==0 || x==size-1 || y==size-1)
        return;
    Anew[y*size+x] = (A[y*size+x-1] + A[y*size+x+1] + A[(y-1)*size+x] + A[(y+1)*size+x]) / 4;
}

__global__ void set_border(double* A,unsigned int net_len,const double lu,const double ld,const double ru,const double rd)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = (ru-lu)/(net_len-1)*i + lu;
    A[net_len*i] = (ld-lu)/(net_len-1)*i + lu;
    A[net_len*(net_len-1)+i] = (rd-ld)/(net_len-1)*i + ld;
    A[net_len-1 + net_len*i] = (rd-ru)/(net_len-1)*i + ru;
}

__global__ void difference(double* A,double* B)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] -= B[i];
}

int NOD(int a, int b)
{
    while(a > 0 && b > 0)
        if(a > b)
            a %= b;
        else
            b %= a;
    return a + b;
}

int main(int argc,char *argv[])
{

    

    //Init default values
    double accuracy = std::pow(10,-6);
    unsigned int net_len=1024;
    unsigned int iteration_cnt = std::pow(10,6);
    
    //Reading arguments
    for (int i =1;i<argc-1;i+=2)
    {
        std::string argument(argv[i]);
        std::string value(argv[i+1]);

        if (argument=="-accuracy")
            argpars(accuracy,value);
        else if (argument=="-net_len")
            argpars(net_len,value);
        else if (argument=="-iteration")
            argpars(iteration_cnt,value);
    }


    //Init net and buffer
    int net_size = net_len*net_len;

    CREATE_DEVICE_ARR(double,buff,net_size)
    CREATE_DEVICE_ARR(double,net,net_size)
    CREATE_DEVICE_ARR(double,net_buff,net_size)
    CREATE_DEVICE_ARR(double,d_out,1)

    cudaMemset(net,0,sizeof(double)*net_size);

    double lu = 10;
    double ru = 20;
    double ld = 30;
    double rd = 20;

    unsigned int threads=net_len;

    if (threads%32!=0)
        throw std::runtime_error("Not a valid net_len");

    threads = NOD(threads,1024);
    unsigned int blocks = net_len/threads;

    set_border<<<blocks,threads>>>(net,net_len,lu,ld,ru,rd);
    CUDACHECK("set_border")

    cudaMemcpy(net_buff,net, sizeof(double)*net_size, cudaMemcpyDeviceToDevice);

    //Solving
    unsigned int iter;
    double max_acc=0;

    

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    
    dim3 dim_for_interpolate(threads/32,threads/32);
    dim3 block_for_interpolate(blocks*32,blocks*32);


    for (iter = 0;iter <iteration_cnt;iter++)
    {  
//Set the new array
        interpolate<<<block_for_interpolate,dim_for_interpolate>>>(net,net_buff,net_len);  

//Doing reduction to find max
        if (iter % 100 == 0 || iter == iteration_cnt-1)
        {
            cudaMemcpy(buff,net_buff, sizeof(double)*net_size, cudaMemcpyDeviceToDevice);
            difference<<<threads*blocks*blocks,threads>>>(buff,net);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
                
            cudaMemcpy(&max_acc,d_out, sizeof(double), cudaMemcpyDeviceToHost);
            max_acc = std::abs(max_acc);
            if (max_acc<accuracy)
                break;            
        }
        std::swap(net,net_buff);
    }
    CUDACHECK("end")
    std::cout<<"Iteration count: "<<iter<<"\n";
    std::cout<<"Accuracy: "<<max_acc<<"\n";
    cudaFree(net);
    cudaFree(net_buff);
    cudaFree(buff);

}