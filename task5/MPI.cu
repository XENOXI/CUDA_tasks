#include <iostream>
#include <mpi.h>
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


__global__ void interpolate(double* A,double* Anew,unsigned int size_x,unsigned int size_y)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x==0 || y==0 || x==size_x-1 || y==size_y-1)
        return;
    Anew[y*size_x+x] = (A[y*size_x+x-1] + A[y*size_x+x+1] + A[(y-1)*size_x+x] + A[(y+1)*size_x+x]) / 4;
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
    int rank,threads_cnt;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&threads_cnt);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount<threads_cnt)
        throw std::runtime_error("Too many MPI threads");
    cudaSetDevice(rank);

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
    unsigned int start = net_len*rank/threads_cnt;
    unsigned int end = net_len*(rank+1)/threads_cnt;
    unsigned int net_len_per_gpu = end-start+1;

    if (threads_cnt==1)
        net_len_per_gpu-=1;
    
    if (rank!=0 && rank != threads_cnt-1)
        net_len_per_gpu += 1;

    unsigned int net_size = net_len_per_gpu*net_len;
    
    double* net_cpu = new double[net_size];
    CREATE_DEVICE_ARR(double,buff,net_size)
    CREATE_DEVICE_ARR(double,net,net_size)
    CREATE_DEVICE_ARR(double,net_buff,net_size)
    CREATE_DEVICE_ARR(double,d_out,1)

    cudaMemset(net,0,sizeof(double)*net_size);

    double lu = 10;
    double ru = 20;
    double ld = 30;
    double rd = 20;

    unsigned int threads_x=NOD(net_len,1024);
    unsigned int blocks_y = net_len_per_gpu;
    unsigned int blocks_x = net_len/threads_x;

    dim3 dim_for_interpolate(threads_x,1);
    dim3 block_for_interpolate(blocks_x,blocks_y);

    if (rank==0)
        for (int i =0;i<net_len;i++)
            net_cpu[i] = (ru-lu)/(net_len-1)*i + lu;
        
    if (rank==threads_cnt-1)
        for (int i =0;i<net_len;i++)
            net_cpu[i+(net_len_per_gpu-1)*net_len] = (rd-ld)/(net_len-1)*i + ld;

    for (int i =0;i<net_len_per_gpu;i++)
    {
        net_cpu[net_len*i] = (ld-lu)/(net_len-1)*(i+start) + lu;
        net_cpu[net_len-1 + net_len*i] = (rd-ru)/(net_len-1)*(i+start) + ru;
    }

    cudaMemcpy(net,net_cpu, sizeof(double)*net_size, cudaMemcpyHostToDevice);
    cudaMemcpy(net_buff,net_cpu, sizeof(double)*net_size, cudaMemcpyHostToDevice);

    unsigned int iter;
    double max_acc=0;

    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
// Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);



//Solving
    for (iter = 0;iter <iteration_cnt;iter++)
    {  
//Set the new array
        interpolate<<<block_for_interpolate,dim_for_interpolate>>>(net,net_buff,net_len,net_len_per_gpu);  
        
//Doing reduction to find max
        if (iter % 100 == 0 || iter == iteration_cnt-1)
        {
            cudaMemcpy(buff,net_buff, sizeof(double)*net_size, cudaMemcpyDeviceToDevice);
            difference<<<blocks_x*blocks_y,threads_x>>>(buff,net);

            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
            cudaMemcpy(&max_acc,d_out, sizeof(double), cudaMemcpyDeviceToHost);
            
            max_acc = std::abs(max_acc);
            bool is_end = false,boolbuff;
            if (max_acc<accuracy)
                is_end=true; 
                

            if(rank!=0)
            {
                MPI_Send(&is_end,1,MPI_C_BOOL,0,0,MPI_COMM_WORLD);
                MPI_Recv(&is_end,1,MPI_C_BOOL,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }         
            else
            {
                for (int i=1;i<threads_cnt;i++)
                {
                    MPI_Recv(&boolbuff,1,MPI_C_BOOL,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    is_end&=boolbuff;
                }
                for (int i=1;i<threads_cnt;i++)
                    MPI_Send(&is_end,1,MPI_C_BOOL,i,0,MPI_COMM_WORLD);
            }
            if(is_end)
                break; 
                                 
        }
        if (rank!=threads_cnt-1)
        {
            MPI_Send(&net_buff[(net_len_per_gpu-2)*net_len],net_len,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD);
            MPI_Recv(&net_buff[(net_len_per_gpu-1)*net_len],net_len,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        if (rank!=0)
        {
            MPI_Recv(net_buff,net_len,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&net_buff[net_len],net_len,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD);
        }
        std::swap(net,net_buff);
        
    }
    CUDACHECK("end")


    if(rank!=0)
        MPI_Send(&max_acc,1,MPI_DOUBLE,0,0,MPI_COMM_WORLD);       
    else
    {
        double max_acc_buff;
        for (int i=1;i<threads_cnt;i++)
        {
            MPI_Recv(&max_acc_buff,1,MPI_DOUBLE,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            max_acc = std::max(max_acc,max_acc_buff);
        }
        std::cout<<"Iteration count: "<<iter<<"\n";
        std::cout<<"Accuracy: "<<max_acc<<"\n";
    }
    cudaFree(net);
    cudaFree(net_buff);
    cudaFree(buff);
    cudaFree(d_out);
    delete[] net_cpu;
    MPI_Finalize();
    return 0;
}


