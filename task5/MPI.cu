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
    MPI_Request request;
    MPI_Init(&argc,&argv);
    
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&threads_cnt);

    //CUDA check device count
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount<threads_cnt)
        throw std::runtime_error("Too many MPI threads");
    cudaSetDevice(rank);

    //Set p2p access
    if (rank!=0)
        cudaDeviceEnablePeerAccess(rank-1,0);
    if (rank!=threads_cnt-1)
        cudaDeviceEnablePeerAccess(rank+1,0);


    //default settings
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
    unsigned int start = net_len*rank/threads_cnt-1;
    unsigned int end = net_len*(rank+1)/threads_cnt+1;

    if (rank==0)
        start+=1;
    if (rank==threads_cnt-1)
        end-=1;

    unsigned int net_len_per_gpu = end-start;
    if (threads_cnt==1)
        net_len_per_gpu=net_len;
    
    unsigned int net_size = net_len_per_gpu*net_len;
    
    double* net_cpu = new double[net_size];
    memset(net_cpu,0,net_size*sizeof(double));

    CREATE_DEVICE_ARR(double,buff,net_size)
    CREATE_DEVICE_ARR(double,net,net_size)
    CREATE_DEVICE_ARR(double,net_buff,net_size)
    CREATE_DEVICE_ARR(double,d_out,1)

    
    //Corners
    double lu = 10;
    double ru = 20;
    double ld = 30;
    double rd = 20;

    //Threads and blocks init
    unsigned int threads_x=NOD(net_len,1024);
    unsigned int blocks_y = net_len_per_gpu;
    unsigned int blocks_x = net_len/threads_x;

    dim3 dim_for_interpolate(threads_x,1);
    dim3 block_for_interpolate(blocks_x,blocks_y);

    //Fill default values
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


    //Init cycle values
    unsigned int iter;
    double max_acc=0;

    //Cub init
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);



    //Start solving
    for (iter = 0;iter <iteration_cnt;iter++)
    {  
        //Set the new array
        interpolate<<<block_for_interpolate,dim_for_interpolate>>>(net,net_buff,net_len,net_len_per_gpu);  
        CUDACHECK("end")
        //Doing reduction to find max
        if (iter % 100 == 0 || iter == iteration_cnt-1)
        {
            cudaMemcpy(buff,net_buff, sizeof(double)*net_size, cudaMemcpyDeviceToDevice);
            difference<<<blocks_x*blocks_y,threads_x>>>(buff,net);

            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, buff, d_out, net_size);
            cudaMemcpy(&max_acc,d_out, sizeof(double), cudaMemcpyDeviceToHost);
            
            //Setting check value on each rank
            max_acc = std::abs(max_acc);
            bool is_end = false,boolbuff;
            if (max_acc<accuracy)
                is_end=true; 


            if(rank!=0)
            {
                //Sending check value to first rank
                MPI_Send(&is_end,1,MPI_C_BOOL,0,0,MPI_COMM_WORLD);
                //Getting check value
                MPI_Recv(&is_end,1,MPI_C_BOOL,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            }         
            else
            {
                //Getting all results and set check value
                for (int i=1;i<threads_cnt;i++)
                {
                    MPI_Recv(&boolbuff,1,MPI_C_BOOL,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    is_end&=boolbuff;
                }
                //Sending it back
                for (int i=1;i<threads_cnt;i++)
                    MPI_Send(&is_end,1,MPI_C_BOOL,i,0,MPI_COMM_WORLD);
            }
            if(is_end)
                break; 
            
                                 
        }
        
        //Exchanging matrix rows between ranks
        //This send penultimate and second rows 
        //and get last and fisrt rows
        if (rank!=threads_cnt-1)
        {
            MPI_Isend(&net_buff[(net_len_per_gpu-2)*net_len+1],net_len-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,&request);
            MPI_Recv(&net_buff[(net_len_per_gpu-1)*net_len+1],net_len-2,MPI_DOUBLE,rank+1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        if (rank!=0)
        {
            MPI_Isend(&net_buff[net_len+1],net_len-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,&request);
            MPI_Recv(&net_buff[1],net_len-2,MPI_DOUBLE,rank-1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
        
        std::swap(net,net_buff);
              
    }
    CUDACHECK("end")

    //Getting results to first process and printing it
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

    //Finishing program
    cudaFree(net);
    cudaFree(net_buff);
    cudaFree(buff);
    cudaFree(d_out);
    delete[] net_cpu;
    MPI_Finalize();
    return 0;
}


