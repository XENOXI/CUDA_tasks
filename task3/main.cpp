#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <exception>
#include <cstring>
#include <openacc.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define CUBLASCHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { printf ("CUBLAS initialization failed\n"); return EXIT_FAILURE; }

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



int main(int argc,char *argv[])
{
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);
    CUBLASCHECK(stat)
    
    

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
    double* net = new double[net_size];
    double* net_buff = new double[net_size];
    double* buff;
    cudaMalloc((void**)&buff, net_size*sizeof(double));
    std::memset(net,0,sizeof(double)*net_size);
    

    double lu = 10;
    double ru = 20;
    double ld = 30;
    double rd = 20;
    
    int idx=0;
    double alpha = -1;



#pragma acc enter data copyin(net[:net_size],net_len),create(net_buff[:net_size])
    #pragma acc parallel loop
    for (unsigned int i=0;i<net_len;i++)
    {
        net[i] = (ru-lu)/(net_len-1)*i + lu;
        net[net_len*i] = (ld-lu)/(net_len-1)*i + lu;
        net[net_len*(net_len-1)+i] = (rd-ld)/(net_len-1)*i + ld;
        net[net_len-1 + net_len*i] = (rd-ru)/(net_len-1)*i + ru;

        net_buff[i] = (ru-lu)/(net_len-1)*i + lu;
        net_buff[net_len*i] = (ld-lu)/(net_len-1)*i + lu;
        net_buff[net_len*(net_len-1)+i] = (rd-ld)/(net_len-1)*i + ld;
        net_buff[net_len-1 + net_len*i] = (rd-ru)/(net_len-1)*i + ru;
    }
    

    //Solving
    unsigned int iter;
    double max_acc=0;
    for (iter = 0;iter <iteration_cnt;iter++)
    {  
//Set the new array
        
        #pragma acc data present(net[:net_size],net_buff[:net_size])
        #pragma acc parallel loop
        for (unsigned int y =1;y<net_len-1;y++)
            #pragma acc loop
            for (unsigned int x=1;x<net_len-1;x++)
            {
                unsigned int id = y*net_len+x;
                net_buff[id] = (net[id-1] + net[id+1] + net[id-net_len] + net[id+net_len])/4;
            }
    

//Doing reduction to find max
        if (iter % 100 == 0 || iter == iteration_cnt-1)
        {
            #pragma acc host_data use_device(net,net_buff)
            {
                cudaMemcpy(buff,net, net_size*sizeof(double), cudaMemcpyDeviceToDevice);
                stat = cublasDaxpy(handle,net_size,&alpha,net_buff,1,buff,1);
                CUBLASCHECK(stat)
                
                stat = cublasIdamax(handle,net_size,buff,1,&idx);
                CUBLASCHECK(stat)
            }
            
            cudaMemcpy(&max_acc,&buff[idx-1], sizeof(double), cudaMemcpyDeviceToHost);
            max_acc = std::abs(max_acc);
            if (max_acc<accuracy)
                break;           
        }

        std::swap(net,net_buff);
    }
    std::cout<<"Iteration count: "<<iter<<"\n";
    std::cout<<"Accuracy: "<<max_acc<<"\n";
// #pragma acc data present(net[:net_size],net_buff[:net_size])
// #pragma acc host_data use_device(net)
//     cudaMemcpy(net_buff,net, net_size*sizeof(double), cudaMemcpyDeviceToHost);
//     for (int i =0;i<15;i++)
//     {
//         for (int j =0;j<15;j++)
//             std::cout<<net_buff[i*15+j]<<" ";
//         std::cout<<"\n";
//     }



#pragma acc exit data delete(net[:net_size],net_buff[:net_size])
    cublasDestroy(handle);
    delete[] net;
    delete[] net_buff;
}