#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <exception>
#include <cstring>
#include <openacc.h>

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

inline double average_neighbours(double* arr,unsigned int id,unsigned int net_len)
{
    double sum =0 ;
    return sum/4;
}



int main(int argc,char *argv[])
{
    acc_set_device_num( 2, acc_device_nvidia );  


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
    std::memset(net,0,sizeof(double)*net_size);
    

    double lu = 10;
    double ru = 20;
    double ld = 30;
    double rd = 20;
    
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
    double max_acc;
    unsigned int iter;
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
            max_acc=0.0;
            #pragma acc data copy(max_acc)
            #pragma acc parallel loop reduction(max:max_acc)
                for (unsigned int y =1;y<net_len-1;y++)
                    #pragma acc loop reduction(max:max_acc)
                    for (unsigned int x=1;x<net_len-1;x++)
                    {
                        unsigned int i = y*net_len+x;
                        max_acc = fmax(max_acc,fabs(net[i] - net_buff[i]));
                    }

            if (max_acc<accuracy)
                break;
        }

        std::swap(net,net_buff);
    }
    std::cout<<"Iteration count: "<<iter<<"\n";
    std::cout<<"Accuracy: "<<max_acc<<"\n";
    

#pragma acc exit data delete(net[:net_size],net_buff[:net_size])

    delete[] net;
    delete[] net_buff;
}