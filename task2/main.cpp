#include <iostream>
#include <cmath>
#include <string>
#include <sstream>
#include <exception>
#include <cstring>

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
    double sum = arr[id-1] + arr[id+1] + arr[id-net_len] + arr[id+net_len];
    return sum/4;
}



int main(int argc,char *argv[])
{

    //Init default values
    double accuracy = std::pow(10,-6);
    unsigned int net_len=128;
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

    //Set limits
    net_len = std::min((unsigned int)std::pow(10,6),net_len);
    accuracy = std::max(std::pow(10,-6),accuracy);


    //Init net and buffer
    int net_size = net_len*net_len;
    double* net = new double[net_size];
    double* net_buff = new double[net_size];
    std::memset(net,0,sizeof(double)*net_size);

    double lu = 10;
    double ru = 20;
    double ld = 30;
    double rd = 20;
#pragma acc data copyin(net[0:net_size]),create(net_buff[0:net_size])
{
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
        max_acc=0.0;

//Set the new array
        if (iter%2==0)
        {
            #pragma acc parallel loop
            for (unsigned int x =1;x<net_len-1;x++)
                #pragma acc loop
                for (unsigned int y=1;y<net_len-1;y++)
                {
                    unsigned int id = y*net_len+x;
                    net_buff[id] = average_neighbours(net,id,net_len);
                }
                    
        }
        else
        {
            #pragma acc parallel loop
            for (unsigned int x =1;x<net_len-1;x++)
                #pragma acc loop
                for (unsigned int y=1;y<net_len-1;y++)
                {
                    unsigned int id = y*net_len+x;
                    net[id] = average_neighbours(net_buff,id,net_len);
                }
        }

    

//Doing reduction to find max
    if (iter % 10 == 0 || iter == iteration_cnt-1)
    {
        #pragma acc data copyin(max_acc)
        #pragma acc parallel loop reduction(max:max_acc)
            for (unsigned int x =1;x<net_len-1;x++)
                #pragma acc loop reduction(max:max_acc)
                for (unsigned int y=1;y<net_len-1;y++)
                {
                    unsigned int i = y*net_len+x;
                    max_acc = fmax(max_acc,fabs(net[i] - net_buff[i]));
                }
                    
        #pragma acc data copyout(max_acc) 
        if (max_acc<accuracy)
            break;
    }


    }

    std::cout<<"Iteration count: "<<iter<<"\n";
    std::cout<<"Accuracy: "<<max_acc<<"\n";
}
#pragma acc exit data delete(net[:net_len],net_buff[:net_len])

    delete[] net;
    delete[] net_buff;
}