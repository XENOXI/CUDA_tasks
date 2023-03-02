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

inline double average_neighbours(double* arr,int id,int net_size)
{

    int x = id%net_size;
    int y = id/net_size;
    int neigh_cnt = 1;
    double sum = arr[id];
    if (x-1 >= 0)
    {
        sum += arr[id-1];
        neigh_cnt++;
    }
    if (x+1 < net_size)
    {
        sum += arr[id+1];
        neigh_cnt++;
    }
    if (y-1 >= 0)
    {
        sum += arr[id-net_size];
        neigh_cnt++;
    }
    if (y+1 < net_size)
    {
        sum += arr[id+net_size];
        neigh_cnt++;
    }
    return sum/neigh_cnt;
}



int main(int argc,char *argv[])
{

    //Init default values
    double accuracy = std::pow(10,-6);
    int net_len=128;
    int iteration_cnt = std::pow(10,6);

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
#pragma acc data copyin(net[0:net_size]),create(net_buff[0:net_size])
{
#pragma acc parallel loop
    for (int i=0;i<net_len;i++)
    {
        net[i] = (ru-lu)/(net_len-1)*i + lu;
        net[net_len*i] = (ld-lu)/(net_len-1)*i + lu;
        net[net_len*(net_len-1)+i] = (rd-ld)/(net_len-1)*i + ld;
        net[net_len-1 + net_len*i] = (rd-ru)/(net_len-1)*i + ru;
    }
    

    //Solving
    double max_acc;
    int iter;
    for (iter = 0;iter <iteration_cnt;iter++)
    {
        max_acc=0.0;

//Set the new array
#pragma acc parallel loop
        for (int i =0;i<net_size;i++)
            net_buff[i] = average_neighbours(net,i,net_len);

//Doing reduction to find max
#pragma acc parallel loop reduction(max:max_acc)
        for (int i =0;i<net_len;i++)
            max_acc = fmax(max_acc,fabs(net[i] - net_buff[i]));

        if (max_acc<accuracy)
            break;

//Copy array back
#pragma acc parallel loop
        for (int i =0;i<net_size;i++)
            net[i]=net_buff[i];

    }

    std::cout<<"Iteration count: "<<iter<<"\n";
    std::cout<<"Accuracy: "<<max_acc<<"\n";
}
#pragma acc exit data delete(net[:net_len],net_buff[:net_len])

    delete[] net;
    delete[] net_buff;
}