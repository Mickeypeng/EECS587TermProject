#include "SVD.h"
using namespace std;

inline void convert(vector<vector<double>> &M, const double a[],int m,int n)
{
    M.clear();
    for(int i=0;i<m;i++)
    {
        M.push_back(vector<double>());
        for(int j=0;j<n;j++)
            M[i].push_back(a[i*n+j]);
    }
}
int main()
{
    double a[9]={1,2,3,4,5,6,7,8,9};
    vector<double> v;
    vector<vector<double> > m;
    convert(m,a,3,3);
    v=SVD(m);
    for(const auto & ele:v)
        cout<<ele<<" ";
    return 0;
}

