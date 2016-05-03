#include <stdio.h>
#include "umfpack.h"
#include <iostream>
#include <fstream>
#include <string>

using namespace std;
#define path 

void solveEquation(int *Ap, int *Ai, double* Ax, double *b, int size);

int main(void)
{
    int *Ap = new int[44416];
    int *Ai = new int[1071417];
    double *Ax = new double[1071417];
    double *b = new double[44415];
    int count = 0;
    
    string line;
    ifstream myfile ("./data/outer.txt");
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            Ap[count++] = stoi(line);
        }
        myfile.close();
    }
    
    
    count = 0;
    ifstream myfile2 ("./data/inner.txt");
    if (myfile2.is_open())
    {
        while ( getline (myfile2,line) )
        {
            Ai[count++] = stoi(line);
        }
        myfile2.close();
    }
    
    count = 0;
    ifstream myfile3 ("./data/value.txt");
    if (myfile3.is_open())
    {
        while ( getline (myfile3,line) )
        {
            Ax[count++] = stof(line);
        }
        myfile2.close();
    }
    
    count = 0;
    ifstream myfile4 ("./data/B.txt");
    if (myfile4.is_open())
    {
        while ( getline (myfile4,line) )
        {
            b[count++] = stof(line);
        }
        myfile4.close();
    }
    
    
    
    int size = 44415;
    solveEquation(Ap, Ai, Ax, b, size);
    
    return 0;
    
}

void solveEquation(int *Ap, int *Ai, double* Ax, double *b, int n)
{
    double x[n];
    void *Symbolic, *Numeric;
    
    /* symbolic analysis */
    umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, NULL, NULL);
    
    /* LU factorization */
    umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, NULL, NULL);
    umfpack_di_free_symbolic(&Symbolic);
    
    /* solve system */
    umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, NULL, NULL);
    umfpack_di_free_numeric(&Numeric);
    
    for (int i = 0; i < n; i++)
        printf("x[%d] = %g\n", i, x[i]);
    
    ofstream myfile;
    myfile.open ("./data/alpha.txt");
    for(int i = 0;i < n; i++)
        myfile<<x[i]<<endl;
    myfile.close();
    
}