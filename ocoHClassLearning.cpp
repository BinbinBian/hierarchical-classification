// Using CSV files exported from Python, we run the 'LEARNING' part which prints
// as result the matrix 'res'. This matrix can be then imported in Python to do
// the post-processing part.

// This implementation is around 10 times faster than in Python.
// Before compiling, change the compiler to C++11. In NetBeans:
// Project Properties --> Build --> C++ Compiler --> C++ Standard --> C++11

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>

using namespace std;

int pmatch(char x[], std::vector<char> y, unsigned int L1, unsigned int L2)
{
    int L    = L2-L1+1;
    int vect = 0;
    
    char z[L1];
    z[L1-1] = '\0';
    
    for(int i=0; i<L; i++){
        for(int j=0; j<L1-1; j++){
            z[j] = y[i+j];
        }
        vect += (strcmp(x,z)==0);
    }

    return vect;
}

int main(int argc, char** argv) {
    
    // -------------------------------------------------------------------------
    
    ifstream myfile;
    myfile.open("Train.csv");
    string value;
    int Rtrain = 0;
    int Ctrain = 0;
    while(myfile.good())
    {
        getline(myfile, value, '\n');
        Rtrain++;
        if(value.length() > Ctrain){
            Ctrain = value.length();
        }
    }
    myfile.close();        
    
    std::vector< std::vector<char> > train(Rtrain, std::vector<char>(Ctrain));
    for(int i=0; i<Rtrain; i++){
        for(int j=0; j<Ctrain; j++){
            train[i][j] = '\0';
        }
    }
    
    myfile.open("Train.csv");
    int i = 0;
    while(myfile.good())
    {
        getline(myfile, value, '\n');
        for(int j=0; j<value.length(); j++){
            train[i][j] = value[j];
        }
        i++;
    }
    myfile.close();
    
    // -------------------------------------------------------------------------
    
    myfile.open("Test.csv");
    int Rtest = 0;
    int Ctest = 0;
    while(myfile.good())
    {
        getline(myfile, value, '\n');
        Rtest++;
        if(value.length() > Ctest){
            Ctest = value.length();
        }
    }
    myfile.close();        
    
    std::vector< std::vector<char> > test(Rtest, std::vector<char>(Ctest));
    for(int i=0; i<Rtest; i++){
        for(int j=0; j<Ctest; j++){
            test[i][j] = '\0';
        }
    }
    
    myfile.open("Test.csv");
    i = 0;
    while(myfile.good())
    {
        getline(myfile, value, '\n');
        for(int j=0; j<value.length(); j++){
            test[i][j] = value[j];
        }
        i++;
    }
    myfile.close();
    
    // -------------------------------------------------------------------------
   
    
    int KK = 7;    
    
    int res[Rtest][KK*2];
    for(int i=0; i<Rtest; i++){
        for(int j=0; j<KK*2; j++){
            res[i][j] = 0;
        }
    }    

    char vet;
    char vetT;
    int  L;
    int  LT;
    std::vector<char> w;
    std::vector<char> wT;
    std::vector<char> wt;
    
    std::vector<int>  sum(Rtrain);
    std::vector<int>  index(Rtrain);
    
    char * tokenC;
    
    int Lte;
    
    for(int nn=0; nn<Rtest; nn++){
        
        vet = test[nn][0];
        L   = 0;
        
        while(vet != '\0'){
            L++;
            vet = test[nn][L];
        }
        
        w.resize(L+1);
        wt.resize(L);
        
        for(int i=0; i<L; i++){
            w[i] = test[nn][i];
            wt[i] = test[nn][i];
        }
        w[L] = ',';
        
        fill(sum.begin(), sum.end(), 0);        
        
        for(int mm=0; mm<Rtrain; mm++){
            
            vetT = train[mm][0];
            LT   = 0;
            
            while(vetT != '\0'){
                LT++;
                vetT = train[mm][LT];        
            }

            wT.resize(LT);
            
            for(int i=0; i<LT; i++){
                wT[i] = train[mm][i];        
            }
            
            // ----------------------------

            std::string   word(w.begin(),w.end());
            istringstream ss(word);
            string        token;    

            while(getline(ss, token, ',')) {
                
                tokenC = new char[token.length()+1];
                strcpy (tokenC, token.c_str());
                Lte = token.length();
                
                sum[mm] += pmatch(tokenC,wT,Lte+1,LT+1);
                
                delete tokenC;
                            
            }
            
            // ----------------------------
            
            wT.resize(LT+1);
            
            for(int i=0; i<LT; i++){
                wT[i] = train[mm][i];
            }
            wT[LT] = ',';
            
            // ----------------------------

            std::string   word2(wT.begin(),wT.end());
            istringstream ss2(word2);
            string        token2;
            
            while(getline(ss2, token2, ',')) {
                
                tokenC = new char[token2.length()+1];
                strcpy (tokenC, token2.c_str());
                Lte = token2.length();
                
                sum[mm] += pmatch(tokenC,wt,Lte+1,L+1);
                
                delete tokenC;
                            
            }
            
            // ----------------------------
            
        }
        
        for(int i=0; i<Rtrain; i++){
            index[i] = i;
        }
        sort (index.begin(), index.end(), [&sum](int i1, int i2) {return sum[i1] > sum[i2];});
    
        sort (sum.begin(), sum.end());
        reverse(sum.begin(), sum.end());
        
        for(int i=0; i<KK; i++){
            res[nn][i]    = sum[i];
            res[nn][i+KK] = index[i];
        }
        
        cout << "Iteration " << nn << '\n';
        
    }
    
    for(int i=0; i<Rtest; i++){
        for(int j=0; j<KK*2; j++){
            cout << res[i][j] << " ";
        }
        cout << '\n';
    }

}
