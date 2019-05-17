#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <math.h>
#include <algorithm>
#include <vector>
#include <queue>
#include <fstream>
using namespace std;
#define mem(a,b) memset(a,b,sizeof(a))
#define rep(i,a,b) for(int i = a;i < b;i++)
#define _rep(i,a,b) for(int i = a;i <= b;i++)
typedef long long ll;

const int numOfPix = 1000;

int main() {
	for(int num = 1;num <= 10;num++) {
		ofstream outFeature("Feature/"+to_string(num)+".txt");
		for(int i = 0;i < numOfPix;i++) {
			for(int j = 0;j < 64;j++) {
				outFeature << (double)rand()/RAND_MAX << ' ';
			}
			outFeature << endl;
		}

		ofstream outLabel("Label/"+to_string(num)+".txt");
		for(int i = 0;i < numOfPix;i++) {
			outLabel << rand()%18 << ' ';
		}

		ofstream outNeighbour("Neighbour/"+to_string(num)+".txt");
		for(int i = 0;i < numOfPix;i++) {
			for(int j = 0;j < numOfPix;j++) {
				outNeighbour << rand()%2 << ' ';
			}
			outNeighbour << endl;
		}

		ofstream outNumber("Number/"+to_string(num)+".txt");
		for(int i = 0;i < numOfPix;i++) {
			outNumber << rand()%numOfPix << ' ';
		}

		ofstream outSeq("Sequence/"+to_string(num)+".txt");
		vector<int> vec;
		for(int i = 0;i < numOfPix;i++) vec.push_back(i);
		next_permutation(vec.begin(),vec.end());
		for(auto x:vec) outSeq << x << ' ';
	}
    return 0;
}
