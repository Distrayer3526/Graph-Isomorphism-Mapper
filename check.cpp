// make sure you have both files graphs and ans in the same folder
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <set>
#include <cmath>
using namespace std;

int n, m;
set<pair<int,int> > h;
int pi[222221], o[222221];

double f(int A, int B) {
	double x = A * 1.0 / B;
	return 5.333 * x * x * x - 4 * x * x + 2.667 * x;
}

int main() {
	ifstream fin("graphs");
	ifstream ans("ans");

	fin >> n >> m;

	for(int i = 1; i <= n; i++) 
		ans >> pi[i];
	ans.close();
	for(int i = 1; i <= n; i++)
		o[pi[i]] = i;

	for(int i = 0; i < m; i++) {
		int a, b;
		fin >> a >> b;
		if(a > b) swap(a, b);
		h.insert(make_pair(a, b));
	}

	int tot = 0;
	for(int i = 0; i < m; i++) {
		int a, b;
		fin >> a >> b;
		a = o[a];
		b = o[b];
		if(a > b) swap(a, b);
		if(h.find(make_pair(a,b)) != h.end())
			tot ++;
	}
	fin.close();

	cout << "you match " << tot << " edges out of " << m << " edges" << endl;
	cout << "your score is " << f(tot, m) << endl;

	return 0;
}