#include <bits/stdc++.h>
#define dec 0.991
using namespace std;
typedef long long ll;

double read()
{
	register int f=1,i=0;char c=getchar();
	while(c<'0'||c>'9') {if(c=='-'){f=-1;}c=getchar();}
	while(c>='0'&&c<='9') {i=(i<<3)+(i<<1)+c-'0';c=getchar();}
	return i*f*1.0;
}

int n, grp[50], a[50];
ll ans = 0, sum[5];

ll get_val() { return fabs(sum[0] - sum[1]);}

inline void change(int pos,int tag){
    sum[grp[pos]] -= a[pos];
    sum[tag] += a[pos];
    grp[pos] = tag;
}

inline void SA()
{
    double T = 1000;
    ll cur = ans;
    while(T>1e-10){
        int pos1 = rand() % n + 1, pos2 = rand() % n + 1;
        int ori1 = grp[pos1], ori2 = grp[pos2];
        change(pos1, ori2); change(pos2, ori1);
        ll temp = get_val();
        if( (temp-cur) < 0 ){
            cur = temp;
            if(cur < ans) ans = cur;
        }
        else if(exp((temp-cur)/T*(-1)) <= (rand() / RAND_MAX)){
            change(pos1, ori1);
            change(pos2, ori2);
        }
        T *= dec;
    }
}
int main()
{
    srand(time(0));
    int T = read();
    while(T--){
        n = read();
        for (int i = 1; i <= n;++i) a[i] = read();
        memset(grp, 0, sizeof(grp));
        memset(sum, 0, sizeof(sum));
        for (int i = 1; i <= n;++i){
            grp[i] = i % 2;
            sum[grp[i]] += a[i];
        }
        ans = get_val();
        for (int i = 1; i <= 100;++i) SA();
        cout << ans << endl;
    }
    return 0;
}