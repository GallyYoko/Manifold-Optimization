%% 固定问题相关参数
n = 500;
p = 50;
seed = 99;

%% 生成二次函数及其梯度、黎曼梯度
[A,B] = Random_Coefficient(n,p,seed);
func = @(x) trace(x'*A*x-2*x'*B);
nablafunc = @(x) 2*A*x-2*B;
gradfunc = @(x) nablafunc(x)-x*(x'*nablafunc(x)+nablafunc(x)'*x)/2;

%% 固定算法相关参数
x0 = Random_Initial(n,p,seed);
t0 = 1;
alpha0 = 1;
rho = 0.5;
c = 0.001;
iteration1 = 50;
method = "qr";
iteration2 = 100;
epsilon = 100*eps;
alphamin = 0.01;
alphamax = 100;

%% 计算最小值的精确值
exa = Exact_Value(A,B);

%% 使用算法
BB_Method(func,gradfunc,x0,alpha0,5,alphamax,alphamin,rho,c,iteration1,method,iteration2,epsilon,exa);