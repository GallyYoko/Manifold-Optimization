%% 固定问题相关参数
n = 10;
p = 3;
seed = 5;

%% 生成二次函数及其梯度、黎曼梯度
[A,B] = Random_Coefficient(n,p,seed);
func = @(x) trace(x'*A*x-2*x'*B);
nablafunc = @(x) 2*A*x-2*B;
gradfunc = @(x) nablafunc(x)-x*(x'*nablafunc(x)+nablafunc(x)'*x)/2;

%% 固定算法相关参数
x0 = Random_Initial(n,p,seed);
t0 = 10;
rho = 0.5;
c = 0.001;
iteration1 = 1000;
method = "qr";
iteration2 = 1000;
epsilon = 10e-8;

%% 使用算法
Gradient_Descent_N(func,gradfunc,x0,t0,10,rho,c,iteration1,method,iteration2,epsilon);