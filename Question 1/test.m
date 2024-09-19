%% 固定问题相关参数
n = 10;
p = 3;
seed = 5;

%% 生成二次函数及其梯度、Hessian矩阵
[A,B] = Random_Coefficient(n,p,seed);
func = @(x) trace(x'*A*x-2*x'*B);
projection = @(x,v) v-x*(x'*v+v'*x)/2;
nablafunc = @(x) 2*A*x-2*B;
gradfunc = @(x) projection(x,nablafunc(x));
nabla2func = @(x,v) 2*A*v;
hessfunc = @(x,v) projection(x,nabla2func(x,v)-v*(x'*nablafunc(x)+nablafunc(x)'*x)/2);

%% 固定算法相关参数
x0 = Random_Initial(n,p,seed);
sigma0 = 10;
gamma = 0.5;
c = 0.001;
ita1 = 0.2;
ita2 = 0.8;
gamma1 = 0.5;
gamma2 = 2;
kappa = 0.5;
iteration1 = 100;
iteration2 = 100;
method = "qr";
iteration3 = 1000;
epsilon = 10e-8;

%% 使用算法
Regularized_Newton_Method(func,gradfunc,hessfunc,x0,sigma0,gamma,c,ita1,ita2,gamma1,gamma2,kappa,iteration1,iteration2,method,iteration3,epsilon);