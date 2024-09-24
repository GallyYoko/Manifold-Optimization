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
t0 = 0.02;
alpha0 = 1;
rho = 0.5;
c = 0.001;
iteration1 = 50;
method = "qr";
iteration2 = 60;
epsilon = 100*eps;
alphamin = 0.01;
alphamax = 100;
M = 5;

%% 计算最小值的精确值
[exa,exa_x] = Exact_Value(A,B);

%% 使用算法
[e1,ex1] = Gradient_Descent_N(func,gradfunc,x0,t0,M,rho,c,iteration1,method,iteration2,epsilon,exa,exa_x);
[e2,ex2] = BB_Method(func,gradfunc,x0,alpha0,M,alphamax,alphamin,rho,c,iteration1,method,iteration2,epsilon,exa,exa_x);


subplot(1,2,1)
semilogy(1:iteration2, e1)
hold on
semilogy(1:iteration2, e2)
legend("梯度下降算法","BB算法")
xlabel("迭代次数")
ylabel("函数值的相对误差")

subplot(1,2,2)
semilogy(1:iteration2, ex1)
hold on
semilogy(1:iteration2, ex2)
legend("梯度下降算法","BB算法")
xlabel("迭代次数")
ylabel("点列关于极小值点的相对误差")