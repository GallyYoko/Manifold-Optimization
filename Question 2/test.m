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
[err,err_x] = Gradient_Descent_N(func,gradfunc,x0,t0,M,rho,c,iteration1,method,iteration2,epsilon,exa,exa_x);
semilogy(1:iteration2,err);

%{
subplot(2,2,1)
semilogy(1:25, e11(1:25,1))
hold on
semilogy(1:25, e12(1:25,1))
hold on
semilogy(1:25, e13(1:25,1))
hold on
semilogy(1:25, e14(1:25,1))
legend("QR分解","SVD分解","极分解","Cayley变换")
xlabel("迭代次数")
ylabel("函数值的相对误差")
title("梯度下降算法")

subplot(2,2,2)
semilogy(1:iteration2, ex11)
hold on
semilogy(1:iteration2, ex12)
hold on
semilogy(1:iteration2, ex13)
hold on
semilogy(1:iteration2, ex14)
legend("QR分解","SVD分解","极分解","Cayley变换")
xlabel("迭代次数")
ylabel("点列的相对误差")
title("梯度下降算法")

subplot(2,2,3)
semilogy(1:25, e21(1:25,1))
hold on
semilogy(1:25, e22(1:25,1))
hold on
semilogy(1:25, e23(1:25,1))
hold on
semilogy(1:25, e24(1:25,1))
legend("QR分解","SVD分解","极分解","Cayley变换")
xlabel("迭代次数")
ylabel("函数值的相对误差")
title("BB算法")

subplot(2,2,4)
semilogy(1:iteration2, ex21)
hold on
semilogy(1:iteration2, ex22)
hold on
semilogy(1:iteration2, ex23)
hold on
semilogy(1:iteration2, ex24)
legend("QR分解","SVD分解","极分解","Cayley变换")
xlabel("迭代次数")
ylabel("点列的相对误差")
title("BB算法")
%}