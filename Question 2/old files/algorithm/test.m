%% 固定问题相关参数
n = 500;
p = 50;
seed = 99;
density = 0.5;

%% 生成二次函数及其梯度、黎曼梯度
[A,B] = Random_Coefficient(n,p,seed,density);
exa = 0;
exa_x = 0;
func = @(x) trace(x'*A*x-2*x'*B);
nablafunc = @(x) 2*A*x-2*B;
gradfunc = @(x) nablafunc(x)-x*(x'*nablafunc(x)+nablafunc(x)'*x)/2;

%% 固定算法相关参数
x0 = Random_Initial(n,p,seed);
t0 = 1;
alpha0 = 0.02;
rho = 0.5;
vrho = 0.5;
c = 0.001;
iteration1 = 50;
method = "qr";
iteration2 = 1;
epsilon = 10e-10;
alphamin = 0.001;
alphamax = 1000;
M = 1;
slot = 0;

%% 计算最小值的精确值
% [exa,exa_x] = Exact_Value(A,B);

%% 使用算法
[err,alpha_list,x2] = BB_Method(func,gradfunc,x0,alpha0,M,alphamax,...
    alphamin,rho,c,iteration1,method,iteration2,epsilon,slot,exa,exa_x);

%% 不同BB步长的影响
%{
alpha = zeros(iteration2,3);
e = zeros(iteration2,3);
ex = zeros(iteration2,3);
for i = 1:3
    [e(:,i),ex(:,i),alpha(:,i)] = BB_Method(func,gradfunc,x0,alpha0,M,alphamax,alphamin,rho,c,iteration1,method,iteration2,epsilon,i-2,exa,exa_x);
end

subplot(1,3,1)
semilogy(1:20,e(1:20,:))
xlabel("迭代次数")
ylabel("函数值的相对误差")
legend("短BB步长","交替BB步长","长BB步长")
title("函数值的相对误差变化");
subplot(1,3,2)
semilogy(1:30,ex(1:30,:))
xlabel("迭代次数")
ylabel("自变量的误差")
legend("短BB步长","交替BB步长","长BB步长")
title("自变量的误差变化");
subplot(1,3,3)
plot(1:iteration2,alpha)
xlabel("迭代次数")
ylabel("BB步长")
legend("短BB步长","交替BB步长","长BB步长")
title("BB步长变化");
%}

%% M对算法的影响
%{
e0 = zeros(50,5);
ex0 = zeros(50,5);
e = zeros(iteration2,5);
ex = zeros(iteration2,5);
for i = 1:4
    [e0(:,i),ex0(:,i)] = Gradient_Descent_N(func,gradfunc,x0,0.02,2*i-1,rho,c,iteration1,method,50,epsilon,exa,exa_x);
    [e(:,i),ex(:,i)] = Gradient_Descent_N(func,gradfunc,x0,0.1,2*i-1,rho,c,iteration1,method,iteration2,epsilon,exa,exa_x);
    disp(i);
end
[e0(:,5),ex0(:,5)] = Gradient_Descent_C(func,gradfunc,x0,0.02,rho,vrho,c,iteration1,method,50,epsilon,exa,exa_x);
[e(:,5),ex(:,5)] = Gradient_Descent_C(func,gradfunc,x0,0.1,rho,vrho,c,iteration1,method,iteration2,epsilon,exa,exa_x);

subplot(2,2,1)
semilogy(1:50,e0)
xlabel("迭代次数")
ylabel("函数值误差")
title("t_0=0.02时函数值误差的变化")
legend("M=1","M=3","M=5","M=7","凸组合");

subplot(2,2,2)
semilogy(1:50,ex0)
xlabel("迭代次数")
ylabel("自变量误差")
title("t_0=0.02时自变量误差的变化")
legend("M=1","M=3","M=5","M=7","凸组合")

subplot(2,2,3)
semilogy(1:iteration2,e)
xlabel("迭代次数")
ylabel("函数值误差")
title("t_0=0.1时函数值误差的变化")
legend("M=1","M=3","M=5","M=7","凸组合");

subplot(2,2,4)
semilogy(1:iteration2,ex)
xlabel("迭代次数")
ylabel("自变量误差")
title("t_0=0.1时自变量误差的变化")
legend("M=1","M=3","M=5","M=7","凸组合")
%}

%% 步长对算法的影响
%{
k = 30;
delta = 100;
e1 = zeros(k,1);
e2 = zeros(k,1);
e3 = zeros(k,1);
for i = 1:k
    [~,temp] = Gradient_Descent_N(func,gradfunc,x0,0.01+0.003*i,1,rho,c,iteration1,"qr",iteration2,epsilon,exa,exa_x);
    e1(i,1) = mean(temp(iteration2-delta:iteration2,1));

    [~,temp] = Gradient_Descent_N(func,gradfunc,x0,0.01+0.003*i,5,rho,c,iteration1,"qr",iteration2,epsilon,exa,exa_x);
    e2(i,1) = mean(temp(iteration2-delta:iteration2,1));

    [~,temp] = Gradient_Descent_C(func,gradfunc,x0,0.01+0.003*i,rho,vrho,c,iteration1,"qr",iteration2,epsilon,exa,exa_x);
    e3(i,1) = mean(temp(iteration2-delta:iteration2,1));

    disp(i);
end
semilogy(0.01+0.003*(1:k),[e1,e2,e3])
xlabel("初始步长")
ylabel("自变量误差")
legend("单调线搜索","Grippo线搜索","凸组合线搜索")
%}