function [err,err_x] = Gradient_Descent_N(func,gradfunc,x0,t0,M,rho,c,iteration1,method,iteration2,epsilon,exa,exa_x)
%{

基于非单调线搜索回退法的梯度下降法

参数: 
func--------光滑函数
gradfunc----光滑函数func的梯度
x0----------流形上的初始点
t0----------初始步长
M-----------非单调线搜索列表长度
rho---------回退幅度
c-----------Armijo条件的判据
iteration1--回退法的最大迭代次数
method------收缩映射的方式
iteration2--梯度下降法的最大迭代次数
epsilon-----梯度阈值
exa---------函数值精确值
exa_x-------自变量精确值

输出:
err---------函数值相对误差
err_x-------自变量误差

%}
[n,p] = size(x0);
X = zeros(n,p,M);
x = x0;
X(:,:,1) = x;
results = zeros(iteration2,1);
err = zeros(iteration2,1);
err_x = zeros(iteration2,1);
for i = 1:iteration2
    v = -gradfunc(x);
    if norm(v,"fro") <= epsilon
        break;
    end
    t = t0;
    min_i = min(i,M);
    for j = 1:iteration1
        judgement = 0;
        for ind = 1:min_i % 获得非单调线搜索的判别准则
            judgement = judgement|(func(Retract(x,t*v,method))<=func(X(:,:,ind))-c*t*norm(v,"fro")^2);
        end
        if judgement
            break;
        end
        t = rho*t;
    end
    %disp(j);
    x = Retract(x,t*v,method);
    X(:,:,mod(i,M)+1) = x; % 更新非单调线搜索的迭代点列表
    results(i,1) = func(x);
    err(i,1) = abs(func(x)-exa)/abs(exa);
    err_x(i,1) = norm(x-exa_x,"fro");
end