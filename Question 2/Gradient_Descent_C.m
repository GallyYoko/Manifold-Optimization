function [err,err_x] = Gradient_Descent_C(func,gradfunc,x0,t0,rho,vrho,c1,iteration1,method,iteration2,epsilon,exa,exa_x)
%{

基于凸组合非单调线搜索回退法的梯度下降法

参数: 
func--------光滑函数
gradfunc----光滑函数func的梯度
x0----------流形上的初始点
t0----------初始步长
rho---------回退幅度
vrho--------记忆元参数
c1----------Armijo条件的判据
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
x = x0;
q = 1;
c = func(x);
err = zeros(iteration2,1);
err_x = zeros(iteration2,1);
for i = 1:iteration2
    v = -gradfunc(x);
    if norm(v,"fro") <= epsilon
        break;
    end
    t = t0;
    for j = 1:iteration1
        if func(Retract(x,t*v,method)) <= c-c1*t*norm(v,"fro")
            break;
        end
        t = rho*t;
    end
    x = Retract(x,t*v,method);
    c = (vrho*q*c+func(x))/(vrho*q+1);
    q = vrho*q+1;
    err(i,1) = abs(func(x)-exa)/abs(exa);
    err_x(i,1) = norm(x-exa_x,"fro");
end