function Regularized_Newton_Method(func,gradfunc,hessfunc,x0,sigma0,gamma,c,ita1,ita2,gamma1,gamma2,kappa,iteration1,iteration2,method,iteration3,epsilon)
%{

正则化牛顿法

参数: 
func--------光滑函数
gradfunc----光滑函数func的梯度
hessfunc----光滑函数func的Hessian矩阵
x0----------流形上的初始点
sigma0------正则参数
gamma-------回退幅度
c-----------Armijo条件的判据
ita1--------正则化调整下限参数
ita2--------正则化调整上限参数
gamma1------正则化缩小幅度
gamma2------正则化增大幅度
kappa-------修正共轭梯度法调整阈值
iteration1--修正共轭梯度法的最大迭代次数
iteration2--回退法的最大迭代次数
method------收缩映射方式
iteration3--正则化牛顿法的最大迭代次数
epsilon-----梯度阈值

%}
x = x0;
v = -gradfunc(x0);
sigma = sigma0;
results = zeros(iteration3,1);
for i = 1:iteration3
    if norm(v,"fro") <= epsilon
        break;
    end
    B = @(y) hessfunc(x,y)+sigma*y;
    d = Modified_Conjugate_Gradient(B,-v,kappa,iteration1);
    t = 1;
    for j = 1:iteration2
        if func(Retract(x,t*d,method)) <= func(x)-c*t*trace(d'*v)
            break;
        end
        t = gamma*t;
    end
    y = Retract(x,t*d,method);
    m = func(x)-trace(v'*(t*d))+0.5*trace(hessfunc(x,t*d)'*(t*d))+0.5*sigma*norm(t*d,"fro")^2;
    rho = (func(y)-func(x))/(m-func(x));
    if rho >= ita2
        sigma = gamma1*sigma;
        x = y;
    elseif rho >= ita1
        x = y;
    else
        sigma = gamma2*sigma;
    end
    results(i,1) = func(x);
end
plot(results);