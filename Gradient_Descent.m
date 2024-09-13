function Gradient_Descent(func,gradfunc,x0,t0,rho,c,iteration1,method,iteration2,epsilon)
%{

基于单调线搜索回退法的梯度下降法

参数: 
func--------光滑函数
gradfunc----光滑函数func的梯度
x0----------流形上的初始点
t0----------初始步长
rho---------回退幅度
c-----------Armijo条件的判据
iteration1--回退法的最大迭代次数
method------收缩映射的方式
iteration2--梯度下降法的最大迭代次数
epsilon-----梯度阈值

%}
x = x0;
for i = 1:iteration2
    if norm(gradfunc(x),"fro") <= epsilon
        break;
    end
    t = Monotopy_Backtrack(func,gradfunc,x,t0,rho,c,iteration1,method);
    x = Retract(x,-t*gradfunc(x),method);
end