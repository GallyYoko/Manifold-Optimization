function t = Monotopy_Backtrack(func,gradfunc,x,t0,rho,c,iteration,method)
%{

基于单调线搜索的回退法

参数: 
func--------光滑函数
gradfunc----光滑函数func的梯度
x-----------流形上的点
t0----------初始步长
rho---------回退幅度
c-----------Armijo条件的判据
iteration---最大迭代次数
method------收缩映射的方式

输出:
t-----------所求步长

%}
t = t0;
for i = 1:iteration
    if func(Retract(x,t*v,method)) <= func(x)+c*t*sum(dot(gradfunc(x),v))
        break;
    end
    t = rho*t;
end