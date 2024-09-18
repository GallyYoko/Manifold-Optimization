function [A,B] = Random_Coefficient(n,p,seed)
%{

随机生成Stiefel流形上二次函数的系数矩阵

参数:
n,p---------矩阵尺寸
seed--------随机种子

输出:
A,B---------二次函数tr(X'*A*X-2X'*B)的系数

%}
rng(seed);
A = rand(n,n);
a = rand(n);
A = A'*A+diag(a);
B = rand(n,p);
