function x = Random_Initial(n,p,seed)
%{

随机生成Stiefel流形上的点

参数:
n,p---------矩阵尺寸
seed--------随机种子

输出:
x-----------Stiefel流形上的点

%}
rng(seed);
x = rand(n,p);
x = Retract(x,zeros(n,p),"qr");