function d = Modified_Conjugate_Gradient(B,g,kappa,iteration)
%{

适用于非正定对称阵的修正共轭梯度法

参数: 
B-----------系数函数
g-----------等式右侧的矩阵
kappa-------(0,1)中的参数
iteration---迭代次数

输出:
d-----------方程Bd=-g的解

%}
epsilon = min(0.5,norm(g,"fro")^0.5)*norm(g,"fro");
z = zeros(size(g));
r = -g;
p = -g;
for i = 1:iteration
    if trace(p'*B(p)) <= 0
        if i == 1
            d = -g;
        else
            d = z;
        end
        break;
    end
    a = trace(r'*r)/trace(p'*B(p));
    z1 = z+a*p;
    r1 = r-a*B(p);
    if -trace(z1'*g) < kappa*norm(g,"fro")*norm(z1,"fro")
        d = z;
        break;
    end
    if r1 < epsilon
        d = z1;
        break;
    end
    b = trace(r1'*r1)/trace(r'*r);
    p = r1+b*p;
    z = z1;
    r = r1;
    d = z;
end