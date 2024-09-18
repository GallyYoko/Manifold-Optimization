function A = Retract(X,V,method) 
%{

Stiefel流形的收缩映射

参数: 
X-----------流形上的点
V-----------X点对应切空间上的切向量
method------收缩映射方式

输出:
A-----------流形上的收缩点

%}
if method == "qr" % 基于QR分解
    B = X+V;
    [Q,R] = qr(B,"econ");
    r = sign(diag(R));
    A = Q*diag(r);
elseif method == "svd" % 基于SVD分解
    B = X+V;
    [U,~,V] = svd(B,"econ");
    A = U*V';
elseif method == "polar" % 基于极分解
    B = X+V;
    C = V'*V;
    [p,~] = size(C);
    C = sqrtm(eye(p)+C);
    A = B/C;
elseif method == "cayley" % 基于Cayley变换
    [n,p] = size(X);
    W = eye(n)-0.5*(X*X');
    U = [W*V,X];
    Z = [X,-W*V];
    A = X+U/(eye(2*p)-0.5*(Z'*U))*Z'*X;
end