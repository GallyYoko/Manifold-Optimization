function result = Exact_Value(A,B)
func = @(x) trace(x'*A*x-2*x'*B);
nablafunc = @(x) 2*A*x-2*B;
gradfunc = @(x) nablafunc(x)-x*(x'*nablafunc(x)+nablafunc(x)'*x)/2;
x0 = zeros(size(B));
t0 = 10;
rho = 0.5;
c = 0.001;
iteration1 = 100;
method = "qr";
iteration2 = 5000;
epsilon = 10e-8;
M = 5;
[n,p] = size(x0);
X = zeros(n,p,M);
x = x0;
X(:,:,1) = x;
for i = 1:iteration2
    v = -gradfunc(x);
    if norm(v,"fro") <= epsilon
        break;
    end
    t = t0;
    min_i = min(i,M);
    for j = 1:iteration1
        judgement = (func(Retract(x,t*v,method))<=func(X(:,:,1))-c*t*norm(v,"fro")^2);
        for ind = 1:min_i % 获得非单调线搜索的判别准则
            judgement = judgement|(func(Retract(x,t*v,method))<=func(X(:,:,ind))-c*t*norm(v,"fro")^2);
        end
        if judgement
            break;
        end
        t = rho*t;
    end
    x = Retract(x,t*v,method);
    X(:,:,mod(i,M)+1) = x; % 更新非单调线搜索的迭代点列表
    result = func(x);
end