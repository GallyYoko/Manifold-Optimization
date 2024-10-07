function [result,x] = Exact_Value(A,B)
func = @(x) trace(x'*A*x-2*x'*B);
nablafunc = @(x) 2*A*x-2*B;
gradfunc = @(x) nablafunc(x)-x*(x'*nablafunc(x)+nablafunc(x)'*x)/2;
x0 = zeros(size(B));
alpha0 = 1;
rho = 0.5;
c = 0.001;
iteration1 = 50;
method = "qr";
iteration2 = 1000;
epsilon = 100*eps;
M = 1;
[n,p] = size(x0);
X = zeros(n,p,M);
x1 = x0;
alpha = alpha0;
alphamax = 100;
alphamin = 0.01;
X(:,:,1) = x1;
v1 = -gradfunc(x1);
results = zeros(iteration2,1);
for i = 1:iteration2
    x = x1;
    v = v1;
    if norm(v,"fro") <= epsilon
        break;
    end
    min_i = min(i,M);
    for j = 1:iteration1
        judgement = (func(Retract(x,alpha*v,method))<=func(X(:,:,1))-c*alpha*norm(v,"fro")^2);
        for ind = 1:min_i % 获得非单调线搜索的判别准则
            judgement = judgement|(func(Retract(x,alpha*v,method))<=func(X(:,:,ind))-c*alpha*norm(v,"fro")^2);
        end
        if judgement
            break;
        end
        alpha = rho*alpha;
    end
    x1 = Retract(x,alpha*v,method);
    X(:,:,mod(i,M)+1) = x1; % 更新非单调线搜索的迭代点列表
    results(i,1) = func(x1);
    v1 = -gradfunc(x1);
    s = x1-x;
    y = v-v1;
    if mod(i,2) == 0
        alphaABB = trace(s'*s)/trace(s'*y);
    else
        alphaABB = trace(s'*y)/trace(y'*y);
    end
    alpha = min(alphamax,max(alphamin,alphaABB));
end
result = func(x);