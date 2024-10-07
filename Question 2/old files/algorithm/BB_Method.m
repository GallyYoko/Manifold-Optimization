function [err,alpha_list] = BB_Method(func,gradfunc,x0,alpha0,M,alphamax,alphamin,rho,c,iteration1,method,iteration2,epsilon,slot,exa,exa_x)
%{

基于非单调线搜索回退法的梯度下降法

参数: 
func--------光滑函数
gradfunc----光滑函数func的梯度
x0----------流形上的初始点
alpha0------初始步长
M-----------非单调线搜索列表长度
alphamax----BB步长上界
alphamin----BB步长下界
rho---------回退幅度
c-----------Armijo条件的判据
iteration1--回退法的最大迭代次数
method------收缩映射的方式
iteration2--梯度下降法的最大迭代次数
slot--------决定是否使用交替步长
epsilon-----梯度阈值
exa---------函数值精确值
exa_x-------自变量精确值

输出:
err---------函数值相对误差
err_x-------自变量误差
alpha_list--BB步长

%}
    [n,p] = size(x0);
    X = zeros(n,p,M);
    x1 = x0;
    alpha = alpha0;
    X(:,:,1) = x1;
    v1 = -gradfunc(x1);
    err = zeros(iteration2,1);
    alpha_list = zeros(iteration2,1);
    for i = 1:iteration2
        x = x1;
        v = v1;
        if norm(v,"fro") <= epsilon
            break;
        end
        min_i = min(i,M);
        for j = 1:iteration1
            %judgement = (func(Retract(x,alpha*v,method))<=func(X(:,:,1))-c*alpha*norm(v,"fro")^2);
            judgement = 0;
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
        err(i,1) = abs(func(x1)-exa)/abs(exa);
        v1 = -gradfunc(x1);
        err(i,1) = norm(v1,"fro");
        s = x1-x;
        y = v-v1;
        alphaSBB = trace(s'*y)/trace(y'*y);
        alphaLBB = trace(s'*s)/trace(s'*y);
        if slot == 0 % 使用交替步长
            if mod(i,2) == 0 
                alphaABB = trace(s'*s)/trace(s'*y);
            else
                alphaABB = trace(s'*y)/trace(y'*y);
            end
        elseif slot == 1 % 使用长步长
            alphaABB = alphaLBB; 
        elseif slot == -1 % 使用短步长
            alphaABB = alphaSBB; 
        end
        alpha = min(alphamax,max(alphamin,alphaABB));
        alpha_list(i,1) = alpha;
    end
end