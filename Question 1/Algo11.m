%% 运行
n = 5000;
p = 30;
alphalist = [1,10,100];
formatSpec = "iter = %d, sum_iter = %d, nrmG = %.2e, time = %.2f\n";
for alpha = alphalist
    [GF, t2, iter, sum_iter] = main(n, p, alpha);
    gf = norm(GF{iter},"fro");
    fprintf(formatSpec, iter-1, sum_iter, gf, t2);
end

%% 主体部分

function [GF, t2, i, sum_iter] = main(n, p, alpha)
    %% 固定问题相关参数
    seed = 2010;
    L = gallery('tridiag', n, -1, 2, -1);
    [Ll,Lu] = lu(L);

    %% 获得初始点

    RandStream.setGlobalStream(RandStream('mt19937ar','seed',seed));
    x = randn(n, p);
    [u, ~, v] = svd(x, 0);
    x_init = u*v';
    tempM2 = alpha*(L\(sum(x_init.^2,2)));
    tempM2 = spdiags(tempM2,0,n,n);
    tempM = L + tempM2;
    [u0, ~, ~] = eigs(tempM, p,'sm');
    x0 = u0;

    %% 固定算法相关参数

    sigma0 = 10;                             % 正则参数
    gamma = 0.2;                             % 回退幅度
    c = 0.001;                               % Armijo条件的判据
    eta1 = 0.01;                             % 正则化调整下限参数
    eta2 = 0.9;                              % 正则化调整上限参数
    gamma1 = 0.2;                            % 正则化缩小幅度
    gamma2 = 10;                             % 正则化增大幅度
    kappa = 0.01;                            % 修正共轭梯度法调整阈值
    method = "cayley";                       % 收缩映射方式
    max_iter = 300;                          % 正则化牛顿法的最大迭代次数
    epsilon = 10e-5;                         % 梯度阈值
    
    %% 记录迭代过程中的变量变化

    X = {x0};                                % 流形上的迭代点
    F = {func(x0)};                          % 函数值
    GF = {gradfunc(x0)};                     % 黎曼梯度
    S = {sigma0};                            % 正则参数
    sum_iter = 0;                            % 总迭代次数

    %% 正则化牛顿算法
    
    t1 = tic;
    for i = 1:max_iter
        x = X{i};
        gf = GF{i};
        sigma = S{i};

        % 若黎曼梯度很小, 跳出循环
        if norm(gf,"fro") <= epsilon
            break;
        end

        % 计算下降方向
        B = @(y) hessfunc(x,y)+sigma*y;
        [d, iter] = Modified_Conjugate_Gradient(B,gf,kappa);
        sum_iter = sum_iter+iter;

        % 使用回退法确定步长
        t = 1;
        y = Retraction(x,t*d,method);
        while func(y) > func(x)+c*t*trace(d'*gf)
            t = gamma*t;
            y = Retraction(x,t*d,method);
        end

        % 计算ratio
        m = func(x)+trace(gf'*(t*d))+0.5*trace(hessfunc(x,t*d)'*(t*d))+...
            0.5*sigma*norm(t*d,"fro")^2;
        rho = (func(y)-func(x))/(m-func(x));

        % 修改正则参数
        if rho >= eta2
            S{i+1} = gamma1*sigma;
        elseif rho >= eta1
            S{i+1} = sigma;
        else
            S{i+1} = gamma2*sigma;
        end

        % 更新迭代点
        if rho >= eta1
            X{i+1} = y;
        else
            X{i+1} = x;
        end

        % 计算函数值和梯度值
        F{i+1} = func(X{i+1});
        GF{i+1} = gradfunc(X{i+1});
    end
    t2 = toc(t1);

    %% 定义各函数

    % 待优化函数
    function f = func(x)
        Lx = L*x;
        rhox = sum(x.^2, 2); % diag(x*x');
        tempa = Lu\(Ll\rhox); tempa = alpha*tempa;
        f = 0.5*sum(sum(x.*(Lx))) + 1/4*(rhox'*tempa);
    end
    
    % 投影映射
    function p = projection(x,v)
        p = v-x*(x'*v+v'*x)/2;
    end
    
    % 欧氏梯度
    function nf = nablafunc(x)
        rhox = sum(x.^2, 2); % diag(x*x');
        tempa = Lu\(Ll\rhox); tempa = alpha*tempa;
        nf = L*x + bsxfun(@times,tempa,x);
    end
    
    % 黎曼梯度
    function gf = gradfunc(x)
        gf = projection(x,nablafunc(x));
    end
    
    % 欧式海森
    function n2f = nabla2func(x, v)
        rhox = sum(x.^2, 2); % diag(x*x');
        rhoxdot = 2*sum(x.*v, 2);
        tempa = Lu\(Ll\rhoxdot);
        tempa = alpha*tempa;
        tempb = Lu\(Ll\rhox);
        tempb = alpha*tempb;
        n2f = L*v + bsxfun(@times,tempa,x) + bsxfun(@times, tempb, v);
    end
    
    % 黎曼海森
    function hf = hessfunc(x, v)
        hf = projection(x,nabla2func(x,v)- ...
            v*(x'*nablafunc(x)+nablafunc(x)'*x)/2);
    end

    % 收缩映射
    function A = Retraction(X,V,method) 
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
    end

    % 修正共轭梯度法
    function [d, k] = Modified_Conjugate_Gradient(B,gf,kappa)
        ep = min(0.5,sqrt(norm(gf,"fro")))*...
            norm(gf,"fro");                      % 判断参数
        Z = {zeros(n,p)};                        % 共轭梯度法的解
        R = {-gf};                               % 残差
        PD = {-gf};                              % 共轭梯度法下降方向
        % 主要循环体
        k = 1;
        while true
            z = Z{k};
            r = R{k};
            pd = PD{k};
            Bpd = B(pd);
            % 判断是否为负曲率方向
            if trace(pd'*Bpd) <= 0
                if k == 1
                    d = -gf;
                else
                    d = z;
                end
                break;
            end
            % 计算步长
            a = trace(r'*r)/trace(pd'*Bpd);
            z1 = z+a*pd;
            r1 = r-a*Bpd;
            % 判断解与黎曼梯度方向的偏差是否过大
            if -trace(z1'*gf) < kappa*norm(gf,"fro")*norm(z1,"fro")
                d = z;
                break;
            end
            % 判断新残差是否足够小
            if norm(r1, "fro") < ep
                d = z1;
                break;
            end
            b = trace(r1'*r1)/trace(r'*r);
            k = k+1;
            PD{k} = r1+b*pd;
            Z{k} = z1;
            R{k} = r1;
        end
    end

end