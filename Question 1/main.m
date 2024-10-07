function Opt = main

    %% 固定问题相关参数

    n = 2000;
    p = 10;
    alpha = 100;
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
    Ipt.x0 = u0;

    %% 固定算法相关参数

    Ipt.sigma0 = 10;                         % 正则参数
    Ipt.gamma = 0.2;                         % 回退幅度
    Ipt.c = 0.001;                           % Armijo条件的判据
    Ipt.eta1 = 0.01;                         % 正则化调整下限参数
    Ipt.eta2 = 0.9;                          % 正则化调整上限参数
    Ipt.gamma1 = 0.2;                        % 正则化缩小幅度
    Ipt.gamma2 = 10;                         % 正则化增大幅度
    Ipt.kappa = 0.01;                        % 修正共轭梯度法调整阈值
    Ipt.method = "cayley";                   % 收缩映射方式
    Ipt.max_iter = 300;                      % 正则化牛顿法的最大迭代次数
    Ipt.epsilon = 10e-5;                     % 梯度阈值
    
    Func.f = @func;                          % 待优化函数
    Func.gf = @gradfunc;                     % 黎曼梯度
    Func.hf = @hessfunc;                     % 黎曼海森算子

    %% 调用算法

    Opt = Regularized_Newton(Func,Ipt);

    %% 定义各函数

    % 正则化牛顿算法
    function Opt = Regularized_Newton(Func,Ipt)
        
        Opt.x = {Ipt.x0};                        % 流形上的迭代点
        Opt.f = {Func.f(Ipt.x0)};                % 函数值
        Opt.gf = {Func.gf(Ipt.x0)};              % 黎曼梯度
        Opt.s = {Ipt.sigma0};                    % 正则参数
        Opt.sum_iter = 0;                        % 总迭代次数
        Opt.time = 0;                            % 花费时间
        Opt.iter = 0;                            % 迭代次数(含初始点)
        
        % 主循环体
        time_start = tic;
        for i = 1:Ipt.max_iter
            x = Opt.x{i};
            gf = Opt.gf{i};
            sigma = Opt.s{i};
    
            % 若黎曼梯度很小, 跳出循环
            if norm(gf,"fro") <= Ipt.epsilon
                break;
            end
    
            % 计算下降方向
            B = @(y) Func.hf(x,y)+sigma*y;
            [d, iter] = Modified_Conjugate_Gradient(B,gf,Ipt.kappa);
            Opt.sum_iter = Opt.sum_iter+iter;
    
            % 使用回退法确定步长
            t = 1;
            y = Retraction(x,t*d,Ipt.method);
            while Func.f(y) > Func.f(x)+Ipt.c*t*trace(d'*gf)
                t = Ipt.gamma*t;
                y = Retraction(x,t*d,Ipt.method);
            end
    
            % 计算ratio
            m = Func.f(x)+trace(gf'*(t*d))...
                +0.5*trace(Func.hf(x,t*d)'*(t*d))...
                +0.5*sigma*norm(t*d,"fro")^2;
            rho = (Func.f(y)-Func.f(x))/(m-Func.f(x));
    
            % 修改正则参数
            if rho >= Ipt.eta2
                Opt.s{i+1} = Ipt.gamma1*sigma;
            elseif rho >= Ipt.eta1
                Opt.s{i+1} = sigma;
            else
                Opt.s{i+1} = Ipt.gamma2*sigma;
            end
    
            % 更新迭代点
            if rho >= Ipt.eta1
                Opt.x{i+1} = y;
            else
                Opt.x{i+1} = x;
            end
    
            % 计算函数值和梯度值
            Opt.f{i+1} = func(Opt.x{i+1});
            Opt.gf{i+1} = gradfunc(Opt.x{i+1});
        end
        Opt.time = toc(time_start);
        Opt.iter = i;
    end

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
    function n2f = nabla2func(x,v)
        rhox = sum(x.^2, 2); % diag(x*x');
        rhoxdot = 2*sum(x.*v, 2);
        tempa = Lu\(Ll\rhoxdot);
        tempa = alpha*tempa;
        tempb = Lu\(Ll\rhox);
        tempb = alpha*tempb;
        n2f = L*v + bsxfun(@times,tempa,x) + bsxfun(@times, tempb, v);
    end
    
    % 黎曼海森
    function hf = hessfunc(x,v)
        hf = projection(x,nabla2func(x,v)...
            -v*(x'*nablafunc(x)+nablafunc(x)'*x)/2);
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

        ep = min(0.5,sqrt(norm(gf,"fro")))...
            *norm(gf,"fro");                     % 判断参数
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