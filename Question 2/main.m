function [Exa, Opt1, Opt2, Opt3] = main
    
    %% 固定问题相关参数
    
    n = 500;
    p = 50;
    seed = 99;
    density = 0.5;
    [A,B] = Random_Coefficient(n,p,seed,density);

    %% 获得初始点
    
    Ipt.x0 = Random_Initial(n,p,seed);

    %% 固定算法相关参数

    Ipt.t0 = 0.02;                           % 梯度下降法初始步长
    Ipt.alpha0 = 0.02;                       % BB算法初始步长
    Ipt.rho = 0.5;                           % 回退幅度
    Ipt.vrho = 0.5;                          % 凸组合法记忆元参数
    Ipt.c = 0.001;                           % Armijo条件的判据
    Ipt.max_iter1 = 50;                      % 回退法的最大迭代次数
    Ipt.method = "qr";                       % 收缩映射的方式
    Ipt.max_iter2 = 500;                     % 梯度下降法的最大迭代次数
    Ipt.epsilon = 10e-10;                    % 梯度阈值
    Ipt.alphamin = 0.001;                    % BB步长下界
    Ipt.alphamax = 1000;                     % BB步长上界
    Ipt.M = 5;                               % 非单调线搜索列表长度
    Ipt.slot = 0;                            % 决定是否使用交替步长

    Func.f = @func;                          % 待优化函数
    Func.gf = @gradfunc;                     % 黎曼梯度

    %% 调用算法

    Exa = Exact_Value(Func);
    Opt1 = Gradient_Descent_N(Func,Ipt);
    Opt2 = Gradient_Descent_C(Func,Ipt);
    Opt3 = BB_Method(Func,Ipt);

    %% 定义各函数

    % 非单调梯度下降法
    function Opt = Gradient_Descent_N(Func,Ipt)

        Opt.x = {Ipt.x0};                        % 流形上的迭代点
        Opt.f = {Func.f(Ipt.x0)};                % 函数值
        Opt.gf = {Func.gf(Ipt.x0)};              % 黎曼梯度
        Opt.time = 0;                            % 花费时间
        Opt.iter = 0;                            % 迭代次数(含初始点)

        % 主循环体
        time_start = tic;
        for i = 1:Ipt.max_iter2
            x = Opt.x{i};
            gf = Opt.gf{i};

            % 梯度范数到达阈值
            if norm(gf,"fro") <= Ipt.epsilon
                break;
            end

            % 非单调线搜索
            t = Ipt.t0;
            min_i = min(i,Ipt.M);
            for j = 1:Ipt.max_iter1
                judgement = 0;
                for ind = 1:min_i
                    judgement = judgement...
                        |(Func.f(Retraction(x,-t*gf,Ipt.method))...
                        <=Func.f(Opt.x{i-ind+1})-Ipt.c*t*norm(gf,"fro")^2);
                end
                if judgement
                    break;
                end
                t = Ipt.rho*t;
            end

            % 更新各参数
            x = Retraction(x,-t*gf,Ipt.method);
            Opt.x{i+1} = x;
            Opt.f{i+1} = Func.f(x);
            Opt.gf{i+1} = Func.gf(x);
        end
        Opt.time = toc(time_start);
        Opt.iter = i;
    end

    % 凸组合梯度下降法
    function Opt = Gradient_Descent_C(Func,Ipt)
        
        Opt.x = {Ipt.x0};                        % 流形上的迭代点
        Opt.f = {Func.f(Ipt.x0)};                % 函数值
        Opt.gf = {Func.gf(Ipt.x0)};              % 黎曼梯度
        Opt.time = 0;                            % 花费时间
        Opt.iter = 0;                            % 迭代次数(含初始点)

        % 主循环体
        time_start = tic;
        c = Opt.f{1};
        q = 1;
        for i = 1:Ipt.max_iter2
            x = Opt.x{i};
            gf = Opt.gf{i};

            % 梯度范数到达阈值
            if norm(gf,"fro") <= Ipt.epsilon
                break;
            end

            % 非单调线搜索
            t = Ipt.t0;
            for j = 1:Ipt.max_iter1
                if Func.f(Retraction(x,-t*gf,Ipt.method))...
                        <= c-Ipt.c*t*norm(gf,"fro")^2
                    break;
                end
                t = Ipt.rho*t;
            end

            % 更新各参数
            x = Retraction(x,-t*gf,Ipt.method);
            c = (Ipt.vrho*q*c+Func.f(x))/(Ipt.vrho*q+1);
            q = Ipt.vrho*q+1;
            Opt.x{i+1} = x;
            Opt.f{i+1} = Func.f(x);
            Opt.gf{i+1} = Func.gf(x);
        end
        Opt.time = toc(time_start);
        Opt.iter = i;
    end

    % BB算法
    function Opt = BB_Method(Func,Ipt)

        Opt.x = {Ipt.x0};                        % 流形上的迭代点
        Opt.f = {Func.f(Ipt.x0)};                % 函数值
        Opt.gf = {Func.gf(Ipt.x0)};              % 黎曼梯度
        Opt.alpha = {Ipt.alpha0};                % BB步长
        Opt.time = 0;                            % 花费时间
        Opt.iter = 0;                            % 迭代次数(含初始点)

        % 主循环体
        time_start = tic;
        for i = 1:Ipt.max_iter2
            x = Opt.x{i};
            gf = Opt.gf{i};
            alpha = Opt.alpha{i};

            % 梯度范数到达阈值
            if norm(gf,"fro") <= Ipt.epsilon
                break;
            end

            % 非单调线搜索
            min_i = min(i,Ipt.M);
            for j = 1:Ipt.max_iter1
                judgement = 0;
                for ind = 1:min_i
                    judgement = judgement...
                        |(func(Retraction(x,-alpha*gf,Ipt.method))...
                        <=func(Opt.x{i-ind+1})...
                        -Ipt.c*alpha*norm(gf,"fro")^2);
                end
                if judgement
                    break;
                end
                alpha = Ipt.rho*alpha;
            end

            % 更新迭代点
            x1 = Retraction(x,-alpha*gf,Ipt.method);
            gf1 = Func.gf(x1);

            % 计算BB步长
            s = x1-x;
            y = gf1-gf;
            alphaSBB = trace(s'*y)/trace(y'*y);
            alphaLBB = trace(s'*s)/trace(s'*y);
            % 使用交替步长
            if Ipt.slot == 0 
                if mod(i,2) == 0 
                    alphaABB = alphaLBB;
                else
                    alphaABB = alphaSBB;
                end
            % 使用长步长
            elseif Ipt.slot == 1 
                alphaABB = alphaLBB; 
            % 使用短步长
            elseif Ipt.slot == -1 
                alphaABB = alphaSBB; 
            end

            % 更新各参数
            Opt.alpha{i+1} = min(Ipt.alphamax,...
                max(Ipt.alphamin,alphaABB));
            Opt.x{i+1} = x1;
            Opt.f{i+1} = Func.f(x1);
            Opt.gf{i+1} = Func.gf(x1);
        end
        Opt.time = toc(time_start);
        Opt.iter = i;
    end

    % 生成二次函数系数
    function [A,B] = Random_Coefficient(n,p,seed,density)
        rng(seed);
        A = sprand(n,n,density);
        A = A'+A;
        B = randn(n,p);
    end

    % 获取初始点
    function x = Random_Initial(n,p,seed)
        rng(seed);
        x = rand(n,p);
        x = Retraction(x,zeros(n,p),"qr");
    end

    % 计算最小值点和最小值的精确值
    function Exa = Exact_Value(Func)
        Ipt.alpha0 = 0.02;                       % BB算法初始步长
        Ipt.rho = 0.5;                           % 回退幅度
        Ipt.c = 0.001;                           % Armijo条件的判据
        Ipt.max_iter1 = 50;                      % 回退法的最大迭代次数
        Ipt.method = "qr";                       % 收缩映射的方式
        Ipt.max_iter2 = 500;                     % 梯度下降法的最大迭代次数
        Ipt.epsilon = 10e-10;                    % 梯度阈值
        Ipt.alphamin = 0.001;                    % BB步长下界
        Ipt.alphamax = 1000;                     % BB步长上界
        Ipt.M = 5;                               % 非单调线搜索列表长度
        Ipt.slot = 0;                            % 决定是否使用交替步长
        Opt = BB_Method(Func,Ipt);
        Exa.x = Opt.x{Opt.iter};
        Exa.f = Opt.f{Opt.iter};
    end

    % 待优化函数
    function f = func(x)
        f = trace(x'*A*x-2*x'*B);
    end

    % 投影映射
    function p = projection(x,v)
        p = v-x*(x'*v+v'*x)/2;
    end

    % 欧氏梯度
    function nf = nablafunc(x)
        nf = 2*A*x-2*B;
    end

    % 黎曼梯度
    function gf = gradfunc(x)
        gf = projection(x,nablafunc(x));
    end

    % 收缩映射
    function A = Retraction(X,V,method) 
        if method == "qr" % 基于QR分解
            BL = X+V;
            [Q,R] = qr(BL,"econ");
            r = sign(diag(R));
            A = Q*diag(r);
        elseif method == "svd" % 基于SVD分解
            BL = X+V;
            [U,~,V] = svd(BL,"econ");
            A = U*V';
        elseif method == "polar" % 基于极分解
            BL = X+V;
            C = V'*V;
            C = sqrtm(eye(p)+C);
            A = BL/C;
        elseif method == "cayley" % 基于Cayley变换
            W = eye(n)-0.5*(X*X');
            U = [W*V,X];
            Z = [X,-W*V];
            A = X+U/(eye(2*p)-0.5*(Z'*U))*Z'*X;
        end
    end

end