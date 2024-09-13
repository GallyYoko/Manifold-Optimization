n = 10;
p = 3;
seed = 0;

[A,B] = Random_Coefficient(n,p,seed);
func = @(x) trace(x'*A*x-2*x'*B);
gradfunc = @(x) 2*A*x-2*B;

x0 = Random_Initial(n,p,seed);
t0 = 1;
rho = 0.5;
c = 0.001;
iteration1 = 100;
method = "qr";
iteration2 = 100;
epsilon = 100*eps;

Gradient_Descent(func,gradfunc,x0,t0,rho,c,iteration1,method,iteration2,epsilon);