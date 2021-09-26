% IML - Project 2 - Q1.1.b
% (Antoine Debor & Jan Held, November 2020)

% Analytical computation of the generalization error of the Bayes
% model

p = 0.75;

fun1 = @(x1,x0) (1/(4*pi*sqrt(1-p^2)))* exp((-1/(2-2*p^2))*(x0.^2-2*p*x0.*x1+x1.^2));
fun2 = @(x0,x1) (1/(4*pi*sqrt(1-p^2)))* exp((-1/(2-2*p^2))*(x0.^2+2*p*x0.*x1+x1.^2));

result_1 = integral2(fun1, 0, +Inf, -Inf, 0);
result_3 = integral2(fun2, 0, +Inf, 0, +Inf);
result = result_1*2 + result_3*2

