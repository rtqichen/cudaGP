% Trains a Gaussian Process on the grayroos dataset
% then predicts the mean and covariance matrix for some test points.
% This file is mainly used to test cudaGP is performing correctly.

clear all, close all

tdfread('grayroos-header.dat');

xtest = linspace(500, 850, 201)';

l = 8;
kern = @(r) exp(-r^2/(2*l^2));

% construct Kyy
A = [];
for i=1:length(x)
    A = [A x(i)-x];
end

K = arrayfun(kern, A);

% construct Kfy and Kff

B = [];
for i=1:length(x)
    B = [B x(i)-xtest];
end

C = [];
for i=1:length(xtest)
    C = [C xtest(i)-xtest];
end

Kfy = arrayfun(kern, B);
Kff = arrayfun(kern, C);

% calculate mean and covariance
pred.mean = Kfy * (K\y);
pred.cov = Kff - Kfy*(K\Kfy');

plot(xtest, pred.mean)
hold on
plot(x,y,'.')