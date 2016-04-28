clear all ; clf

tdfread('train_x.dat')
tdfread('train_y.dat')
tdfread('test_x.dat')
tdfread('pred20_mean.dat')
tdfread('pred20_var.dat')

ubound = mean+sqrt(var)*2;
lbound = mean-sqrt(var)*2;

% function for filling between two lines Y1 and Y2
fill_between_lines = @(X,Y1,Y2,C) fill( [X fliplr(X)],  [Y1 fliplr(Y2)], C );

plot(xtest, mean, 'b', 'LineWidth', 2)
hold on
plot(xtest, lbound, 'r', 'LineWidth', 0.5)
plot(xtest, ubound, 'r', 'LineWidth', 0.5)

xx = [xtest;flip(xtest)];
yy = [lbound;flip(ubound)];
h=fill(xx,yy,'r');
set(h,'facealpha',.2);
set(h,'EdgeColor','None');

set(gca,'fontsize',18)

plot(x,y,'xk', 'MarkerSize', 6)

grid on

ylim([-3 3])
xlabel('inputs,x')
ylabel('outputs,y')