clear all
clf

tdfread('train_x.dat')
tdfread('train_y.dat')
tdfread('test_x.dat')
tdfread('pred8_mean.dat')
tdfread('pred8_var.dat')

plot(x,y,'.k')
hold on
plot(xtest, mean, 'b')
plot(xtest, mean+sqrt(var)*2, 'r')
plot(xtest, mean-sqrt(var)*2, 'r')

title('Distributed GP Prediction with 8 Clusters')
ylim([-3 3])
xlabel('x')
ylabel('y')