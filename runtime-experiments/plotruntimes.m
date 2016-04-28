clear all; clf;

tdfread('runtimes');

R = [N Time];

% RGB values
C = [[213,62,79],
[244,109,67],
[253,174,97],
[254,224,139],
[230,245,152],
[171,221,164],
[102,194,165],
[50,136,189],
[0,0,0]];

C = C/255;

kreggie = R(K==-1,:); % runtimes for reggie CPU implementation
k1 = R(K==1,:);
k10 = R(K==10,:);
k20 = R(K==20,:);
k50 = R(K==50,:);
k100 = R(K==100,:);
k200 = R(K==200,:);
k500 = R(K==500,:);
k1000 = R(K==1000,:);

lt = '-o';
lw = 2;
ms = 6;
mfc = 'none';

loglog(k1(:,1), k1(:,2), lt, 'LineWidth', lw, 'Color', C(1,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
hold on; grid on
loglog(k10(:,1), k10(:,2), lt, 'LineWidth', lw, 'Color', C(2,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(k20(:,1), k20(:,2), lt, 'LineWidth', lw, 'Color', C(3,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(k50(:,1), k50(:,2), lt, 'LineWidth', lw, 'Color', C(4,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(k100(:,1), k100(:,2), lt, 'LineWidth', lw, 'Color', C(5,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(k200(:,1), k200(:,2), lt, 'LineWidth', lw, 'Color', C(6,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(k500(:,1), k500(:,2), lt, 'LineWidth', lw, 'Color', C(7,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(k1000(:,1), k1000(:,2), lt, 'LineWidth', lw, 'Color', C(8,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)
loglog(kreggie(:,1), kreggie(:,2), lt, 'LineWidth', lw, 'Color', C(9,:), 'MarkerSize',ms, 'MarkerFaceColor', mfc)

set(gca,'fontsize',18)

title('Runtime of Sparse GP Prediction on CUDA with Varying N and K')
xlabel('Dataset size,N')
ylabel('Runtime (seconds)')
legend('Full GP K=1', 'Sparse K=10', 'Sparse K=20', 'Sparse K=50', 'Sparse K=100', 'Sparse K=200', 'Sparse K=500', 'Sparse K=1000', 'reggie CPU', 'location', 'southeast')
