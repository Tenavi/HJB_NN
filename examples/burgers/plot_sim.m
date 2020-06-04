% Dimension
D = 10;

W = 0;
load(['D', int2str(D), '/tspan/results/sim_data.mat'])

xi = [1,xi,-1];
X_NN = [zeros(1,size(X_NN,2)); X_NN; zeros(1,size(X_NN,2))];

ZLIM = [-3,2];

close all

fig = figure;
fig.Position(3:4) = [350, 250];

view(30,30)
hold on
box on

ax = gca;
ax.FontSize = 12;
ax.XLim = [-1, 1];
ax.YLim = [min(t), max(t)];
ax.ZLim = ZLIM;

[T,XI] = meshgrid(t,xi);

surf(XI',T',X_NN','edgecolor','none','facecolor','interp','facealpha',0.75)

for d=[1,length(xi)]
    plot3(xi(d)*ones(size(t)),t,X_NN(d,:),'k')
end
plot3(xi,t(1)*ones(size(xi)),X_NN(:,1),'k')
plot3(xi,t(end)*ones(size(xi)),X_NN(:,end),'k')

%colormap('gray')
caxis(ZLIM)

xlabel('$\xi$', 'FontSize',16,'interpreter','latex')
ylabel('$t$', 'FontSize',16,'interpreter','latex')
zlabel('$X(t,\xi)$', 'FontSize',16,'interpreter','latex')

fig = figure;
fig.Position(3:4) = [350, 150];

ax = gca;
ax.FontSize = 12;
ax.YLim = [-1,1];

hold on
box on

plot(t,U_BVP,'DisplayName', 'optimal open-loop')
if any(any(W))
    stairs(t,U_NN,':','DisplayName', 'NN feedback','linewidth', 2)
else
    plot(t,U_NN,':','DisplayName', 'NN feedback','linewidth', 2)
end
%plot(t,U_LQR,'--','DisplayName', 'LQR','linewidth', 1.5)

xlabel('$t$','FontSize',16,'interpreter','latex')
ylabel('$u$','FontSize',16,'interpreter','latex')

lgd = legend;
lgd.FontSize = 14;
lgd.Interpreter = 'latex';
lgd.Location = 'northeast';