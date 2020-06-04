load('t0/results/train_results.mat')

figure

box on
hold on

iters = 1:round_iters(end);

ax = gca;
ax.FontSize = 12;
ax.YScale = 'log';
ax.ColorOrder = ax.ColorOrder([1,2,3],:);
%ax.YLim = [10^-4, 10];
ax.XLim = [0, iters(end)];

title('\textbf{Training progress}','FontSize',16,'interpreter','latex')

train_err = smooth(train_err, length(train_err)/1000);
train_grad_err = smooth(train_grad_err, length(train_grad_err)/1000);
train_ctrl_err = smooth(train_ctrl_err, length(train_ctrl_err)/1000);

% Training error over time
plot(iters, train_err)
plot(iters, train_grad_err)
plot(iters, train_ctrl_err)
% (Final) validation error
stairs([0,round_iters], [val_err,val_err(end)], ':', 'linewidth', 2)
stairs([0,round_iters], [val_grad_err,val_grad_err(end)], ':', 'linewidth', 2)
stairs([0,round_iters], [val_ctrl_err,val_ctrl_err(end)], ':', 'linewidth', 2)

lgd = legend('value (training)', 'costate (training)',...
    'control (training)','value (validation)', 'costate (validation)',...
    'control (validation)');
lgd.FontSize = 16;
lgd.Interpreter = 'latex';
lgd.Location = 'Northeast';

xlabel('iteration','FontSize',16,'interpreter','latex')
ylabel('error','FontSize',16,'interpreter','latex')