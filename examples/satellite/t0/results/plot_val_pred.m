load('val_pred.mat')

fig = figure;
fig.Position(3:4) = [350, 300];
hold on
box on
axis tight

ax = gca;
ax.FontSize = 12;

plotarg = 'V';
plotarg = 'u';

if plotarg == 'V'
    surf(squeeze(X(1,:,:)), squeeze(X(2,:,:)), V,...
        'edgecolor','none', 'facecolor','interp','facealpha',0.75)
else
    surf(squeeze(X(1,:,:)), squeeze(X(2,:,:)), U,...
        'edgecolor','none', 'facecolor','interp','facealpha',0.75)
end

view(30,30)

labels = ["",""];
for i=1:2
    if plotdims(i) == 1
        labels(i) = "$\phi$";
    elseif plotdims(i) == 2
        labels(i) = "$\theta$";
    elseif plotdims(i) == 3
        labels(i) = "$\psi$";
    else
        labels(i) = "$\omega_{" + (plotdims(i)-3) + "}$";
    end
end

xlabel(labels(1), 'Interpreter', 'Latex', 'Fontsize', 16)
ylabel(labels(2), 'Interpreter', 'Latex', 'Fontsize', 16)
if plotarg == 'V'
    zlabel(['$', plotarg, '(\mathbf x)$'], 'Interpreter', 'latex', 'Fontsize', 16)
else
    zlabel('$u^{NN}$',...
        'Interpreter', 'latex', 'Fontsize', 16)
end