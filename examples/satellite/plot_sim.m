linewidth = [1.25,2];

W = 0;
load('t0/results/sim_data.mat')

plot_BVP = 0;
plot_LQR = 0;

U_max = 4;
XLim = [0,t(end)];

fig = figure;
%fig.Position(3:4) = [350, 350];
fig.Position(3:4) = [700, 450];

for i=1:3
    subplot(3,1,i);
    hold on
    
    axis tight
    box on

    ax = gca;
    ax.FontSize = 12;
    ax.XLim = XLim;
    ax.ColorOrder = ax.ColorOrder([1,2,4],:);
    
    if i==1
        ax.YLim = [-pi/2,pi/2];
        ax.YTick = [-pi/3,0,pi/3];
        ax.YTickLabel = {'-\pi/3','0','\pi/3'};

        if plot_BVP
            plot(t, X_BVP(1,:),'k-','linewidth', linewidth(1))
            plot(t, X_BVP(2,:),'k--','linewidth', linewidth(1))
            plot(t, X_BVP(3,:),'k:','linewidth', linewidth(2))
        elseif plot_LQR
            plot(t, X_LQR(1,:),'k-','linewidth', linewidth(1))
            plot(t, X_LQR(2,:),'k--','linewidth', linewidth(1))
            plot(t, X_LQR(3,:),'k:','linewidth', linewidth(2))
        end
            
        plot(t, X_NN(1,:),'-','linewidth', linewidth(1))
        plot(t, X_NN(2,:),'--','linewidth', linewidth(1))
        plot(t, X_NN(3,:),':','linewidth', linewidth(2))

        ylabel('$\mathbf v$','FontSize',16,'interpreter','latex')
    elseif i==2
        ax.YLim = [-pi/4,pi/4];
        ax.YTick = [-pi/4,0,pi/4];
        ax.YTickLabel = {'-\pi/4','0','\pi/4'};

        if plot_BVP
            plot(t, X_BVP(4,:),'k-','linewidth', linewidth(1))
            plot(t, X_BVP(5,:),'k--','linewidth', linewidth(1))
            plot(t, X_BVP(6,:),'k:','linewidth', linewidth(2))
        elseif plot_LQR
            plot(t, X_LQR(4,:),'k-','linewidth', linewidth(1))
            plot(t, X_LQR(5,:),'k--','linewidth', linewidth(1))
            plot(t, X_LQR(6,:),'k:','linewidth', linewidth(2))
        end
            
        plot(t, X_NN(4,:),'-','linewidth', linewidth(1))
        plot(t, X_NN(5,:),'--','linewidth', linewidth(1))
        plot(t, X_NN(6,:),':','linewidth', linewidth(2))
        
        ylabel('\boldmath $\omega$','FontSize',16,'interpreter','latex')
    elseif i==3
        ax.YLim = [-U_max,U_max];
        ax.YTick = [-U_max,0,U_max];
        
        if plot_BVP
            plot(t, U_BVP(1,:),'k-','linewidth', linewidth(1))
            plot(t, U_BVP(2,:),'k--','linewidth', linewidth(1))
            plot(t, U_BVP(3,:),'k:','linewidth', linewidth(2))
        elseif plot_LQR
            if any(any(W))
                stairs(t, U_LQR(1,:),'k-','linewidth', linewidth(1))
                stairs(t, U_LQR(2,:),'k--','linewidth', linewidth(1))
                stairs(t, U_LQR(3,:),'k:','linewidth', linewidth(2))
            else
                plot(t, U_LQR(1,:),'k-','linewidth', linewidth(1))
                plot(t, U_LQR(2,:),'k--','linewidth', linewidth(1))
                plot(t, U_LQR(3,:),'k:','linewidth', linewidth(2))
            end
        end
           
        if any(any(W))
            stairs(t, U_NN(1,:),'-','linewidth', linewidth(1))
            stairs(t, U_NN(2,:),'--','linewidth', linewidth(1))
            stairs(t, U_NN(3,:),':','linewidth', linewidth(2))
        else
            plot(t, U_NN(1,:),'-','linewidth', linewidth(1))
            plot(t, U_NN(2,:),'--','linewidth', linewidth(1))
            plot(t, U_NN(3,:),':','linewidth', linewidth(2))
        end
        
        xlabel('$t$','FontSize',16,'interpreter','latex')
        ylabel('$\mathbf u$','FontSize',16,'interpreter','latex')
    end
end