%Creates a bandgap plot for the TFR-setup; but using the curl equations and
%non-classical Kirchoff condition at central vertex

clear;
close all;

%NB: This computation works by assuming the opposite derivative convention
%to EKK paper, so alpha corresponds to -alpha in the TFR computations.
alpha = -1;
titStr = strcat('Bandgap Plot, $\alpha=', num2str(alpha, '%.2f'), '$');

%setup the range of samples for kappa
kPts = 1000;
kappaRange = linspace(0,3*pi,kPts);

%setup the range of samples for w
wPts = 1000;
wRange = linspace(0,6*pi,wPts);

%setup matrix that will display bandgap plot
drValsMatrix = zeros(kPts, wPts); %remember, surface plots in MATLAB are rows: y and cols: x.
for k=1:kPts
    %first find the index of the first admissible value for w
    kappa = kappaRange(k);
    firstInd = find(wRange>=kappa, 1);
        
    %compute spectrum for all allowable values of w
    drVals = ThickVertex_DispExpr(wRange(firstInd:end), kappaRange(k), alpha);
    %insert into matrix of dispersion relation values
    drValsMatrix(k, firstInd:end) = drVals;
    %set non-admissible w values to be NaNs in the drValsMatrix
    drValsMatrix(k, 1:firstInd-1) = NaN;
end %for, k

%drValsMatrix is now the matrix of all the (valid) disperion relation
%values. From this we can create the bandgap plot.
%Eigenvalues occur when the DRel has absolute value <=1, cast through
%sign() so as to covert to float data type for plotting
bandgapPlot = sign(abs(drValsMatrix)<=1);

figure;
%contour(wRange./pi, kappaRange./pi, bandgapPlot);
surf(wRange./pi, kappaRange./pi, bandgapPlot, 'Edgecolor','none'); view(0,90); %colorbar;
xlabel('Frequency, $\frac{\omega}{\pi}$','interpreter','latex');
ylabel('Wavenumber, $\frac{\kappa}{\pi}$','interpreter','latex');
xlim([wRange(1)/pi wRange(end)/pi])
ylim([kappaRange(1)/pi, kappaRange(end)/pi])
title(titStr, 'interpreter','latex')

%plot with minimal whitespace, and save to PDF
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
%autosave... but saving the figure in the interactive window is probably
%safer for now!
% print(fig,'MySavedFile','-dpdf')
% fprintf('Saved figure to MySavedFile.pdf - previous has been overwritten!');