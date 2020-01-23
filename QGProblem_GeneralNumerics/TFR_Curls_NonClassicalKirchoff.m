%Visualise spectrum for TFR-setup; but using the curl equations and
%non-classical Kirchoff condition at central vertex

clear;
close all;

kappa = pi;
%NB: This computation works by assuming the opposite derivative convention
%to EKK paper, so alpha corresponds to -alpha in the TFR computations.
alpha = 8;
titStr = strcat('Spectral Plot, $\kappa=', num2str(kappa, '%.2f'), ', \ \alpha=', num2str(alpha, '%.2f'), '$');

%w<kappa not allowed!
wPts = 1000;
wRange = linspace(0,6*pi,wPts) + kappa;
drVals = DispExpr(wRange, kappa, alpha);

%highlight spectrum in red
specPlot = zeros(size(drVals));
specPlot(abs(drVals)>1) = 1;
specWVals = wRange(specPlot==0);
specToDraw = specPlot(specPlot==0);

figure;
hold on;
plot(wRange./pi, drVals,'-b');
plot(wRange./pi, ones(size(wRange)), '-k');
plot(wRange./pi, -1.*ones(size(wRange)), '-k');
plot(specWVals./pi, specToDraw, '.r');
xlabel('$\frac{\omega}{\pi}$','interpreter','latex');
ylabel('Dispersion Expression Value','interpreter','latex');
xlim([wRange(1)/pi wRange(end)/pi])
yDist = 5;
ylim([-1-yDist, 1+yDist])
title(titStr, 'interpreter','latex')

function [val] = DispExpr(w, kappa, alpha)
%this is the expression which, when it's between -1 and 1, gives
%eigenvalues w.

eta = sqrt(w.*w - kappa.*kappa);

val = cos(eta) - (alpha/4).*(w.*w).*sin(eta)./eta;

end% function