%Visualise the range of valid w when
%(w^2-k^2)^1/2 = a+2n*pi where -pi<=a<pi

clear;

kappa = 1;
nMax = 3;

nIntervals = -nMax:nMax;
piIntervals = [ 2*nIntervals-1 ; 2*nIntervals+1 ].*pi;
differenceIntervals = piIntervals.*abs(piIntervals);
wSquareIntervals = differenceIntervals + kappa^2;
wIntervals = sign(wSquareIntervals).*sqrt(abs(wSquareIntervals));

figure;
hold on;
for i=1:(2*nMax+1)
    plot(wIntervals(:,i), [0,0], '.-b');
end %for
ylim([-0.1,0.1])
hold off;