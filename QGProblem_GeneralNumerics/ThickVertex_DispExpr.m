function [val] = ThickVertex_DispExpr(w, kappa, alpha)
%this is the expression which, when it's between -1 and 1, gives
%eigenvalues w.

eta = sqrt(w.*w - kappa.*kappa);

val = cos(eta) - (alpha/4).*(w.*w).*sin(eta)./eta;

end% function