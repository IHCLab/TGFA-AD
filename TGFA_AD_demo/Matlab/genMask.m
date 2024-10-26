function [Mask ]= genMask(a)
gTerm = gamma(1-a);
sqrt2a = sqrt(2^a);
g12 = 1*a*sqrt2a/(2*gTerm);
g11 = 1*a/(1*gTerm);

Mask=  [g12, 0, -g12;
       g11, 0, -g11;
       g12, 0, -g12;];

end