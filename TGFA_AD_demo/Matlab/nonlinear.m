function residual_pp=nonlinear(anundance3D,Cdl)
R=abs(anundance3D-Cdl);
nonlinear_residual=(1-exp(-1*R));
dcetection_result=sum(nonlinear_residual,3).^2;
residual_pp=nor(dcetection_result);% normalize to 0-1
