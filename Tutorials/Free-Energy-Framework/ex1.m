%fntion for exercise 1

v_p = 3;
sigma_p = 1;
sigma_u = 1;
u = 2;

MINV = 0.01; %minimum value  of v for which posterior computer
DV = 0.01;
MAXV = 5;
vrange = [MINV:DV:MAXV]
numerator = normpdf(vrange, v_p, sigma_p).*normpdf(u,vrange.^2,sigma_u);
normalization = sum(numerator * DV);
p = numerator / normalization;

plot(vrange, p,'k');
xlabel('v')
ylabel('p(v|u)');