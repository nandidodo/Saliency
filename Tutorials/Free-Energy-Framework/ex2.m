function exercise2

v_p=3;
sigma_p = 1;
sigma_u =1
u=2;
DT = 0.01;
MAXT = 5;
phi(1) = v_p
for i = 2:MAXT/DT
	phi(i)=phi(i-1 + DT *((v_p-phi(i-1)/sigma_p + ...
		(u-phi(i-1)/sigma_u * (2*phi(i-1)))));
end
plot([DT:DT:MAXT],phi,'k');
xlabel('Time');
ylabel('\phi');
axis([0 MAXT -2 3.5]);