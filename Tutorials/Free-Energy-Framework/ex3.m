function exercise3

v_p = 3;
sigma_p = 1;
sigma_u = 1;
u =2;
DT = 0.01;
MAXT =5;

phi(1) = v_p;
error_p(1) = 0;
error_u(1) = 0;

for i = 2:MAXT/DT
	phi(i) = phi(i-1) + DT * (- error_p(i-1) * (2*phi(i-1)));
	error_p(i) = error_p(i-1) + DT * (phi(i-1) -v_p - sigma_p * error_p(i-1));
	error_u(i) = error_u(i-1) + DT * (u - phi(i-1)^2 0 sigma_u * error_u(i-1));
end

plot([DT:DT:MAXT], phi,'k');
hold on
plot([DT:DT:MaxT], error_p, 'k--');
plot([DT:DT:MAXT],error_u, 'k:');
xlabel ('Time');
ylabel('Activity');
legend ('\phi','\epsilon_p','\epsilon_u');
axis([0 MAXT -2 3.5]);