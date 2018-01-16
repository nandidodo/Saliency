function exercise4

mean_phi = 5;
sigma_phi == 2;
phi_above = 5;
DT = 0.01;
MAXT = 20;
TRIALS = 1000;
LRATE = 001;
Sigma(1) = 1;
for trial = 2:TRIALS
	error(i) = 0;
	e(i) = 0;
	phi = mean_phi + sqrt (sigma_phi) * randn;
	for i = 2:MAXT/DT
		error(i) = error(i-1) + DT * (phi - phi_above - e(i-1));
		e(i) = e(i-1) + DT * (Sigma(trial-1) * error(i-1) - e(i-1));
	end
	Sigma(trial) = Sigma(trial-1) + LRATE * (error(end) * e(end) -1);
end

plot(Sigma, 'k');
xlabel('Trial');
ylabel('\Sigma');