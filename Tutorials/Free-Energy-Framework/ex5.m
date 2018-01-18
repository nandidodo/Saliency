%this is meant to simulate the learning sigma for a single variable and node, but I don't understand how
it is actually meant to work'

mean_phi = 5;
sigma_phi = 2;
phi_above 5 %input from the layer above
DT = 0.01;
MAXT = 20;
TRIALS = 1000;
LRATE = 0.01;

Sigma(1) = 1; %initialise weight value!
for trial = 2:TRIALS
	error(1) = 0; %initialise the perdiction error
	e(1) = 0; % initialize the inhibitory interneuron
	phi = mean_phi + sqrt(sigma_phi) * randn; %draw our pretend phi frmo normal 
	%distribution with mean and variance
	for i = 2: MAXT/DT
		error(i) = error(i-1) + DT * (phi - phi_above) - e(i-1);
		e(i) = e(i-1) + DT * (sigma(trial-1) * error(i-1)  e(i-1));
	end
	sigma(trial) = sigma(trial-1) + LRATE * (error(end)*e(end) -1);
end

plot(sigma, 'k');
xlabel('Trial');
ylabel('\sigma');