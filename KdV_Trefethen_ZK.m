% Matt Tranter

% Solving KdV equation in Matlab, pseudospectral, Trefethen modification
% of Runge-Kutta method for time-stepping

% Clear variable list
clear;

% Parameters in the equation
alpha = 1.0;
beta = 0.022^2;

% Create directory with relevant name and navigate to created directory
mkdir('KdV_ZK');
cd('KdV_ZK');

% Set number of points and domain size
xa = 0;
xb = 2;
N = 400;
h = (xb - xa)/N;

% Pre-allocated arrays
u1 = zeros(1, N);

% Calculate s
s = 2*pi/(xb - xa);

% Time step
dt = 0.0001;

% Set tmax
tmax = 10;
nt = tmax/dt;

% Set number of outputs
fout = 0.5;
nout = tmax/fout;
fo = fout/dt;

% Spatial discretisations
x = linspace(xa, xb-h, N);

% Anti-aliasing array coefficients (Orszag 2/3 rule)
AA = ones(1,N);
AA(round(N/3):round(2*N/3)) = 0;

% Calculate initial condition
U = cos(pi*x);

% Set k array
k = [0:N/2,-(N/2-1):-1];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve KdV equation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Output initial condition
h = figure('Units','Normalized','Position',[.1 .1 .55 .55]);
plot(x, U, '-b','LineWidth',1.5);
xlim([0 2]); % Manually adjust to appropriate value
title(['Solution of the KdV equation at t = 0']);
set(gca, 'FontSize', 32);
xlabel('x');
ylabel('u','Rotation',0);
ax = gca;
ax.LineWidth = 2.5;
savefig('KdVn0.fig');
close(h);

% Convert initial condition to Fourier space
U = fft(U);

% Anti-aliasing on initial condition
U = U.*AA;

% Progress bar
h1 = waitbar(0,sprintf('KdV equation'));

% Define E and RKCoef (for Trefethen method)
E = exp(-(1.0/2.0)*(1i)*dt*(-beta*s^3*(k.^3)));
RKCoef = -(1i)*s*alpha*dt*k;

% Solve using Runge-Kutta 4th order method
% Two loops - one for inner calculation, one for number of outputs
for i = 1:nout
    for j = 1:fo

        % Calculate RK4 coefficients for both leading order and higher order terms
        for z = 1:4
            switch z
                case 1
                    utemp = U;
                case 2
                    utemp = E.*(U + u1(1,:)/2);
                case 3
                    utemp = E.*U + u1(2,:)/2;
                case 4
                    utemp = (E.^2).*U + E.*u1(3,:);
            end

            % Compute u1
            u1(z,:) = (RKCoef/2).*fft(ifft(U.*AA).^2);

            % Anti-aliasing
            u1(z,:) = u1(z,:).*AA;
        end

        % Find next solution
        U = (E.^2).*U + (1/6)*((E.^2).*u1(1,:) + 2*E.*(u1(2,:) + u1(3,:)) + u1(4,:));

        % Update progress bar
        waitbar(((i-1)*fo + j)/(nout*fo));
    end

    % Comparison plot - closed immediately as now saved in directory
    h = figure('Units','Normalized','Position',[.1 .1 .55 .55]);
    plot(x, real(ifft(U.*AA)), '-b','LineWidth',1.5);
    xlim([0 2]); % Manually adjust to appropriate value
    title(['Solution of the KdV equation at t = ' num2str(i*fo*dt)]);
    set(gca, 'FontSize', 32);
    xlabel('x');
    ylabel('u','Rotation',0);
    ax = gca;
    ax.LineWidth = 2.5;

    % Save figure with relevant name
    savefig(['KdVn' num2str(i*fo*dt) '.fig']);
    close(h);
end
close(h1);