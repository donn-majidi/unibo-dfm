clear; clear all; clc;
addpath('src/matlab');
%Initialize the data
X_data = readtable('./data/original_stationary_data.csv', 'VariableNamingRule','preserve');
sasdate = X_data.sasdate;
X_raw = table2array(X_data(:, 3:end));
%X_raw = X_raw(2:end, 3:end);
%dates = X_raw(1:end, 1);
col_means = mean(X_raw, 'omitnan');
col_std = std(X_raw, 'omitnan');

function [y, M, s] = ML_Standardize(x)

T=size(x,1);
s = nanstd(x);
M = nanmean(x);
ss = ones(T,1)*s;
MM = ones(T,1)*M;
y = (x-MM)./ss;

end

[X, M, std] = ML_Standardize(X_raw);

X(isnan(X)) = 0;

%%
%The ABC criterion for finding the number of common factors
rng(1776);
N = size(X,2);
T = size(X,1);
kmax = 8;
nbck = floor(N/10);
cmax = 3;
graph = 1;
[rhat1, rhat2] = ABC_crit(X, kmax, nbck, cmax, graph);

%disp(sprintf(' True number of factors %d', r));
%disp(sprintf(' Estimated number of factors with large window %d', rhat1));
%disp(sprintf(' Estimated number of factors with small window %d', rhat2));
%disp(sprintf(' Estimated number of factors average %f', (rhat1+rhat2)/2));

%%
%We first use the PCA estimates of the factors to determine the lag order
%of the state equation

r = 6;
q = 6;
p = 1;
iter = 50;
thresh = 10^(-4);
s0 = 0;

[~, PCA] = BL_Estimate(X, r, q, p, iter, thresh, s0);
%%
for i = 1:size(PCA.F, 2)
    figure(i);
    parcorr(PCA.F(:,i));
end

%%
max_lag = 4;
criterion = 'BIC';
[optimal_lag, criteria_values] = select_var_lag_order(PCA.F, max_lag, criterion);
disp(['Optimal lag order (', criterion, '): ', num2str(optimal_lag)]);
disp(['Criterion values for each lag:']);
disp(criteria_values);

figure;
plot(1:max_lag, criteria_values, '-o');
xlabel('Lag Order');
ylabel(criterion);
title(['Lag Selection using ', criterion]);
grid on;
%%
r = 6;
q = 6;
p = 2;
iter = 50;
thresh = 10^(-4);
s0 = 0;

[EM, ~] = BL_Estimate(X, r, q, p, iter, thresh, s0);

%%
%Update the original data by replacing missing values with the estimated
%common components
X(X == 0) = EM.chi(X == 0);
%%
%Now re-run everything using updated database
N = size(X,2);
T = size(X,1);
kmax = 8;
nbck = floor(N/10);
cmax = 3;
graph = 1;
[rhat1, rhat2] = ABC_crit(X, kmax, nbck, cmax, graph);
%%
r = 6;
q = 6;
p = 2;
iter = 50;
thresh = 10^(-4);
s0 = 0;

[EM, PCA] = BL_Estimate(X, r, q, p, iter, thresh, s0);
%%
%We now save the estimates of the idiosyncratic components to import them
%in R
writematrix(EM.xi, './estimated_idiosyncratic.csv');
%%
max_lag = 4;
criterion = 'BIC';
[optimal_lag, criteria_values] = select_var_lag_order(PCA.F, max_lag, criterion);
disp(['Optimal lag order (', criterion, '): ', num2str(optimal_lag)]);
disp(['Criterion values for each lag:']);
disp(criteria_values);

figure;
plot(1:max_lag, criteria_values, '-o');
xlabel('Lag Order');
ylabel(criterion);
title(['Lag Selection using ', criterion]);
grid on;
%%
%Plots of factors 
for i = 1:size(EM.F, 2)
    figure(i);
    plot(EM.F(:,i));
    title(sprintf('Factor %d Estimate', i), 'FontSize', 12, 'FontWeight', 'bold');
    tick_indices = 2:20:length(sasdate);
    xticks(tick_indices);
    xticklabels(string(sasdate(tick_indices)));
    grid on;
end
%%
%Forecasting
%Since the state equation is a VAR(2) process, we define the selection
%matrix as follows to be able to extract the desired component from the
%companion form
R1 = [eye(6); zeros(6)];
disp(R1');

forecast_series = [1,2,57,18,22,78,142];

prediction_errors = zeros(length(forecast_series),42);
forecasted = zeros(length(forecast_series),42);

for i = 1:162
    % For rolling window:
    subX = X(i:i+79,:);
    % For expanding window:
    %subX = X(1:i+79,:);
    [EM, ~] = BL_Estimate(subX, r, q, p, iter, thresh, s0);
    ftt = EM.F(end, :);
    ftt1 = EM.F(end-1, :);
    stacked = [ftt, ftt1];
    for j = forecast_series
        forecasted_common = EM.Lambda(j, 1:6) * R1' * EM.A * stacked';
        forecasted(j,i) = col_means(1,j) + col_std(1,j) * forecasted_common;
        prediction_errors(j,i) = X_raw(i+80,1) - forecasted(j,i);
    end
end
%%
x1 = 1:242;
x2 = 81:242;
mse = zeros(1,length(forecast_series));
for j = forecast_series
    mse(1,j) = sum(prediction_errors(j,:).^2) / size(prediction_errors,2);
    figure(j);
    plot(x1, X_raw(1:end,j), '-', 'DisplayName', 'Observed Series', 'Color', 'k');
    hold on;
    plot(x2, forecasted(j,:), '-x', 'DisplayName', 'Forecasted Series', 'Color', 'r');
    xlabel('Time');
    ylabel('Values');
    %title('1 Step Ahead Forecasts Using The Common Component');
    tick_indices = 2:20:length(sasdate);
    xticks(tick_indices);
    xticklabels(string(sasdate(tick_indices)));
    xlim([1 242])
    legend;
    grid on;
end

