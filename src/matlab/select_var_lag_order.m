function [optimal_lag, criteria_values] = select_var_lag_order(factors, max_lag, criterion)
% SELECT_VAR_LAG_ORDER Determines the optimal lag order for a VAR model.
%
% INPUTS:
%   factors    - Matrix of factors (T x n), where T is the number of time steps and n is the number of factors.
%   max_lag    - Maximum lag order to consider.
%   criterion  - Criterion for lag selection ('AIC', 'BIC', 'HQIC').
%
% OUTPUTS:
%   optimal_lag     - Optimal lag order based on the specified criterion.
%   criteria_values - Vector of criterion values for lags 1 to max_lag.

    [T, n] = size(factors);
    criteria_values = zeros(max_lag, 1);

    % Loop over possible lag orders
    for lag = 1:max_lag
        % Create lagged predictors and responses
        [X_lagged, Y] = create_lagged_data(factors, lag);
        T_lagged = size(Y, 1); % Effective number of observations
        
        % Fit VAR model
        B = (X_lagged' * X_lagged) \ (X_lagged' * Y); % OLS estimator
        residuals = Y - X_lagged * B;                % Residuals
        sigma = cov(residuals);                      % Residual covariance matrix
        
        % Compute log-likelihood
        logL = -0.5 * T_lagged * (n * log(2 * pi) + log(det(sigma)) + 1);

        % Compute information criteria
        switch upper(criterion)
            case 'AIC'
                criteria_values(lag) = -2 * logL + 2 * (n^2 * lag);
            case 'BIC'
                criteria_values(lag) = -2 * logL + log(T_lagged) * (n^2 * lag);
            case 'HQIC'
                criteria_values(lag) = -2 * logL + 2 * log(log(T_lagged)) * (n^2 * lag);
            otherwise
                error('Invalid criterion. Choose ''AIC'', ''BIC'', or ''HQIC''.');
        end
    end

    % Find the optimal lag order
    [~, optimal_lag] = min(criteria_values);
end

function [X_lagged, Y] = create_lagged_data(X, lags)
% CREATE_LAGGED_DATA Creates lagged predictors and responses for VAR model.
%
% INPUTS:
%   X    - Matrix of time series data (T x n).
%   lags - Number of lags to include.
%
% OUTPUTS:
%   X_lagged - Lagged predictor matrix ((T-lags) x (lags*n)).
%   Y        - Response matrix ((T-lags) x n).

    [T, n] = size(X);
    X_lagged = [];
    
    % Create lagged predictor matrix
    for lag = 1:lags
        X_lagged = [X_lagged, X(lag:T-lags+lag-1, :)];
    end
    
    % Create response matrix
    Y = X(lags+1:end, :);
end