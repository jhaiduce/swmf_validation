import numpy as np

def interp_timeseries(data,oldtimes,newtimes):
    """
    Interpolates a timeseries dataset to a new set of times

    data: Data to be interpolated
    oldtimes: Times associated with the data (sequence of datetime objects having same length as data)
    newtimes: Sequence of times onto which the data should be interpolated

    Returns: Interpolated data
    """
    
    from scipy.interpolate import interp1d

    if len(oldtimes)==0:
        interp=np.empty([len(newtimes)])
        interp.fill(np.nan)
        return interp

    epoch=oldtimes[0]

    oldtimes_f=[(t-epoch).total_seconds() for t in oldtimes]
    newtimes_f=[(t-epoch).total_seconds() for t in newtimes]

    interpolator=interp1d(oldtimes_f,data,fill_value=np.nan,bounds_error=False)

    return interpolator(newtimes_f)

def error(forecast,obs):
    """
    Compute the errors of the forecast values (forecast-obs)
    """
    return forecast-obs

def mean_error(forecast,obs):
    """
    Compute mean error of the forecast values (i.e., mean(forecast-obs) )
    """
    return np.ma.mean(forecast-obs)

def mean_error_e(forecast,obs):
    """
    Compute uncertainty of the mean error
    """
    
    e=error(forecast,obs)
    return np.ma.std(e)/np.sqrt(np.ma.count(e))

def mean_error_stdnorm(forecast,obs):
    """
    Compute the mean error, normalized by the standard deviation of the observed values.
    """
    
    return np.ma.mean(forecast-obs)/np.ma.std(obs)

def mean_error_stdnorm_e(forecast,obs):
    """
    Uncertainty of mean error, normalized by the standard deviation of the observed values.
    """
    
    e=error(forecast,obs)
    return np.ma.std(e)/np.sqrt(np.ma.count(e))/np.ma.std(obs)

def mean_squared_error(forecast,obs):
    """
    Compute mean squared error of the forecast
    """
    
    return np.ma.mean((forecast-obs)**2)

def root_mean_squared_error(forecast,obs):
    """
    Compute root mean squared error (RMSE) of the forecast
    """
    
    return np.ma.sqrt(mean_squared_error(forecast,obs))

def root_mean_squared_error_e(forecast,obs):
    """
    Compute uncertainty of root mean squared error (RMSE) for the forecast
    """
    
    e=error(forecast,obs)
    return np.ma.sqrt(np.ma.std(e**2)/np.sqrt(np.ma.count(e)))

def root_mean_squared_error_stdnorm(forecast,obs):
    """
    Compute root mean squared error (RMSE) of the forecast, normalized by the standard deviation of the observed values
    """
    
    return np.ma.sqrt(mean_squared_error(forecast,obs))/np.ma.std(obs)

def root_mean_squared_error_stdnorm_e(forecast,obs):
    """
    Compute uncertainty of root mean squared error (RMSE) of the forecast, normalized by the standard deviation of the observed values
    """

    e=error(forecast,obs)
    return np.ma.sqrt(np.ma.std(e**2)/np.sqrt(np.ma.count(e)))/np.ma.std(obs)

def mean_absolute_error(forecast,obs):
    """
    Compute mean absolute error
    """
    
    return np.ma.mean(np.abs(forecast-obs))

def median_absolute_error(forecast,obs):
    """
    Compute median absolute error
    """
    
    return np.ma.median(np.abs(forecast-obs))

def magnitude_of_relative_error(forecast,obs):
    """
    Compute magnitude of relative error
    """
    
    return np.abs((forecast-obs)/obs)

def mean_absolute_percentage_error(forecast,obs):
    """
    Compute mean absolute percentage error (MAPE)
    """

    nonzero=(obs!=0)
    return 100*np.ma.mean(magnitude_of_relative_error(forecast,obs)[nonzero])

def scaled_error(forecast,obs,relative=None):
    """
    Compute scaled error

    forecast: Forecast values
    obs: Observed values
    relative: Offset values used to compute normalization. Default value is None, which results in a 1-step persistence forecast being used for the normalization.
    """
    
    if relative is None:
        relative=obs[:-1]
    return (obs-forecast)/(np.ma.mean(obs[1:]-relative))

def mean_absolute_scaled_error(forecast,obs,relative=None):
    """
    Compute scaled error

    forecast: Forecast values
    obs: Observed values
    relative: Offset values used to compute normalization. Default value is None, which results in a 1-step persistence forecast being used for the normalization.
    """

    return np.ma.mean(np.ma.abs(scaled_error(forecast,obs,relative)))

def relative_error(forecast,obs):
    """
    Compute relative error
    """

    return np.ma.array(forecast-obs)/np.ma.abs(obs)

def mean_relative_error(forecast,obs):
    """
    Compute mean relative error
    """

    return np.ma.mean(((forecast-obs)/np.ma.abs(obs)).compressed())

def mean_relative_error_e(forecast,obs):
    """
    Compute uncertainty of mean relative error
    """
    
    e_rel=error(forecast,obs)/np.ma.abs(obs)
    return np.ma.std(e_rel)/np.sqrt(np.ma.count(e_rel))

def mean_magnitude_relative_error(forecast,obs):
    """
    Compute mean magnitude of relative error (MMRE)
    """
    
    mmre=np.ma.mean(np.ma.abs((forecast-obs)/obs))
    return mmre

def mean_magnitude_relative_error_e(forecast,obs):
    """
    Compute uncertainty of mean magnitude of relative error (MMRE)
    """
    
    e_rel_abs=np.ma.abs(error(forecast,obs)/obs)
    return np.ma.std(e_rel_abs.compressed())/np.sqrt(np.ma.count(e_rel_abs))

def magnitude_relative_error(forecast,obs):
    """
    Compute magnitude of relative error
    """
    
    return np.ma.abs((forecast-obs)/obs)

def accuracy_ratio(forecast,obs):
    """
    Compute accuracy ratio Q
    """
    
    return forecast/obs

def MdLQ(forecast,obs):
    """
    Compute median log accuracy ratio (MdLQ)
    """
    
    return np.ma.median(np.log10(accuracy_ratio(forecast,obs)))

def symmetric_accuracy(forecast,obs):
    """
    Compute symmetric accuracy
    """
    
    return 100*(np.ma.exp(np.ma.median(np.ma.abs(np.ma.log10(accuracy_ratio(forecast,obs)))))-1)

def median_accuracy_ratio(forecast,obs):
    """
    Compute median of the accuracy ratio Q
    """
    
    ar=np.ma.MaskedArray(accuracy_ratio(forecast,obs)).compressed()
    if len(ar)==0:
        return np.nan
    else:
        return np.percentile(ar,50)

def median_log_accuracy_ratio(forecast,obs):
    """
    Compute median log accuracy ratio
    """
    
    return np.ma.median(np.ma.log10(accuracy_ratio(forecast,obs)))

def geometric_mean_accuracy_ratio(forecast,obs):
    """
    Compute the geometric mean of the accuracy ratio
    """
    
    return np.ma.exp(np.ma.mean(np.ma.log10(accuracy_ratio(forecast,obs))))

def median_symmetric_accuracy(forecast,obs):
    """
    Compute median symmetric accuracy
    """
    
    return 100*(np.ma.exp(np.ma.median(np.ma.abs(np.ma.log10(forecast/obs))))-1)
