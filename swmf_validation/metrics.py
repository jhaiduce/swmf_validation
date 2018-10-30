import numpy as np

def interp_timeseries(data,oldtimes,newtimes):
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
    return forecast-obs

def mean_error(forecast,obs):
    return np.ma.mean(forecast-obs)

def mean_error_e(forecast,obs):
    e=error(forecast,obs)
    return np.ma.std(e)/np.sqrt(np.ma.count(e))

def mean_error_stdnorm(forecast,obs):
    return np.ma.mean(forecast-obs)/np.ma.std(obs)

def mean_error_stdnorm_e(forecast,obs):
    e=error(forecast,obs)
    return np.ma.std(e)/np.sqrt(np.ma.count(e))/np.ma.std(obs)

def mean_squared_error(forecast,obs):
    return np.ma.mean((forecast-obs)**2)

def root_mean_squared_error(forecast,obs):
    return np.ma.sqrt(mean_squared_error(forecast,obs))

def root_mean_squared_error_e(forecast,obs):
    e=error(forecast,obs)
    return np.ma.sqrt(np.ma.std(e**2)/np.sqrt(np.ma.count(e)))

def root_mean_squared_error_stdnorm(forecast,obs):
    return np.ma.sqrt(mean_squared_error(forecast,obs))/np.ma.std(obs)

def root_mean_squared_error_stdnorm_e(forecast,obs):
    e=error(forecast,obs)
    return np.ma.sqrt(np.ma.std(e**2)/np.sqrt(np.ma.count(e)))/np.ma.std(obs)

def mean_absolute_error(forecast,obs):
    return np.ma.mean(np.abs(forecast-obs))

def median_absolute_error(forecast,obs):
    return np.ma.median(np.abs(forecast-obs))

def magnitude_of_relative_error(forecast,obs):
    return np.abs((forecast-obs)/obs)

def mean_absolute_percentage_error(forecast,obs):
    nonzero=(obs!=0)
    return 100*np.ma.mean(magnitude_of_relative_error(forecast,obs)[nonzero])

def scaled_error(forecast,obs,relative=None):
    if relative is None:
        relative=obs[:-1]
    return (obs-forecast)/(np.ma.mean(obs[1:]-relative))

def mean_absolute_scaled_error(forecast,obs,relative=None):
    return np.ma.mean(np.ma.abs(scaled_error(forecast,obs,relative)))

def relative_error(forecast,obs):
    return np.ma.array(forecast-obs)/np.ma.abs(obs)

def mean_relative_error(forecast,obs):
    return np.ma.mean(((forecast-obs)/np.ma.abs(obs)).compressed())

def mean_relative_error_e(forecast,obs):
    e_rel=error(forecast,obs)/np.ma.abs(obs)
    return np.ma.std(e_rel)/np.sqrt(np.ma.count(e_rel))

def mean_magnitude_relative_error(forecast,obs):
    mmre=np.ma.mean(np.ma.abs((forecast-obs)/obs))
    return mmre

def mean_magnitude_relative_error_e(forecast,obs):
    e_rel_abs=np.ma.abs(error(forecast,obs)/obs)
    return np.ma.std(e_rel_abs.compressed())/np.sqrt(np.ma.count(e_rel_abs))

def magnitude_relative_error(forecast,obs):
    return np.ma.abs((forecast-obs)/obs)

def accuracy_ratio(forecast,obs):
    return forecast/obs

def MdLQ(forecast,obs):
    return np.ma.median(np.log10(accuracy_ratio(forecast,obs)))

def symmetric_accuracy(forecast,obs):
    return 100*(np.ma.exp(np.ma.median(np.ma.abs(np.ma.log10(accuracy_ratio(forecast,obs)))))-1)

def median_accuracy_ratio(forecast,obs):
    ar=np.ma.MaskedArray(accuracy_ratio(forecast,obs)).compressed()
    if len(ar)==0:
        return np.nan
    else:
        return np.percentile(ar,50)

def median_log_accuracy_ratio(forecast,obs):
    return np.ma.median(np.ma.log10(accuracy_ratio(forecast,obs)))

def geometric_mean_accuracy_ratio(forecast,obs):
    return np.ma.exp(np.ma.mean(np.ma.log10(accuracy_ratio(forecast,obs))))

def median_symmetric_accuracy(forecast,obs):
    return 100*(np.ma.exp(np.ma.median(np.ma.abs(np.ma.log10(forecast/obs))))-1)
