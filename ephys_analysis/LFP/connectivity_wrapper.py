from spectral_connectivity import Multitaper, Connectivity


def connectivity_wrapper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    power = calculate_power(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    coherence = calculate_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    granger = calculate_granger(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # pdc = calculate_partial_directed_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    return connectivity, frequencies, power, coherence, granger#, pdc


def calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    multi_t = Multitaper(
        # multitaper takes in a time_series that is time by signals (regions)
        time_series=rms_traces,
        sampling_frequency=downsample_rate,
        time_halfbandwidth_product=halfbandwidth,
        time_window_duration=timewindow,
        time_window_step=timestep,
    )
    connectivity = Connectivity.from_multitaper(multi_t)
    frequencies = connectivity.frequencies
    return connectivity, frequencies


def calculate_power(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # connectivity.power.() = [timebins, frequencies, signal]
    power = connectivity.power()
    print("Power Calculated")
    return power


def calculate_phase():
    return

def calculate_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # calculates a matrix of timebins, frequencies, region, region
    # such that [x,y,a,a] = nan
    # and [x,y,a,b] = [x,y,b,a] which is the coherence between region a & b
    # for frequency y at time x
    coherence = connectivity.coherence_magnitude()
    print("Coherence calcualatd")
    return coherence

# def calculate_partial_directed_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
#     connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
#     pdc = connectivity.partial_directed_coherence()
#     print('Partial Directed Coherence calculated')
#     return pdc
    

def calculate_granger(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # calculates a matrix of timebins, frequencies, region, region
    # such that [x,y,i,j] = nan
    # and [x,y,i,j] =/= [x,y,i,j]
    # [x,y,i,j] -> j to i granger based on the plot_directional code in the following link: bruh how did i misread this the first time 
    # https://spectral-connectivity.readthedocs.io/en/latest/examples/Tutorial_Using_Paper_Examples.html
    # New comment from one of developers: https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues/31
    # granger j --> i 
    granger = connectivity.pairwise_spectral_granger_prediction()
    print("Granger causality calculated")
    return granger


    
