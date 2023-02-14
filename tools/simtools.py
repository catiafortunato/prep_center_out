from simulation.test import test_model
import simulation.defs as rnn_defs
import pandas as pd
import numpy as np

def get_outdir(seed, sim_number):
    outdir = rnn_defs.PROJ_DIR +rnn_defs.RESULTS_FOLDER + str(seed) + '/' + str(sim_number) + '/' 
    return outdir

def model_to_pyaldata (seed: int, sim: int):
    """
    Converts model results to Pyaldata format.
    Parameters
    ----------
    seed: random seeds for each network
    sim: number associated with simulation parameters
    Returns
    -------
    df: in pyaldata format
    """
    #get directories and model output
    outdir = get_outdir(seed, sim)
    datadir, output, activity = test_model(outdir)

    datname = rnn_defs.PROJ_DIR + datadir
    data = np.load(datname+'.npy',allow_pickle = True).item()
    test_data = data['test_set1']

    # columns needed for pyaldata
    column_names = ['seed', 'sim', 'session', 'target_id', 'trial_id', 'bin_size', 
        'idx_trial_start', 'idx_target_on', 'idx_go_cue', 'idx_trial_end', 
        'MCx_rates']
    df = pd.DataFrame(columns = column_names)
    ntrials = output.shape[0]
    tsteps = rnn_defs.TSTEPS
    #populate columns
    df['target_id'] = test_data['target_id']
    df['trial_id'] = range(ntrials)
    df['seed'] = seed
    df['sim'] = sim
    df['bin_size'] = rnn_defs.DT
    df['idx_trial_start'] = 0
    df['idx_target_on'] = 0
    df['idx_go_cue'] = rnn_defs.GO_CUE
    df['idx_trial_end'] = tsteps-1
    df['MCx_rates'] =[activity[i,:] for i in range(ntrials)] 
    df['pos'] = [output[i,:] for i in range(ntrials)] 
    return df