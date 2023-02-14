#%%

from config_manager import base_configuration
from simulation.config_template import ConfigTemplate
import defs as rnn_defs
from simulation.runner import Runner
import os
import contextlib
import numpy as np
import pandas as pd
import torch
import scipy
from simulation.task_data import Task_Params, Task_Dataset

#path = os.path.dirname(os.path.realpath(__file__))

def test_model(config_dir):
    """ 
    Test a trained model

    Parameters
    ----------
    config_dir: str
        directory for configuration file

    """
    #setup runner
    config_path = config_dir + "config.yaml"
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        config = base_configuration.BaseConfiguration(
            configuration= config_path, template = ConfigTemplate.base_config_template)

    #set random seeds 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    runner = Runner(config, rnn_defs.PROJ_DIR, training = False)

    #test
    datadir, output, activity1 = runner.run_test()

    return datadir, output, activity1

def model_to_pyaldata ( ourdir, session: str = None):
#def model_to_pyaldata ():
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
    #outdir = get_outdir(seed, sim)

    datadir = '/home/cf620/git/base_centerout_rnn/simulation/data/Chewie_CO_CS_2016-10-14'

    data = Task_Dataset(datadir, training = False)
    params = Task_Params(datadir)

    #outdir = '/home/cf620/git/base_centerout_rnn/simulation/results/100028/5/'
    datadir, output, activity = test_model(outdir)
    # columns needed for pyaldata
    column_names = ['session', 'target_id', 'trial_id', 'bin_size', 
        'idx_trial_start', 'idx_target_on', 'idx_go_cue', 'idx_trial_end', 
        'MCx_rates']
    df = pd.DataFrame(columns = column_names)
    ntrials = output.shape[0]
    tsteps = params.tsteps
    #populate columns
    df['target_id'] = data.labels
    df['trial_id'] = range(ntrials)
    #df['seed'] = seed
    #df['sim'] = sim
    df['session'] = session or 'init'
    df['bin_size'] = params.dt
    df['idx_trial_start'] = 0
    df['idx_target_on'] = 0
    df['idx_go_cue'] = 300
    df['idx_trial_end'] = tsteps-1
    df['MCx_rates'] =[activity[i,:] for i in range(ntrials)] 
    df['pos'] = [output[i,:] for i in range(ntrials)] 
    return df

if __name__ == "__main__":
    from scipy.io import savemat  
    import os
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    
    for i in range(1,6):
        outdir = '/home/cf620/git/base_centerout_rnn/simulation/results/100028/'+str(i)+'/'
        print(outdir)
        df = model_to_pyaldata(outdir)

        a_dict = {col_name : df[col_name].values for col_name in df.columns.values}
    
        scipy.io.savemat(outdir+"/pydata.mat", {'struct':a_dict})


# %%
