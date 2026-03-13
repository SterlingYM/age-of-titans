import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from p_tqdm import p_umap
from tqdm import tqdm
import pandas as pd


def DTD(t_Gyr,R1,alpha):
    return R1 * (t_Gyr)**alpha

def SFH_to_SN_age_PDF(sfh, lookback_times, 
              R1=1.0, alpha=-1.07, 
              time_grid_Gyr=np.arange(0.04, 14, 0.01),
              normalize=True):
    ''' grab an SFH and convert it to an SN age PDF using DTD.
    '''
    DTD_values = DTD(time_grid_Gyr, R1, alpha)
    
    # interpolate the SFH onto the new time grid
    sfh_interp = np.interp(time_grid_Gyr, lookback_times, sfh)
    
    # apply the DTD to the interpolated SFH
    SN_age_pdf = sfh_interp * DTD_values
    if normalize:
        SN_age_pdf /= np.trapezoid(SN_age_pdf, time_grid_Gyr)  # Normalize the PDF
    return SN_age_pdf

def sample_from_PDF(time_grid_Gyr, SN_age_pdf, N_sample_per_SFH=100):
    ''' Sample from a given PDF.
    '''
    cdf = cumulative_trapezoid(SN_age_pdf, time_grid_Gyr, initial=0)
    cdf /= cdf[-1]

    # inverse CDF interpolator
    inv_cdf = interp1d(cdf, time_grid_Gyr, kind='linear', bounds_error=False,
                    fill_value=(time_grid_Gyr[0], time_grid_Gyr[-1]))
    u = np.random.random(N_sample_per_SFH)
    posterior_age_sample = inv_cdf(u)
    return posterior_age_sample

def helper(sfh_samples, lookback_times):
    ''' turn a list of SFHs into a list of SN age samples.
    '''
    sampled_ages = []
    for i in np.random.choice(len(sfh_samples), size=1000, replace=True):
        sfh = sfh_samples[i]
        time_grid_Gyr = np.arange(0.04, 14, 0.01)
        SN_age_pdf = SFH_to_SN_age_PDF(sfh, lookback_times, time_grid_Gyr=time_grid_Gyr)
        sampled_ages.append(sample_from_PDF(time_grid_Gyr, SN_age_pdf))
    return np.array(sampled_ages)

def load_h5(file_path):
    # Read all data into memory before parallelizing (h5py objects can't be pickled)
    with h5py.File(file_path, "r") as hf:
        print("Keys in HDF5 file:")
        for key in hf.keys():
            print(f" - {key}")

        data = []
        for CID in hf.keys():
            g = hf[CID]
            data.append((CID, g["sfh_samps"][:], g["lookback_times_gyr"][:]))

    return data

def process_data(data):
    def _worker(args):
        CID, sfh_samps, lookback_times = args
        return CID, helper(sfh_samps, lookback_times)

    results = p_umap(_worker, data)
    list_of_CIDs, list_of_SN_samps = zip(*results)
    return list_of_CIDs, np.array(list_of_SN_samps)

def _streaming_worker(args):
    CID, sfh_samps, lookback_times = args
    return CID, helper(sfh_samps, lookback_times)

def process_data_streaming(file_path, n_workers=4, chunk_size=50, output_path=None):
    """Process HDF5 data in chunks to avoid loading the full file into RAM.

    Reads `chunk_size` galaxies at a time, processes each chunk with a pool,
    and accumulates results — keeping only one chunk in memory at once.

    If `output_path` is given, results are saved incrementally to an HDF5 file
    after each chunk. Any CIDs already present in that file are skipped, so the
    run can be resumed after an interruption.
    """
    import multiprocessing as mp

    # Load already-processed CIDs from a previous (partial) run.
    completed_CIDs = set()
    if output_path is not None:
        import os
        if os.path.exists(output_path):
            with h5py.File(output_path, "r") as out_hf:
                completed_CIDs = set(out_hf.keys())
            print(f"Resuming: {len(completed_CIDs)} CIDs already processed.")

    list_of_CIDs = []
    list_of_SN_samps = []

    with h5py.File(file_path, "r") as hf:
        all_keys = [k for k in hf.keys() if k not in completed_CIDs]
        print(f"Galaxies to process: {len(all_keys)}")

        chunks = range(0, len(all_keys), chunk_size)
        with mp.Pool(n_workers) as pool:
            for start in tqdm(chunks, desc="Processing galaxies", unit="chunk"):
                chunk_keys = all_keys[start : start + chunk_size]
                chunk = []
                for CID in chunk_keys:
                    try:
                        chunk.append((CID, hf[CID]["sfh_samps"][:], hf[CID]["lookback_times_gyr"][:]))
                    except KeyError as e:
                        print(f"  Skipping {CID}: missing dataset {e}")
                if not chunk:
                    continue
                results = pool.map(_streaming_worker, chunk)

                if output_path is not None:
                    with h5py.File(output_path, "a") as out_hf:
                        for CID, sn_samps in results:
                            out_hf.create_dataset(CID, data=sn_samps)

                for CID, sn_samps in results:
                    list_of_CIDs.append(CID)
                    list_of_SN_samps.append(sn_samps)

    return list_of_CIDs, np.array(list_of_SN_samps)