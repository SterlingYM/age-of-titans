import numpy as np
from SFH_to_SNR_Iyer_tools import process_data_streaming

if __name__ == "__main__":
    file_path = "combined_iyer2019_sfhs.h5"
    list_of_CIDs, SN_age_samples = process_data_streaming(file_path, n_workers=4, chunk_size=300)
    np.savez("SN_age_samples.npz", CIDs=list_of_CIDs, SN_age_samples=SN_age_samples)