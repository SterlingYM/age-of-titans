from SFH_to_SNR_Iyer_tools import process_data_streaming

if __name__ == "__main__":
    file_path = "combined_iyer2019_sfhs.h5"
    output_path = "SN_age_samples.h5"
    process_data_streaming(file_path, n_workers=4, chunk_size=300, output_path=output_path)