import numpy as np
import pandas as pd
import h5py

H5_FILENAME = "data/SN_age_samples_300Myr_slope-1.07.h5"
OUT_CSV = "host_props_with_SN_age_good_300Myr_slope-1.07.csv"

df_props = pd.read_csv("host_props.csv")
df_reliable = pd.read_csv("all_parameters_reliable.csv").rename(columns={"objname": "transient"})
df_ddlr = pd.read_csv("titan_ddlr.csv").rename(columns={"objname": "transient"})
df_all = pd.read_csv("titan_snia.csv").rename(columns={"objname": "transient"})
print(len(df_props))
df_props = df_props.merge(df_ddlr[["transient", "d_dlr"]], on="transient", how="left")
df_props = df_props.merge(df_all[["transient", "class_flag"]], on="transient", how="left")
df_props = df_props[df_props["transient"].isin(df_reliable["transient"]) & df_props["d_dlr"].le(10)]
df_props = df_props[df_props["class_flag"].le(2)]
print(len(df_props))

CIDs_good = set(df_props["transient"].values)
p16, p50, p84 = {}, {}, {}

with h5py.File(H5_FILENAME, "r") as f:
    n_total = len(f.keys())
    n_done = 0
    for cid in f.keys():
        if cid not in CIDs_good:
            continue
        age = f[cid][:].mean(axis=1)
        p16[cid] = np.nanpercentile(age, 16)
        p50[cid] = np.median(age)
        p84[cid] = np.nanpercentile(age, 84)
        n_done += 1
        if n_done % 500 == 0:
            print(f"  {n_done} matched CIDs processed")

print(f"matched {n_done} of {len(CIDs_good)} good CIDs against {n_total} h5 keys")

df_props["prog_age_16"] = df_props["transient"].map(p16)
df_props["prog_age_50"] = df_props["transient"].map(p50)
df_props["prog_age_84"] = df_props["transient"].map(p84)

df_props.to_csv(OUT_CSV, index=False)
print(f"wrote {OUT_CSV}")
