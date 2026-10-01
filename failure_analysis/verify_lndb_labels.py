"""Verify supplied labels against official LNDb v4 average suspicion scores.

This reconstructs the observed label mapping; it is not the missing original
crop extraction or evaluation-sampling script. No reader aggregation is redone.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
import config
from utils import ensure_output_dirs, write_table, write_text

URL="https://zenodo.org/api/records/8348419/files/allNods.csv/content"
SHA256="bfdb261199323f86ff2fb96b0ece785955f8c6198b3542c1d076a2a6e576c3b8"
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official",type=Path,default=os.environ.get("FAILURE_ANALYSIS_LNDB_OFFICIAL_CSV"))
    parser.add_argument("--metadata",type=Path,default=config.LNDB_METADATA_PATH)
    args=parser.parse_args()
    raw=Path(args.official).read_bytes() if args.official else urlopen(URL,timeout=30).read()
    if hashlib.sha256(raw).hexdigest()!=SHA256: raise ValueError("Unexpected official LNDb v4 file hash")
    official=pd.read_csv(io.BytesIO(raw),dtype={"LNDbID":str,"FindingID":str})
    official["global_id"]=official.LNDbID+"_"+official.FindingID
    if official.global_id.duplicated().any(): raise ValueError("Nonunique official composite ID")
    meta=pd.read_csv(args.metadata,dtype={"FindingID":str,"LNDbID":str})
    if meta.groupby("FindingID")[["LNDbID","label"]].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Conflicting metadata repeats")
    unique=meta.drop_duplicates("FindingID").merge(official[["global_id","Malignancy"]],
        left_on="FindingID",right_on="global_id",how="left",validate="one_to_one")
    if unique.Malignancy.isna().any(): raise ValueError("Unmatched official labels")
    if ((unique.Malignancy>2)&(unique.Malignancy<4)).any(): raise ValueError("Intermediate score in binary audit")
    expected=unique.Malignancy.ge(4).astype(int)
    if not expected.eq(unique.label).all(): raise ValueError("Supplied labels disagree with score extremes")
    ensure_output_dirs()
    write_table(unique,config.LOG_DIR/"lndb_official_label_matches.csv",latex=False)
    result={"record":"LNDb Dataset v4", "doi":"10.5281/zenodo.8348419", "source_url":URL,
            "official_sha256":SHA256, "label_field":"allNods.Malignancy",
            "aggregation":"Official average radiologist suspicion score used as supplied; no new reader averaging",
            "thresholds":{"negative_max_inclusive":2,"positive_min_inclusive":4,"intermediate_excluded":True},
            "unique_selected":len(unique),"negative":int(expected.eq(0).sum()),"positive":int(expected.sum()),
            "weighted_input_rows":len(meta),"label_mismatch_n":0,
            "id_construction":"LNDbID + '_' + merged within-CT FindingID",
            "original_sampling_reproduced":False,"pathology_confirmation_established":False}
    write_text(config.LOG_DIR/"lndb_label_verification.json",json.dumps(result,indent=2)+"\n")
    print(f"Verified {len(unique)} LNDb labels against hash-pinned official v4 suspicion scores.")
if __name__=="__main__": main()
