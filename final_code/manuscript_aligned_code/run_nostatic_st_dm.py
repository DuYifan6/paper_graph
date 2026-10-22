"""Paired daily-loss test; no old project_config or hard-coded prediction paths."""
import argparse
from pathlib import Path
import pandas as pd
from papergraph.config import write_json
from papergraph.statistics import paired_daily,dm_test

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zero-daily',type=Path,required=True)
    p.add_argument('--graph2-daily',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    differences=paired_daily(pd.read_csv(args.graph2_daily),[pd.read_csv(args.zero_daily)])
    args.out.mkdir(parents=True,exist_ok=True)
    differences.to_csv(args.out/'paired_daily.csv',index=False)
    write_json(args.out/'hac_test.json',dm_test(differences.loss_advantage,2))

if __name__=='__main__':main()
