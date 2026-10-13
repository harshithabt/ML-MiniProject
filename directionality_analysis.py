import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.records import process_folder
from src.metadata import load_metadata
from src.pairs import build_pairs
from src.models import evaluate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--at2",required=True); ap.add_argument("--flatfile",required=True)
    ap.add_argument("--out",default="outputs"); ap.add_argument("--max-km",type=float,default=100)
    args=ap.parse_args(); out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    records,issues=process_folder(args.at2)
    if records.empty: raise RuntimeError("No valid paired AT2 records; inspect output/filename azimuths")
    records.to_csv(out/'orientations.csv',index=False)
    issues.to_csv(out/'excluded_records.csv',index=False)
    metadata=load_metadata(args.flatfile)
    merged=records.merge(metadata,on="rsn",how="inner")
    merged.to_csv(out/'records_with_metadata.csv',index=False)
    pairs=build_pairs(merged,args.max_km)
    pairs.to_csv(out/'station_pairs.csv',index=False)
    summary={"at2_record_count":len(records),"excluded_rsn_count":len(issues),
             "metadata_matched_count":len(merged),"pair_count":len(pairs),
             "pair_events":int(pairs.event_id.nunique()) if not pairs.empty else 0,
             "max_station_separation_km":args.max_km,
             "note":"Exploratory replication, not verified paper-equivalent target or sampling."}
    if not pairs.empty:
        plt.figure(figsize=(6,4));plt.scatter(pairs.station_separation_km,pairs.delta_orientation_deg,alpha=.55)
        plt.xlabel("Station separation (km)");plt.ylabel("Absolute axial angle difference (degrees)")
        plt.tight_layout();plt.savefig(out/'distance_vs_angle.png',dpi=180);plt.close()
    try:
        metrics,pred=evaluate(pairs)
        metrics.to_csv(out/'metrics.csv',index=False)
        pred.to_csv(out/'predictions.csv',index=False)
        summary['metrics']=metrics.to_dict('records')
    except ValueError as exc:
        summary['model_status']=str(exc)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print(json.dumps(summary,indent=2,default=str))
if __name__=="__main__":main()
