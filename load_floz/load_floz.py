import zipfile
import pandas as pd
import yaml
import pytz


def load_floz(filename):
    fileobj = open(filename, mode="rb")

    with zipfile.ZipFile(file=fileobj, mode="r") as archive:
        centroid_csv = "centroid.csv"
        centroid_csv_info = archive.getinfo(centroid_csv)
        if centroid_csv_info.file_size > 0:
            centroid_df = pd.read_csv(
                archive.open(centroid_csv),
                comment="#",
            )
        else:
            centroid_df = None

        tracking_state_df = pd.read_csv(
            archive.open("tracking_state.csv"),
            dtype={"centroid_timestamp": object},
            comment="#",
        )


        motor_positions_df = pd.read_csv(
            archive.open("motor_positions.csv"),
            comment="#",
        )

        metadata = yaml.safe_load(archive.open("flo-metadata.yaml").read())

    if "timezone" not in metadata:
        # Prior to adding this field to metadata, all data acquired
        # in Germany.
        metadata["timezone"] = "Europe/Berlin"

    # put times into local timezone
    tz_pytz = pytz.timezone(metadata["timezone"])
    dfs_to_fix = [tracking_state_df, motor_positions_df]
    if centroid_df is not None:
        dfs_to_fix.append( centroid_df )
    for df in dfs_to_fix:
        for colname in df.columns:
            if not (colname.endswith("timestamp") or colname == "local"):
                continue
            stamps = pd.to_datetime(df[colname], utc=True)
            stamps = stamps.dt.tz_convert(tz_pytz)
            df[colname] = stamps

    if centroid_df is not None:
        # calculate X and Y centroids.
        centroid_df["x"] = centroid_df.mu10 / centroid_df.mu00
        centroid_df["y"] = centroid_df.mu01 / centroid_df.mu00
    return {
        "centroid_df": centroid_df,
        "tracking_state_df": tracking_state_df,
        "motor_positions_df": motor_positions_df,
        "metadata": metadata,
    }
