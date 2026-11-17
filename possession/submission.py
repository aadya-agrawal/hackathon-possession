import csv
import os

import numpy as np

from possession import LABELS


def write_submission(path, codes):
    invalid = np.setdiff1d(np.unique(codes), np.arange(len(LABELS)))
    if len(invalid):
        raise ValueError("cannot write unknown label codes: {}".format(invalid.tolist()))
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows((frame, LABELS[int(code)]) for frame, code in enumerate(codes))


def validate_submission_file(path, expected_rows):
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != ["frame", "label"]:
            raise ValueError("{}: expected header frame,label".format(path))
        count = 0
        for count, row in enumerate(reader, start=1):
            if len(row) != 2:
                raise ValueError("{}: row {} must have two columns".format(path, count + 1))
            expected_frame = count - 1
            if int(row[0]) != expected_frame:
                raise ValueError(
                    "{}: expected frame {}, found {}".format(path, expected_frame, row[0])
                )
            if row[1] not in LABELS:
                raise ValueError("{}: invalid label {!r}".format(path, row[1]))
    if count != expected_rows:
        raise ValueError("{}: expected {} rows, found {}".format(path, expected_rows, count))


def summarize(codes, fps):
    counts = {LABELS[index]: int(np.count_nonzero(codes == index))
              for index in range(len(LABELS))}
    live = codes != 2
    starts = live & np.concatenate(([True], codes[1:] != codes[:-1]))
    possession_starts = np.flatnonzero(starts)
    durations = []
    for start in possession_starts:
        end = start + 1
        while end < len(codes) and codes[end] == codes[start]:
            end += 1
        durations.append((end - start) / float(fps))
    return {
        "counts": counts,
        "possessions": len(durations),
        "mean_possession_seconds": float(np.mean(durations)) if durations else 0.0,
        "shorter_than_one_second": int(np.count_nonzero(np.asarray(durations) < 1.0)),
    }
