import logging
import re

import pandas as pd

from brainfeatures.data_set.tuh_abnormal import _read_all_file_names


def merge_two_columns(df, to_replace, replace_by, drop_replaced_column=False):
    assert to_replace != replace_by, "columns are equal"
    for column in [to_replace, replace_by]:
        if column not in df:
            logging.warning("did not find column {}".format(column))
            return df

    for i in range(len(df)):
        if not pd.isnull(df[replace_by].iloc[i]) and not pd.isnull(
                df[to_replace].iloc[i]):
            logging.error("columns intersect at {}".format(i))
            continue

        if pd.isnull(df[to_replace].iloc[i]):
            df[to_replace].iloc[i] = df[replace_by].iloc[i]

    if drop_replaced_column:
        df = df.drop(columns=replace_by)
    return df


# TODO: merge all the columns that should obviously the same but are not
# TODO: due to typos
def merge_all_columns(df):
    columns_to_be_merged = [
        ["CLINICALHISTORY", "HISTORY"],
        ["CLINICALCORRELATION", "CLINICALINTERPRETATION"]
    ]
    for to_replace, replace_by in columns_to_be_merged:
        df = merge_two_columns(df, to_replace, replace_by,
                               drop_replaced_column=True)
    return df


def main(report_parent_dir, n_reports=None):
    reports = _read_all_file_names(report_parent_dir, extension=".txt",
                                   key="time")

    categoy_pattern = "^([A-Z\s]{2,}):"
    content_pattern = ":?(.*)"
    df = pd.DataFrame()
    for i, report in enumerate(reports):
        if n_reports is not None and i == n_reports:
            break

        try:
            with open(report, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # TODO: handle unicode decode error
            continue
            # with open(report, "r", encoding="latin-1") as f:
            #     content = f.read()

        content = content.strip()
        categories = re.findall(categoy_pattern, content, re.MULTILINE)

        row = ["/".join(report.split("/")[-9:])]
        header = ["REPORT"]
        for j in range(len(categories) - 1):
            start = categories[j]
            stop = categories[j + 1]
            match = re.findall(start + content_pattern + stop, content,
                               re.DOTALL)
            row.append(match[0].strip())
            header.append(re.sub(r"\r\n|\t", "", start).strip().replace(" ",
                                                                        ""))
        match = re.findall(stop + content_pattern, content, re.DOTALL)
        row.append(match[0].strip())
        header.append(re.sub(r"\r\n|\t", " ", stop).strip().replace(" ", ""))
        try:
            df = df.append(pd.DataFrame(data=[row], columns=header),
                           ignore_index=True)
        except (ValueError, AssertionError):
            # TODO: handle multiple occurences of same category
            continue

    df = merge_all_columns(df)
    return df


if __name__ == "__main__":
    report_parent_dir = "/data/schirrmr/gemeinl/tuh-abnormal-eeg/raw/v2.0.0/"
    main(report_parent_dir)
