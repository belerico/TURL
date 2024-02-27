from __future__ import annotations

import sys

sys.path.append("..")

import ast
import json
import os
import pickle
import time
import urllib.parse
import urllib.request
from operator import add, itemgetter
from typing import Any, Dict, List
from urllib.parse import unquote

import findspark
import pandas as pd
from pandarallel import pandarallel
from tqdm import tqdm

from src.data_loader.el_data_loaders import ELDataset
from src.utils.util import load_dbpedia_type_vocab

findspark.init()

import pyspark
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import Row

pandarallel.initialize(progress_bar=True, nb_workers=64, use_memory_fs=True)


def wikidata_lookup(query: Any, retry: int = 3, dbpedia_types: Dict[str, List[str]] | None = None):
    service_url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&search={}&language=en&limit=50&format=json"
    )
    if query != "":
        try:
            url = service_url.format(urllib.parse.quote(str(query)))
        except Exception:
            print(query)
            return [query, []]
        for _ in range(retry):
            try:
                response = urllib.request.urlopen(url)
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code == 503:
                    response = e.code
                    time.sleep(1)
                    continue
                else:
                    response = e.code
                    break
            except urllib.error.URLError as e:
                response = None
                break
            else:
                response = json.loads(response.read())
                break
        if isinstance(response, dict):
            response = [
                [
                    z.get("id"),
                    z.get("label", ""),
                    z.get("description", ""),
                    dbpedia_types.get(z.get("id"), []) if dbpedia_types is not None else [],
                ]
                for z in response.get("search", [])
            ]
        else:
            response = []
    else:
        response = []
    return [query, response]


def lamapi_lookup(
    query: Any,
    retry: int = 3,
    dbpedia_types: Dict[str, List[str]] | None = None,
    fuzzy: bool = False,
    ngrams: bool = False,
):
    service_url = "http://149.132.176.50:8097/lookup/entity-retrieval?name={}&limit=100&token=insideslab-lamapi-2022&kg=wikidata&fuzzy={}&ngrams={}"
    if query != "":
        try:
            url = service_url.format(urllib.parse.quote(str(query)), fuzzy, ngrams)
        except Exception:
            print(query)
            return [query, []]
        for _ in range(retry):
            try:
                response = urllib.request.urlopen(url)
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code == 503:
                    response = e.code
                    time.sleep(1)
                    continue
                else:
                    response = e.code
                    break
            except urllib.error.URLError as e:
                response = None
                break
            else:
                response = json.loads(response.read())
                break
        if isinstance(response, dict):
            response = [
                [
                    z.get("id"),
                    z.get("name", ""),
                    z.get("description", ""),
                    dbpedia_types.get(z.get("id"), []) if dbpedia_types is not None else [],
                    z.get("es_score", 0.0),
                    z.get("ed_score", 0.0),
                    z.get("cosine_similarity", 0.0),
                ]
                for z in response.get(str(query).lower(), [])
            ]
            response = sorted(response, key=lambda l: (float(l[-3]), float(l[-2]), float(l[-1])), reverse=True)
            response = response[:50]
            response = [z[:-3] for z in response]
        else:
            response = []
    else:
        response = []
    return [query, response]


def wikidata_description_from_qids(qid: str, retry: int = 3) -> str:
    service_url = "https://www.wikidata.org/w/api.php?action=wbgetentities&ids={}&languages=en&format=json"
    url = service_url.format(urllib.parse.quote(qid))
    if qid.lower() == "nil":
        return ""
    for _ in range(retry):
        try:
            response = urllib.request.urlopen(url)
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code == 503:
                response = e.code
                time.sleep(1)
                continue
            else:
                response = e.code
                break
        except urllib.error.URLError as e:
            response = None
            break
        else:
            response = json.loads(response.read())
            break
    if isinstance(response, dict):
        try:
            desc = response.get("entities", "")[qid].get("descriptions", {}).get("en", {}).get("value", "")
        except Exception:
            print(response)
            raise Exception()
    else:
        desc = ""
    return desc


if __name__ == "__main__":
    # Arguments parser
    import argparse

    parser = argparse.ArgumentParser(description="Prepare SemTab data")
    parser.add_argument(
        "--table",
        type=str,
        default="Round4_2020",
        help="The name of the table to process",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="~/semtab-data/raw/Round4_2020/gt/cea.csv",
        help="The path to the file containing the ground-truth",
    )
    parser.add_argument(
        "--tables_folder",
        type=str,
        default="~/semtab-data/raw/Round4_2020/tables/",
        help="The path to the real tables folder",
    )
    parser.add_argument(
        "--lookup",
        type=str,
        default="wikidata",
        help="The lookup service to use",
    )
    parser.add_argument(
        "--insert_target_mention_in_candidates",
        action="store_true",
        help="Whether to insert the target mention in the candidates list",
    )
    parser.add_argument(
        "--load_dbpedia_types_dict_from",
        type=str,
        default="",
        help="The path to load the dbpedia types dictionary. If not specified, the dictionary will be created from "
        "the dbpedia types file specified in `--dbpedia_types_path`",
    )
    parser.add_argument(
        "--save_dbpedia_types_dict_to",
        type=str,
        default="",
        help="The path to save the dbpedia types dictionary. If not specified, the dictionary will not be saved",
    )
    parser.add_argument(
        "--dbpedia_types_path",
        type=str,
        default="~/turl-data/dbpedia_types/2019_08_30/instance_type_en.ttl",
        help="The path to the dbpedia types file",
    )
    args = parser.parse_args()

    # Init spark session
    print("Initializing Spark session")
    conf = pyspark.SparkConf().setAll(
        [
            ("spark.executor.memory", "8g"),
            ("spark.executor.cores", "2"),
            ("spark.executor.instances", "7"),
            ("spark.driver.memory", "150g"),
            ("spark.driver.maxResultSize", "100g"),
            ("spark.driver.extraClassPath", "~/Downloads/sqlite-jdbc-3.36.0.3.jar"),
        ]
    )
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # The lookup service to use
    lookup = args.lookup

    # The name of the table to process
    table = args.table
    table_type = table.lower()

    # The path to the file containing the ground-truth
    # gt_path = "~/semtab-data/raw/{}/gt/CEA_Round1_gt_WD.csv".format(table)
    # gt_path = "~/semtab-data/raw/HardTablesR2/gt/cea.csv"
    # gt_path = "~/semtab-data/raw/HardTablesR3/gt/cea.csv"
    # gt_path = "~/semtab-data/raw/2T_Round4/gt/cea.csv"
    # gt_path = "~/semtab-data/raw/Round3_2019/gt/CEA_Round3_gt_WD.csv"
    gt_path = args.gt_path

    # The path to the real tables folder
    tables_folder = args.tables_folder

    # Expand the paths
    tables_folder = os.path.expanduser(tables_folder)
    gt_path = os.path.expanduser(gt_path)

    # The `names` could be different for different datasets
    gt_df = pd.read_csv(gt_path, encoding="utf-8", names=["tableName", "row", "col", "id"])

    # Pre-process the ground-truth dataset
    try:
        gt_df["id"] = gt_df["id"].astype(str).apply(lambda x: [qid.split("/")[-1] for qid in x.split()][0])
    except Exception as e:
        print(
            "Trying to retrieve the QID from the wikidata URL, but something has failed. An example of the QID is: {}".format(
                gt_df.loc[0, "id"]
            )
        )
        raise e
    gt_df["tableName"] = gt_df["tableName"].astype(str)
    gt_df["row"] = gt_df["row"].astype(int) - 1  # Do not consider the header
    gt_df["col"] = gt_df["col"].astype(int)

    # Get the mentions for each table
    print("Getting the mentions for each table")
    table_names = gt_df["tableName"].unique()
    for table_name in tqdm(table_names):
        table_path = os.path.join(tables_folder, table_name + ".csv")
        table = pd.read_csv(table_path, encoding="utf-8")
        for i, row in gt_df[gt_df["tableName"] == table_name].iterrows():
            gt_df.at[i, "mention"] = table.iloc[row["row"], row["col"]]

    # Remove rows with NaN mentions
    print("Removing rows with NaN mentions")
    gt_df.dropna(subset=["mention"], inplace=True)

    total_number_of_mentions = len(gt_df)
    print("Total number of mentions:", total_number_of_mentions)

    if args.load_dbpedia_types_dict_from != "":
        print("Loading the dbpedia types dictionary from", args.load_dbpedia_types_dict_from)
        with open(args.load_dbpedia_types_dict_from, "rb") as f:
            dbpedia_types = pickle.load(f)
    else:
        print("Mapping Wikipedia titles to Wikidata IDs")
        # you can create the index-enwiki dump use this library https://github.com/jcklie/wikimapper
        wikipedia_wikidata_mapping = (
            spark.read.format("jdbc")
            .options(
                url="jdbc:sqlite:~/turl-data/index_enwiki-20190420.db",
                driver="org.sqlite.JDBC",
                dbtable="mapping",
            )
            .load()
        )
        wikipedia_wikidata_mapping.show()

        print("Creating the dbpedia types dictionary")
        dbpedia_types = dict(
            spark.createDataFrame(
                sc.textFile(args.dbpedia_types_path)
                .map(lambda x: x.split())
                .map(
                    lambda x: Row(
                        wikipedia_title=unquote(x[0][1:-1]).replace("http://dbpedia.org/resource/", ""),
                        type=x[2][1:-1].split("/")[-1],
                    )
                )
            )
            .join(wikipedia_wikidata_mapping, "wikipedia_title", "inner")
            .rdd.map(lambda x: (x["wikidata_id"], [x["type"]]))
            .reduceByKey(add)
            .collect()
        )
        if args.save_dbpedia_types_dict_to != "":
            print("Saving the dbpedia types dictionary to", args.save_dbpedia_types_dict_to)
            with open(args.save_dbpedia_types_dict_to, "wb") as f:
                pickle.dump(dbpedia_types, f)

    print("Getting types for every mention")
    gt_df["types"] = gt_df["id"].parallel_apply(lambda x: dbpedia_types.get(x, []))

    # # Save gt dataset pre-processed (i.e. with types every mention already computed)
    # if args.save_mentions_with_types_to != "":
    #     print("Saving the mentions with types to", args.save_mentions_with_types_to)
    #     gt_df.to_csv(args.save_mentions_with_types_to, index=False)

    # Get description for every QID
    print("Getting description for every unique QID")
    qids = pd.DataFrame(gt_df["id"].unique(), columns=["id"])
    qids["description"] = qids.parallel_apply(lambda row: wikidata_description_from_qids(row["id"]), axis=1)
    gt_df = gt_df.merge(qids, on="id", how="left")

    # # Save gt dataset pre-processed (i.e. with types and description for every mention already computed)
    # if args.save_mentions_with_types_desc_to != "":
    #     print("Saving the mentions with types and description to", args.save_mentions_with_types_desc_to)
    #     gt_df.to_csv(args.save_mentions_with_types_desc_to, index=False)

    # Get candidates for every mention
    unique_mentions = gt_df.drop_duplicates(subset=["mention"])
    if lookup == "wikidata":
        unique_mentions.loc[:, "candidates"] = unique_mentions.parallel_apply(
            lambda row: wikidata_lookup(row["mention"], dbpedia_types=dbpedia_types)[1], axis=1
        )
    elif lookup == "lamapi":
        unique_mentions.loc[:, "candidates"] = unique_mentions.parallel_apply(
            lambda row: lamapi_lookup(row["mention"], dbpedia_types=dbpedia_types, fuzzy=True)[1], axis=1
        )
    else:
        raise ValueError("Invalid lookup: {}".format(lookup))
    unique_mentions_with_candidates = {}
    for i, row in tqdm(unique_mentions.iterrows(), total=unique_mentions.shape[0]):
        unique_mentions_with_candidates[row["mention"]] = row["candidates"]
    gt_df["candidates"] = ""
    for i, row in tqdm(gt_df.iterrows(), total=gt_df.shape[0]):
        cand = unique_mentions_with_candidates[row["mention"]]
        gt_df.at[i, "candidates"] = str(cand)
    gt_df["candidates"] = gt_df["candidates"].parallel_apply(lambda x: ast.literal_eval(x))

    # Save gt dataset pre-processed (i.e. with types and candidates for every mention already computed)
    print(
        "Saving the mentions with types and candidates to",
        os.path.join(
            os.path.dirname(gt_path),
            "cea_gt_with_{}_candidates.csv".format(lookup),
        ),
    )
    gt_df.to_csv(os.path.join(os.path.dirname(gt_path), "cea_gt_with_{}_candidates.csv".format(lookup)), index=False)

    # # Read the gt file with candidates (if needed): this is the full gt dataset, without anything removed
    # gt_df = pd.read_csv(
    #     os.path.join(os.path.dirname(gt_path), "cea_gt_with_wikidata_candidates.csv")
    # )
    # gt_df["description"] = gt_df["description"].fillna("").astype(str)
    # gt_df["candidates"] = gt_df["candidates"].parallel_apply(
    #     lambda x: ast.literal_eval(x)
    # )
    # gt_df["types"] = gt_df["types"].parallel_apply(
    #     lambda x: ast.literal_eval(x)
    # )

    # Create dataset for TURL evaluation
    # We can have two cases:
    #
    # 1. We keep only those mentions that are contained in the candidate list generated by the wikidata lookup:
    # in this case we evaluate the overall system
    # 2. If the mention is not present in the candidate list, we add it ourself:
    # in this case we evaluate only the disambiguation model (TURL in this case)

    if not args.insert_target_mention_in_candidates:
        # Remove rows that do not contain any candidates (nothing to link to)
        gt_df = gt_df[gt_df["candidates"].apply(len).gt(0)]

        # Get only the rows where the list of candidates contains the mention (we want to test the disambiguation algorithm)
        gt_df = gt_df[gt_df.apply(lambda x: x["id"] in list(map(itemgetter(0), x["candidates"])), axis=1)]

    # Read headers from all the tables
    print("Reading headers from all the tables")
    table_names = gt_df["tableName"].unique().tolist()
    raw_tables_names = os.listdir(tables_folder)
    headers = {}
    for table_name in tqdm(raw_tables_names):
        if ".csv" not in table_name:
            continue
        table_path = os.path.join(tables_folder, table_name)
        headers[os.path.splitext(table_name)[0]] = (
            pd.read_csv(os.path.join(tables_folder, table_name), nrows=0, encoding="utf-8").columns.str.lower().tolist()
        )

    # Prepare data for TURL evaluation
    tables = []
    total_mention_per_table = 50
    table
    for table_name in tqdm(table_names):
        table_sample = gt_df[gt_df["tableName"] == table_name].sort_values(["row", "col"], ascending=[True, True])

        # Table-meta information
        page_title = ""
        section_title = ""
        caption = ""
        table_headers = list(map(str, headers[table_name]))

        # Table mentions to be linked
        all_mentions = table_sample.apply(lambda x: [[int(x["row"]), int(x["col"])], str(x["mention"])], axis=1)
        if len(all_mentions) == 0:
            continue
        else:
            all_mentions = all_mentions.tolist()

        # Loop over `all_mentions` in chunks of `total_mention_per_table`
        tmpt = total_mention_per_table if total_mention_per_table > 0 else len(all_mentions)
        for i in range(0, len(all_mentions), tmpt):
            mentions = all_mentions[i : i + tmpt]

            # Create candidates for each mention
            labels = []
            all_candidates = []
            entities_index = []
            for row_idx, row in table_sample[i : i + tmpt].iterrows():
                candidates = row["candidates"]
                try:
                    label_index = list(map(itemgetter(0), candidates)).index(row["id"])
                except ValueError:
                    if args.insert_target_mention_in_candidates:
                        label_index = 0
                        candidates = [[row["id"], row["mention"], row["description"], row["types"]]] + candidates
                    else:
                        continue
                label_index += len(all_candidates)
                candidates_without_id = [x[1:] for x in candidates]
                candidates_without_id_str = []
                for candidate in candidates_without_id:
                    mention = str(candidate[0])
                    description = str(candidate[1])
                    types = candidate[2]
                    candidates_without_id_str.append([mention, description, types])
                labels.append(int(label_index))
                all_candidates.extend(candidates_without_id_str)
                entities_index.append(list(range(len(all_candidates) - len(candidates), len(all_candidates))))
            if len(all_candidates) != 0:
                tables.append(
                    [
                        str(table_name),
                        page_title,
                        section_title,
                        caption,
                        table_headers,
                        mentions,
                        all_candidates,
                        labels,
                        entities_index,
                    ]
                )

    # Dump the dataset for TURL evaluation
    print(
        "Saving tables to",
        "~/turl-data/{}{}{}.table_entity_linking.json".format(
            table_type, "_all" if args.insert_target_mention_in_candidates else "", "_" + lookup
        ),
    )
    with open(
        "~/turl-data/{}{}{}.table_entity_linking.json".format(
            table_type, "_all" if args.insert_target_mention_in_candidates else "", "_" + lookup
        ),
        "w",
    ) as f:
        json.dump(tables, f)

    # Pre-process dataset with TURL ELDataset
    print("Creating ELDataset")
    data_dir = "~/turl-data"
    type_vocab = load_dbpedia_type_vocab(data_dir)
    train_dataset = ELDataset(
        data_dir,
        type_vocab,
        max_input_tok=500,
        src=(table_type + "_all" if args.insert_target_mention_in_candidates else table_type) + "_" + lookup,
        max_length=[50, 10, 10, 100],
        force_new=True,
        tokenizer=None,
    )
