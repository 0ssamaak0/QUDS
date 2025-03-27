from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from tqdm import tqdm
from utils import *
import cohere
import chromadb
from thefuzz import fuzz


app = Flask(__name__, template_folder="templates")

# Load environment variables and create clients
from dotenv import load_dotenv

load_dotenv()
cohere_api = os.environ["cohere_api"]
co = cohere.ClientV2(api_key=cohere_api)
client = chromadb.PersistentClient("db")
collection_saddi = client.get_or_create_collection(
    "saddi", metadata={"hnsw:space": "cosine"}
)
collection_muyassar = client.get_or_create_collection(
    "muyassar", metadata={"hnsw:space": "cosine"}
)

# Load and process surah data
with open("surah.json") as f:
    data = json.load(f)

surahs = []
for num in data:
    surahs.append(
        {
            "name": data[num]["name"],
            "nAyah": data[num]["nAyah"],
            "start": data[num]["start"],
            "end": data[num]["end"],
        }
    )
df = pd.DataFrame(surahs)


def parse_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    sections = content.split("# ")
    text_dict = {}

    for section in sections:
        if section.strip():
            lines = section.split("\n", 1)
            number = lines[0].strip()
            text = lines[1].strip() if len(lines) > 1 else ""
            text_dict[number] = text

    return text_dict


parsed_ayas = parse_markdown("uthmani-qurancom.md")
parsed_ayas_simple = parse_markdown("uthmani-simple-qurancom.md")
parsed_ayas_list_simple = list(parsed_ayas_simple.values())


def get_ayah(number):
    return parsed_ayas.get(str(number), "Number not found")


# write a function that gets a number and using start and end it returns the surah name and the aya num
def get_surah_name(num):
    surah_name = df.loc[(df["start"] <= num) & (df["end"] >= num), "name"].values[0]
    aya_num = (
        num - df.loc[(df["start"] <= num) & (df["end"] >= num), "start"].values[0] + 1
    )
    return surah_name, aya_num


def multi_search(
    query, collections, embedding_n_results=25, rerank_n_results=5, rerank=False
):
    response = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=["float"],
    ).embeddings.float_[0]

    result_dicts = []
    for collection in collections:
        result = collection.query(response, n_results=embedding_n_results)
        ids = result["ids"][0]
        distances = result["distances"][0]
        documents = result["documents"][0]
        # create dict of ayahs and distances
        result_dict = [
            {"id": id_, "distance": distance, "document": document}
            for id_, distance, document in zip(ids, distances, documents)
        ]
        result_dicts.append(result_dict)

    merged_list = filter_results_by_distance(result_dicts)

    # Group items by distance
    grouped_by_distance = defaultdict(list)
    surahs = defaultdict(list)
    ayas = defaultdict(list)

    for item in merged_list:
        surah_name, aya_num = get_surah_name(int(item["id"]))
        grouped_by_distance[item["distance"]].append(get_ayah(item["id"]))
        surahs[item["distance"]].append(surah_name)
        ayas[item["distance"]].append(aya_num)

    # Concatenate documents with the same distance and add [surah: ayafirst - ayalast]
    result_ayas = []
    for distance, documents in grouped_by_distance.items():
        surah_aya_map = defaultdict(list)
        for surah_name, aya_num in zip(surahs[distance], ayas[distance]):
            surah_aya_map[surah_name].append(aya_num)

        # Combine all documents with the same distance
        combined_docs = " ".join(documents)

        formatted_ayas = " ".join(
            (
                f"{combined_docs} [{surah} : {min(aya_nums)} - {max(aya_nums)}]"
                if len(aya_nums) > 1
                else f"{combined_docs} [{surah} : {aya_nums[0]}]"
            )
            for surah, aya_nums in surah_aya_map.items()
        )
        result_ayas.append(formatted_ayas)

    result_documents = [i["document"] for i in merged_list]
    merged_dict_reranked, result_ayas_reranked = None, None
    if rerank:
        response = co.rerank(
            model="rerank-multilingual-v3.5",
            query=query,
            documents=result_documents,
            top_n=rerank_n_results,
        )
        indices = [i.index for i in response.results]
        ids = [merged_list[i]["id"] for i in indices]
        distances = [1 - i.relevance_score for i in response.results]

        merged_dict_reranked = [
            {"id": id_, "distance": distance, "document": merged_list[i]["document"]}
            for i, (id_, distance) in enumerate(zip(ids, distances))
        ]

        result_ayas_reranked = []
        for item in merged_dict_reranked:
            surah_name, aya_num = get_surah_name(int(item["id"]))
            result_ayas_reranked.append(
                f"{get_ayah(item['id'])} [{surah_name} : {aya_num}]"
            )

    return (
        result_ayas[:rerank_n_results],
        merged_list,
        result_ayas_reranked,
        merged_dict_reranked,
    )


def fuzzy_search(
    query,
    parsed_ayas_list,
    max_values=10,
    upper_bound=90,
    lower_bound=75,
    min_aya_length=10,
):
    """
    Finds the top matches for a query in a list of parsed Ayas using fuzzy matching.

    Parameters:
        query (str): The query string to match against the list.
        parsed_ayas_list (list): The list of parsed Ayas.
        max_values (int): The target number of results for scores between bounds.
        upper_bound (int): Return all matches above this score regardless of max_values.
        lower_bound (int): Minimum score to include in results.
        min_aya_length (int): Minimum length of Aya to consider for matching.

    Returns:
        candidates (list): The list of top matching Ayas.
        values (list): The list of corresponding fuzzy matching scores.
    """
    query = query.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    # drop tashkeel
    query = (
        query.replace("َ", "")
        .replace("ُ", "")
        .replace("ِ", "")
        .replace("ّ", "")
        .replace("ْ", "")
        .replace("ٌ", "")
        .replace("ٍ", "")
    )
    # Calculate partial ratios, applying a length condition
    results = [
        (
            fuzz.partial_ratio(query, parsed_ayas_list[i])
            if len(parsed_ayas_list[i]) > min_aya_length
            else 0
        )
        for i in range(len(parsed_ayas_list))
    ]

    # Get indices sorted by scores in descending order
    sorted_indices = sorted(range(len(results)), key=lambda i: results[i], reverse=True)

    # First, get all results above upper_bound
    high_scoring_indices = [i for i in sorted_indices if results[i] >= upper_bound]

    # Then handle scores between bounds
    mid_scoring_indices = [
        i for i in sorted_indices if lower_bound <= results[i] < upper_bound
    ]

    if mid_scoring_indices:
        # If we have less than max_values high-scoring results, we can add more mid-scoring ones
        remaining_slots = max_values - len(high_scoring_indices)

        if remaining_slots > 0:
            # Get the score at the max_values cutoff point
            cutoff_index = min(remaining_slots - 1, len(mid_scoring_indices) - 1)
            cutoff_score = results[mid_scoring_indices[cutoff_index]]

            # Include all results with scores equal to the cutoff score
            mid_scoring_indices = [
                i for i in mid_scoring_indices if results[i] >= cutoff_score
            ]

    # Combine high and mid scoring indices
    final_indices = high_scoring_indices + mid_scoring_indices
    if len(final_indices) == 0:
        if upper_bound - lower_bound < 20:
            return fuzzy_search(
                query,
                parsed_ayas_list,
                max_values=5,
                upper_bound=90,
                lower_bound=lower_bound - 5,
                min_aya_length=10,
            )
        else:
            return [], []
    surah_names = [get_surah_name(i) for i in final_indices]
    surah_names, aya_nums = zip(*surah_names)
    # Retrieve the candidates and their scores
    candidates = [get_ayah(i) for i in final_indices]
    candidates = [
        f"{candidates[i]} [{surah_names[i]}:{aya_nums[i]}]"
        for i in range(len(candidates))
    ]
    values = [results[i] for i in final_indices]

    return candidates, values


# Flask app routes
@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query")
    rerank = (
        request.args.get("rerank", "false").lower() == "true"
    )  # Convert string to boolean
    collections = [collection_saddi, collection_muyassar]  # List your collections here
    max_values = 5
    upper_bound = 90
    lower_bound = 80

    candidates, values = fuzzy_search(
        query,
        parsed_ayas_list_simple,
        max_values=max_values,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
    )
    if len(query) < 15 or (len(candidates) > 0 and values[0] > lower_bound - 5):
        print("Thefuzz")
        return jsonify(candidates)
    print("Deep")
    result_ayas, merged_list, result_ayas_reranked, merged_dict_reranked = multi_search(
        query, collections, rerank=rerank
    )
    if rerank:
        return jsonify(result_ayas_reranked)
    else:
        return jsonify(result_ayas)


@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(debug=True, host="0.0.0.0", port=port)
