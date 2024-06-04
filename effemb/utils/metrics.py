
import collections

TaskResult = collections.namedtuple("task_result", ["name", "category", "score"])


def extract_main_metrics(task_description, result):
    metrics = {}
    original_metrics = {}

    # pair classification is strange ['cos_sim']['ap']
    # STS, summarization ['cos_sim']['spearman']
    leaderboard_metric = None
    # import pdb; pdb.set_trace()
    for split in task_description["eval_splits"]:
        split_res = result[split]
        for lang in task_description["eval_langs"]:
            if len(task_description["eval_langs"]) > 1:
                lang_res = split_res[lang]
            else:
                lang_res = split_res

            if task_description["type"] == "PairClassification":
                task_result = lang_res["cos_sim"]["ap"]
                metrics[f"{split}_{lang}_cos_sim_ap"] = task_result
            elif (
                task_description["type"] == "STS"
                or task_description["type"] == "Summarization"
            ):
                task_result = lang_res["cos_sim"]["spearman"]
                metrics[f"{split}_{lang}_cos_sim_spearman"] = task_result
            else:
                task_result = lang_res[task_description["main_score"]]
                metrics[
                    f"{split}_{lang}_{task_description['main_score']}"
                ] = task_result

            if split == "test" and (lang == "en" or lang == "en-en"):
                leaderboard_metric = TaskResult(
                    task_description["name"], task_description["type"], task_result
                )

    if leaderboard_metric is None:
        print("==============================")
        print("No leaderboard metric")
        print(task_description)
        print("==============================")
        print(result)
        print("==============================")
    return metrics, leaderboard_metric
