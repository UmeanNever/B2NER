import json
import os
import re
import copy
import argparse
import shutil
import pandas as pd

from evaluation.evaluator import *

CLIP_CHAR_LENGTH = 2048

def main():
    os.environ['EXPORT_IMG'] = '0'      # 是否导出混淆矩阵图片
    parser = argparse.ArgumentParser(description='Calculate F1 scores.')
    parser.add_argument('--root', type=str, 
                        default='/mnt/data/user/yang_yuming/proj2023/UOpenFS/B2NER/sample_predictions/B2NER-7B',
                        help='Root directory to calculate F1 scores for.')
    args = parser.parse_args()
    root = args.root
    iterate_over_dirs(root)
    # iterate_over_dirs(root, should_clip=True)


def iterate_over_dirs(root, should_clip=False):
    # iterate over checkpoint-* or eval-* dirs in root
    print(f"Walking through {root}")
    agg_df = pd.DataFrame()
    for dir in os.listdir(root):
        if not re.match(r'checkpoint-\d+', dir) and not re.match(r'eval-\d+', dir):
            continue
        print(f"Predict for {dir}")
        root_ckpt = os.path.join(root, dir)
        if not os.path.exists(os.path.join(root_ckpt, 'predict_eval_predictions.jsonl')):
            continue
        ckpt_result_rows = calculate_f1(root_ckpt, should_clip)
        # ckpt_result_rows.append(("hal_rate", check_answer_keys(root_ckpt)))
        # move img folder in current directory to root_ckpt
        if os.path.exists('img'):
            # remove if exists
            if os.path.exists(os.path.join(root_ckpt, 'img')):
                shutil.rmtree(os.path.join(root_ckpt, 'img'))
            shutil.move('img', root_ckpt)
        # append epoch_result_rows to agg_df
        if len(agg_df) == 0:
            agg_df = pd.DataFrame(ckpt_result_rows, columns=['epoch', dir.split('-')[1]])
        else:
            agg_df[dir.split('-')[1]] = [row[1] for row in ckpt_result_rows]
        
    # write df
    if len(agg_df) > 0:
        print(f"Aggregated results for {root}:")
        # set epoch column as index, order columns by name
        agg_df = agg_df.set_index('epoch')
        agg_df = agg_df[sorted(agg_df.columns)]
        print(agg_df.round(4))
        csv_dir = root.replace('output', 'output/csvs')
        os.makedirs(csv_dir, exist_ok=True)
        csv_name = "agg.csv" if not should_clip else "agg-clip.csv"
        agg_df.round(4).to_csv(os.path.join(csv_dir, csv_name), index=True)
        agg_df.round(4).to_csv(os.path.join(root, csv_name), index=True)
    
        
    # root = "../output/gpt4/unified+all"
    if len(agg_df) == 0:
        calculate_f1(root)
    # check_answer_keys(root)
    # if os.path.exists('img'):
    #     # remove if exists
    #     if os.path.exists(os.path.join(root, 'img')):
    #         shutil.rmtree(os.path.join(root, 'img'))
    #     shutil.move('img', root)
    

# distinguish unseen and mix datasets
unseen_ent_datasets = ["CAIL2021", "math-high", "CCKS2021address"]
mix_ent_datasets = ["CLUENER", "CMeEEV2", "weibo", "zh-ontonotes"]
except_datasets = ["math-junior", "CAIL2022"]

def calculate_f1(output_dir, should_clip=False):
    EvaluatorDict = {
        'RE':EvaluatorRE,
        'EE':EvaluatorEvent,
        'NER':EvaluatorNER,
        'EET':EvaluatorEET,
        'EEA':EvaluatorEEA,
        'newEEA':EvaluatorEEA,
        'newEET':EvaluatorEET,
        'Sentiment':EvaluatorSentiment
    }
    task_dict = dict()      # str -> dict
    task_path = os.path.join(output_dir, 'predict_eval_predictions.jsonl')
    report_dir_root = os.path.join(output_dir, 'report' if not should_clip else 'report_clip')
    with open(task_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentence = data["Instance"]["sentence"]

            task_name = data['Task']
            dataset_name = data['Dataset']
            prediction = data['Prediction']
            language = "" if 'Language' not in data else data['Language']
            
            if should_clip:
                if data["Instance"]["id"] == "-1":
                    continue
                max_char_length = CLIP_CHAR_LENGTH / 4 if language=='zh' else CLIP_CHAR_LENGTH
                if len(sentence) > max_char_length:
                    continue
            
            if ('gpt' in output_dir or 'eval+full' in output_dir) and 'NER' in task_name:
                prediction = gpt_format_fix(prediction)
            task_name_w_lang = task_name + "_" + language
            if task_name_w_lang not in task_dict:
                task_dict[task_name_w_lang] = dict()
            if dataset_name not in task_dict[task_name_w_lang]:
                task_dict[task_name_w_lang][dataset_name] = EvaluatorDict[task_name]()
            task_dict[task_name_w_lang][dataset_name].add(data, prediction)

    # export report
    if not os.path.exists(report_dir_root):
        os.mkdir(report_dir_root)

    # export tsv
    all_result_rows = []
    all_scores = []
    for task_name, eval_dict in task_dict.items():
        print('\n'+'-'*16+task_name+'-'*16+'\n')
        rows = []
        scores = []
        scores_unseen = []
        scores_mix = []
        report_dir = os.path.join(report_dir_root, task_name)
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        for dataset_name, evaluator in eval_dict.items():
            evaluator.dump_audit_report(os.path.join(report_dir, dataset_name+'.json'))
            dataset_metric = evaluator.get_metric()
            rows.append((task_name + "_" + dataset_name, dataset_metric))
            if dataset_name not in except_datasets:
                scores.append(dataset_metric)
                all_scores.append(dataset_metric)
            if dataset_name in unseen_ent_datasets:
                scores_unseen.append(dataset_metric)
            if dataset_name in mix_ent_datasets:
                scores_mix.append(dataset_metric)
        rows = sorted(rows, key=lambda x: x[0].lower())
        # if len(scores) == 0:
            # continue
        f1_unseen = sum(scores_unseen)/len(scores_unseen) if len(scores_unseen) > 0 else 0
        f1_mix = sum(scores_mix)/len(scores_mix) if len(scores_mix) > 0 else 0
        f1_avg = sum(scores)/len(scores) if len(scores) > 0 else 0
        # if task_name == 'NER_zh':
        #     rows.append((task_name + "_" + 'Average_unseen', f1_unseen))
        #     rows.append((task_name + "_" + 'Average_mix', f1_mix))
        rows.append((task_name + "_" + 'Average', f1_avg))
        with open(os.path.join(report_dir_root, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
            for row in rows:
                f.write(f'{row[0]}\t{row[1]}\n')
                print('%48s\t%g'%row)
        all_result_rows.extend(rows)
    f1_all_avg = sum(all_scores)/len(all_scores) if len(all_scores) > 0 else 0
    all_result_rows.append(('Average', f1_all_avg))
    return all_result_rows


def gpt_format_fix(prediction_text):
    """
    Transfrom "A: B, C; D: E, F" to "A: B; A: C; D: E; D: F"
    """
    predictions = prediction_text.split(";")
    predictions_new = []
    for prediction in predictions:
        if ":" not in prediction or ("," not in prediction and "、" not in prediction):
            predictions_new.append(prediction)
            continue
        key = prediction.split(":")[0]
        # split by : then split by , or 、
        values = re.split(r"[,、]", prediction.split(":")[1])
        for value in values:
            predictions_new.append(key+":"+value)
    return ";".join(predictions_new)


def check_answer_keys(output_dir):
    task_dict = dict()      # str -> dict
    task_path = os.path.join(output_dir, 'predict_eval_predictions.jsonl')
    report_dir_root = os.path.join(output_dir, 'report_key_check')
    if not os.path.exists(report_dir_root):
        os.mkdir(report_dir_root)
    with open(task_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            task_name = data['Task']
            dataset_name = data['Dataset']
            prediction = data['Prediction']

            # Extract keys from prediction
            prediction_keys = [piece.split(":")[0].strip() for piece in prediction.split(";") if ":" in piece]
            prediction_keys = [EvaluatorBase._format(x) for x in prediction_keys]

            # Extract keys from instruction!
            ground_truth_keys = EvaluatorBase._resolve_option(data['Instance']['instruction'])

            if task_name not in task_dict:
                task_dict[task_name] = dict()
            if dataset_name not in task_dict[task_name]:
                task_dict[task_name][dataset_name] = {'total': 1, 'mismatch': 0}
            for key in prediction_keys:
                task_dict[task_name][dataset_name]['total'] += 1
                if key not in ground_truth_keys:
                    task_dict[task_name][dataset_name]['mismatch'] += 1

    # export tsv
    for task_name, dataset_dict in task_dict.items():
        print('\n'+'-'*16+task_name+'-'*16+'\n')
        rows = []
        report_dir = os.path.join(report_dir_root, task_name)
        if not os.path.exists(report_dir):
            os.mkdir(report_dir)
        total_mismatch = 0
        total_total = 1
        for dataset_name, stats in dataset_dict.items():
            mismatch_percentage = stats['mismatch'] / stats['total']
            total_mismatch += stats['mismatch']
            total_total += stats['total']
            rows.append((dataset_name, mismatch_percentage))
        rows = sorted(rows, key=lambda x: x[0].lower())
        rows.append(('TotalHallucination', total_mismatch / total_total))
        with open(os.path.join(report_dir_root, 'report_%s.tsv'%task_name), 'w', encoding='utf-8') as f:
            for row in rows:
                if row[1] > 0.1 or row[0]=='TotalHallucination':
                    f.write(f'{row[0]}\t{row[1]}\n')
                    print('%48s\t%g'%row)
    return rows[-1][1]

if __name__ == '__main__':
    os.environ['RANDOM_RECORD'] = '1'   # 是否开启随机记录
    main()
