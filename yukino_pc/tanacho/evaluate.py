import json
import argparse


def MAP(sub, ans, k):
    eval_files = set(ans).intersection(set(sub))
    print('\nEvaluating {} samples...\n'.format(len(eval_files)))
    score = 0
    for eval_file in sorted(eval_files):
        s = ap(sub[eval_file], ans[eval_file], k)
        score += s
        print('Average Precision for {}: {}'.format(eval_file, s))

    return score / len(eval_files)


def ap(preds, gts, k):
    pred_count = 0
    dtc_count = 0
    num_positives = len(gts)
    score = 0
    for pred in preds:
        pred_count += 1
        if pred in gts:
            dtc_count += 1
            score += dtc_count/pred_count
            gts.remove(pred)
        if len(gts) == 0:
            break
    score /= min(num_positives, k)

    return score


def validate(sub, ans, k):
    message = 'ok'
    status = 0
    eval_files = set(ans).intersection(set(sub))
    if len(eval_files) == 0:
        message = 'No sample for evaluation.'
        status = 1
        return sub, ans, message, status
    for eval_file in eval_files:
        gt = ans[eval_file]
        pr = sub[eval_file]
        if not isinstance(gt, list):
            message = 'Invalid data type found in {} in the answer file. Should be list.'.format(eval_file)
            status = 1
            return sub, ans, message, status
        if not isinstance(pr, list):
            message = 'Invalid data type found in {} in the prediction file. Should be list.'.format(eval_file)
            status = 1
            return sub, ans, message, status
        if len(gt) != 1:
            message = 'The answer should be only one in {} in the answer file.'.format(eval_file)
            status = 1
            return sub, ans, message, status
        if len(pr) > k:
            message = 'The number of predictions exceeded in {}(maximum is {}).'.format(eval_file, k)
            status = 1
            return sub, ans, message, status
    return sub, ans, message, status


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground-truth-path', default = './data/ans.json')
    parser.add_argument('--predictions-path', default = './data/sub.json')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # parse the arguments
    args = parse_args()

    # load the files
    with open(args.ground_truth_path) as f:
        ans = json.load(f)
    with open(args.predictions_path) as f:
        sub = json.load(f)

    # validation
    k = 10 # The maximum number of predictions
    sub, ans, message, status = validate(sub, ans, k)

    # evaluation
    if status == 0:
        score = MAP(sub, ans, k)
        print('\nMAP@{}: {}'.format(k, score))
    else:
        print(message)
