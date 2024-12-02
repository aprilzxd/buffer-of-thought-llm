import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--result_path',type=str)
parser.add_argument('--run_test',type=str,default=False,help='Whether this is a test run that doesn\'t update the meta buffer')

if __name__ == "__main__":
    args = parser.parse_args()
    result_path = args.result_path
    run_test = args.run_test

    data_dir = 'gsm8k/train.jsonl' if not run_test else 'gsm8k/test.jsonl'

    correct = 0
    total = 0
    with open('tmp.jsonl', 'w') as output_file:
        for line_y, line_pred in zip(open(data_dir), open(result_path)):
            result = json.loads(line_pred)['result']
            true_y = json.loads(line_y)['ans']
            pred = result.split('\n')[-1]
            eval_result = '0'
            if str(true_y) in pred or str(int(true_y)) in pred:
                eval_result = '1'
                correct += 1
            total += 1
            output_file.write(f"{eval_result} | {int(true_y)} | {pred}\n")


        print(f'Total number:{total},Correct number:{correct},Accuracy:{correct/total}')
        output_file.write(f'Total number:{total},Correct number:{correct},Accuracy:{correct/total}')

        
    # benchmark_path_dict = {
    #     'gameof24':'benchmarks/gameof24.jsonl',
    #     'checkmate':'benchmarks/CheckmateInOne.jsonl',
    #     'wordsorting':'benchmarks/word_sorting.jsonl'
    # }
    # test_path_dict = {
    #     'gameof24':'test_results/BoT_gameof24.jsonl',
    #     'checkmate':'test_results/BoT_checkmate.jsonl',
    #     'wordsorting':'test_results/BoT_wordsorting.jsonl'
    # }
    # benchmark_path = benchmark_path_dict[task]
    # correct = 0
    # truth = []
    # test = []
    # for line in (open(benchmark_path)):
    #     answer = json.loads(line)['target']
    #     truth.append(answer)
    # for line in (open(test_path)):
    #     result = json.loads(line)['result']
    #     result = result.split('\n')[0]
    #     if task == 'gameof24':
    #         result = result.split('=')[0]
    #         test.append(result)
    #         try:
    #             if eval(result) == 24:
    #                 correct += 1
    #         except:
    #             continue
    #     else:
    #         test.append(result)
    # if correct == 0:
    #     for i in range(len(test)):
    #         if truth[i] == test[i]:
    #             correct += 1
    