import json

def validator(data_path, result_filepath, output_filepath):
    correct = 0
    total = 0
    with open(output_filepath, 'w') as output_file:
        for line_y, line_pred in zip(open(data_path), open(result_filepath)):
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
    