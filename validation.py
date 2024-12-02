import json

def check_answer(true_y, result):
    pred = result.split('\n')[-1]
    eval_result = False
    if str(true_y) in pred or str(int(true_y)) in pred:
        eval_result = True
    return eval_result

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
