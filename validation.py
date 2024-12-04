import json

def check_answer(true_y, result):
    pred = result.split('\n')[-1]
    eval_result = False
    if str(true_y) in pred or str(int(true_y)) in pred or f"{int(true_y):,}" in pred:
        eval_result = True
    return eval_result

def validator(data_path, result_filepath, output_filepath):
    data_dict = {}
    with open(data_path, 'r') as data_file:
        for line in data_file:
            line_y = json.loads(line)
            question = line_y['question']
            data_dict[question] = line_y

    result_dict = {}
    with open(result_filepath, 'r') as result_file:
        for line in result_file:
            line_pred = json.loads(line)
            input_question = line_pred['input']
            result_dict[input_question] = line_pred
        
    correct = 0
    total = 0
    with open(output_filepath, 'w') as output_file:
        for question, line_y in data_dict.items():
            if question in result_dict:
                line_pred = result_dict[question]
                result = line_pred['result']
                true_y = line_y['ans']
                pred = result.split('\n')[-1]
                eval_result = '0'
                if str(true_y) in pred or str(int(true_y)) in pred or f"{int(true_y):,}" in pred:
                    eval_result = '1'
                    correct += 1
                # else:
                    # output_file.write(f"{eval_result} | {int(true_y)} | {pred}\n")
                total += 1
                output_file.write(f"{eval_result} | {int(true_y)} | {pred}\n")

        print(f'#Question: {total}, #Correct: {correct}, Acc: {correct/total},')
        output_file.write(f'#Question: {total}, #Correct: {correct}, Acc: {correct/total},')       

if __name__ == '__main__':
    result_filepath = 'GSM8K_qwen2_5-7b_train_distill_True.jsonl'
    data_path = 'gsm8k/train.jsonl'
    output_filepath = 'gsm8k_qwen2_5-7b_train_distill_True_eval.txt'
    validator(data_path, result_filepath, output_filepath)
