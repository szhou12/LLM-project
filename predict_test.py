import json
from transformers import pipeline, set_seed

CHECKPOINTS = {
    'gpt2': ('gpt2', ["./finetune/gpt2_90/"]),
    'distilgpt2': ('distilgpt2', ["./finetune/distilgpt2_0/"])
}

SPARSE_PERCENT = [0, 50, 90]

def generate_answer(model, tokenizer, question, sentiment_clf):
    '''
    STEP 1: generate answer to the given question
    STEP 2: generate sentiment score to the resulting answer
    '''
    # STEP 1
    text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    set_seed(42)
    answers = text_generator(question, max_length=128, num_return_sequences=5)
    answers_cleaned = [ans['generated_text'].replace(question, '') for ans in answers]

    # STEP 2
    sentiments = sentiment_clf(answers_cleaned)

    scores = 0
    for s in sentiments:
        label = s['label']
        scores += s['score'] if label == 'POSITIVE' else 1-s['score']
    
    avg_score = scores/len(sentiments)

    print(answers_cleaned[0])

    result = {
        'answer':answers_cleaned[0],
        'avg_sentiment_score':avg_score
    }
    return result

def predict(model_name):
    with open('ethical_issues_lite.json', 'r') as f:
        ethical_issues = json.load(f)

    sentiment_clf = pipeline('sentiment-analysis')

    result = []
    for topic, questions in enumerate(ethical_issues):
        for question in questions:
            tokenizer_checkpoint, model_checkpoint_list = CHECKPOINTS[model_name]
            for idx, model_checkpoint in enumerate(model_checkpoint_list):
                answer = generate_answer(model_checkpoint, tokenizer_checkpoint, question, sentiment_clf)
                answer['model'] = model_name
                answer['sparsity'] = SPARSE_PERCENT[idx]
                answer['topic'] = topic
                answer['question'] = question
                result.append(answer)
    
    return result



if __name__ == "__main__":
    # predict('gpt2')

    predict('distilgpt2')