import streamlit as st
import argparse
import time
from transformers import pipeline, set_seed
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

@st.cache(allow_output_mutation=True)
def get_model():
    text_generator = pipeline('text-generation', model='gpt2')
    return text_generator

model = get_model()

def generate_answer(model, topic, context):
    question = f'{topic}: {context}'
    set_seed(42)
    answers = model(question, max_length=128, num_return_sequences=1)
    answers_cleaned = [ans['generated_text'].replace(question, '') for ans in answers]
    first_answer = answers_cleaned[0].strip()
    return first_answer

def run():
    st.markdown(
        """
        ## GPT Model: Q & A App
        """
    )

    st.sidebar.subheader("Parameters")

    generate_max_len = st.sidebar.number_input("generate_max_len", min_value=0, max_value=512, value=32, step=1)
    top_k = st.sidebar.slider("top_k", min_value=0, max_value=10, value=3, step=1)
    top_p = st.sidebar.number_input("top_p", min_value=0.0, max_value=1.0, value=.95, step=.01)
    temperature = st.sidebar.number_input("temperature", min_value=0.0, max_value=100.0, value=1.0, step=.1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_max_len', default=generate_max_len, type=int, help='max length to generate')
    parser.add_argument('--top_k', default=top_k, type=float, help='')
    parser.add_argument('--top_p', default=top_p, type=float, help='')
    parser.add_argument('--temperature', default=temperature, type=float, help='')
    parser.add_argument('--max_len', default=128, type=int, help='')
    args = parser.parse_args()

    title = st.text_area('Enter Topic:', max_chars=512)
    context = st.text_area('Enter Question:', max_chars=512)

    if st.button('Submit'):
        start_message = st.empty()
        start_message.write('Processing...')
        start_time = time.time()
        # Call pre-trained model
        result = generate_answer(model, title, context)
        end_time = time.time()
        start_message.write(f'Finished in {end_time-start_time}s')
        st.text_area('Result:', value=result, key=None)
    else:
        st.stop()

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        pass