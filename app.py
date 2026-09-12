# Original product reference: https://huggingface.co/spaces/fffiloni/Image-to-Story/blob/main/app.py

import spaces
import gradio as gr
import re
import os 
hf_token = os.environ.get('HF_TOKEN')

from gradio_client import Client, handle_file

clipi_client = Client("fffiloni/CLIP-Interrogator-2")

from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import InferenceClient

model_path = "Qwen/Qwen2.5-3B-Instruct"  # Qwen/Qwen3-8B was too large
remote_model_path = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, token=hf_token).half().cuda()

@spaces.GPU
def gen_safety_advice(prompt, platform):
    """Generate a 2-4 sentence segment of safety advice using the qwen model based on a prompt.
    
    Args:
        prompt: A string prompt containing an image description and safety_advice generation instructions.
        
    Returns:
        A generated 2-4 sentence segment of safety advice string with special formatting and tokens removed.
    """

    instruction = """[INST] <<SYS>>\nYou are a professional safety analyst. You will be given an image caption and must provide a numbered list of core safety advice to consider.
            In your response, please limit safety advice to 2-4 sentences of the most crucial safety advice to consider. Provide a concise title before the safety advice.
            Always answer with the top safety advice, while being safe as possible.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

    
    prompt = instruction.format(prompt)

    if platform == "Local (Qwen/Qwen2.5-3B-Instruct)":
        gr.Info('Calling Qwen2.5-3B-Instruct (local)...')
        generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096)
        output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    else:
        gr.Info('Calling OpenAI/gpt-oss-20b (remote)...')
        try:
            inf_client = InferenceClient(token=hf_token)
            output_text = inf_client.text_generation(model=remote_model_path, inputs=prompt, max_new_tokens=4096)
        except Exception as e:
            gr.Info(f"Error: {e}")
    #print(generate_ids)
    #print(output_text)
    pattern = r'\[INST\].*?\[/INST\]'
    cleaned_text = re.sub(pattern, '', output_text, flags=re.DOTALL)
    print(f"cleaned_test: {cleaned_text}")
    return cleaned_text

def get_text_after_colon(input_text):
    # Find the first occurrence of ":"
    colon_index = input_text.find(":")
    
    # Check if ":" exists in the input_text
    if colon_index != -1:
        # Extract the text after the colon
        result_text = input_text[colon_index + 1:].strip()
        return result_text
    else:
        # Return the original text if ":" is not found
        return input_text

def infer(image_input, running_platform):
    """Generate 2-4 sentence of the most important safety advice based on an image using CLIP Interrogator and an LLM.
    
    Args:
        image_input: A file path to the input image to analyze.
        running_platform: A string indicating the target running platform for the LLM to run on (local or remote).
    
    Returns:
        A formatted, 2-4 sentence segment of safety advice based on the provided image.

    """
    gr.Info('Calling CLIP Interrogator ...')

    clipi_result = clipi_client.predict(
		input_image=handle_file(image_input),
		interrogation_mode="best",
		best_mode_max_flavors=4,
		api_name="/clipi2"
    )
    print(clipi_result)

    capt_prompt = f"""
    I'll give you a simple image caption, please provide a 2-4 sentence segment of the most important safety advice that would fit well with the image.
    Here's the image description: 
    '{clipi_result}'
    
    """
    result = gen_safety_advice(capt_prompt, running_platform)

    result = get_text_after_colon(result)

    # Split the text into paragraphs based on actual line breaks
    paragraphs = result.split('\n')
    
    # Join the paragraphs back with an extra empty line between each paragraph
    formatted_text = '\n\n'.join(paragraphs)


    return formatted_text

css="""
#col-container {max-width: 910px; margin-left: auto; margin-right: auto;}
div#safety_advice textarea {
    font-size: 1.5em;
    line-height: 1.4em;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(
            """
            <h1 style="text-align: center">Image to Safety Advice</h1>
            <p style="text-align: center">Upload an image, get safety advice based on the image content!</p>
            """
        )
        with gr.Row():
            with gr.Column():
                image_in = gr.Image(label="Image Input", type="filepath", elem_id="image-in")
                running_platform = gr.Radio(label="LLM Model", choices=["Local (Qwen/Qwen2.5-3B-Instruct)", "Remote (OpenAI/gpt-oss-20b)"], value="Children")
                submit_btn = gr.Button('Give me safety advice')
            with gr.Column():
                #caption = gr.Textbox(label="Generated Caption")
                safety_advice = gr.Textbox(label="Generated Safety Advice", elem_id="safety_advice")
        
    submit_btn.click(fn=infer, inputs=[image_in, running_platform], outputs=[safety_advice])

demo.queue(max_size=12).launch(ssr_mode=False, mcp_server=True)