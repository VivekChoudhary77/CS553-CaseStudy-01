# Original product reference: https://huggingface.co/spaces/fffiloni/Image-to-Story/blob/main/app.py

import spaces
import gradio as gr
import re
import os 
hf_token = os.environ.get('HF_TOKEN')

from gradio_client import Client, handle_file

clipi_client = Client("fffiloni/CLIP-Interrogator-2")

from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "Qwen/Qwen2.5-3B-Instruct"  # Qwen/Qwen3-8B was too large

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(model_path, token=hf_token).half().cuda()

@spaces.GPU
def quen_gen_safety_advice(prompt):
    """Generate a list of safety instructions using the qwen model based on a prompt.
    
    Args:
        prompt: A string prompt containing an image description and safety_advice generation instructions.
        
    Returns:
        A generated list of safety instructions string with special formatting and tokens removed.
    """

    instruction = """[INST] <<SYS>>\nYou are a professional safety analyst. You will be given an image caption and must provide a bulleted list of core safety instructions to consider. 
            For that given you'll be asked to generate a list of safety instructions that you think could fit very well with the image provided.
            Always answer with a list of safety instructions, while being safe as possible.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

    
    prompt = instruction.format(prompt)
    
    generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096)
    output_text = tokenizer.decode(generate_ids[0], skip_special_tokens=True)
    #print(generate_ids)
    #print(output_text)
    pattern = r'\[INST\].*?\[/INST\]'
    cleaned_text = re.sub(pattern, '', output_text, flags=re.DOTALL)
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
    """Generate a bulleted list of safety advice based on an image using CLIP Interrogator and an LLM.
    
    Args:
        image_input: A file path to the input image to analyze.
        running_platform: A string indicating the target running platform for the LLM to run on (local or remote).
    
    Returns:
        A formatted, list of safety instructions based on the provided image.

    """
    gr.Info('Calling CLIP Interrogator ...')

    clipi_result = clipi_client.predict(
		input_image=handle_file(image_input),
		interrogation_mode="best",
		best_mode_max_flavors=4,
		api_name="/clipi2"
    )
    print(clipi_result)

    if running_platform == "Local (Qwen2.5-3B-Instruct)":
        qwen = f"""
        I'll give you a simple image caption, please provide a bulleted list of safety instructions that would fit well with the image.
        Here's the image description: 
        '{clipi_result}'
        
        """
        gr.Info('Calling Qwen3 ...')
        result = quen_gen_safety_advice(qwen)

        print(f"Qwen3 result: {result}")

        result = get_text_after_colon(result)
    else:
        # Put remote model inference code here
        result = "No remote model configured"
        print(f"No remote model configured")

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
                running_platform = gr.Radio(label="LLM Model", choices=["Local (Qwen2.5-3B-Instruct)", "Remote (model name here)"], value="Children")
                submit_btn = gr.Button('Give me safety advice')
            with gr.Column():
                #caption = gr.Textbox(label="Generated Caption")
                safety_advice = gr.Textbox(label="Generated Safety Advice", elem_id="safety_advice")
        
    submit_btn.click(fn=infer, inputs=[image_in, running_platform], outputs=[safety_advice])

demo.queue(max_size=12).launch(ssr_mode=False, mcp_server=True)
