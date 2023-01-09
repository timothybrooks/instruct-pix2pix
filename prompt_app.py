from __future__ import annotations

from argparse import ArgumentParser

import datasets
import gradio as gr
import numpy as np
import openai

from dataset_creation.generate_txt_dataset import generate


def main(openai_model: str):
    dataset = datasets.load_dataset("ChristophSchuhmann/improved_aesthetics_6.5plus", split="train")
    captions = dataset[np.random.permutation(len(dataset))]["TEXT"]
    index = 0

    def click_random():
        nonlocal index
        output = captions[index]
        index = (index + 1) % len(captions)
        return output

    def click_generate(input: str):
        if input == "":
            raise gr.Error("Input caption is missing!")
        edit_output = generate(openai_model, input)
        if edit_output is None:
            return "Failed :(", "Failed :("
        return edit_output

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        txt_input = gr.Textbox(lines=3, label="Input Caption", interactive=True, placeholder="Type image caption here...")  # fmt: skip
        txt_edit = gr.Textbox(lines=1, label="GPT-3 Instruction", interactive=False)
        txt_output = gr.Textbox(lines=3, label="GPT3 Edited Caption", interactive=False)

        with gr.Row():
            clear_btn = gr.Button("Clear")
            random_btn = gr.Button("Random Input")
            generate_btn = gr.Button("Generate Instruction + Edited Caption")

            clear_btn.click(fn=lambda: ("", "", ""), inputs=[], outputs=[txt_input, txt_edit, txt_output])
            random_btn.click(fn=click_random, inputs=[], outputs=[txt_input])
            generate_btn.click(fn=click_generate, inputs=[txt_input], outputs=[txt_edit, txt_output])

    demo.launch(share=True)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--openai-api-key", required=True, type=str)
    parser.add_argument("--openai-model", required=True, type=str)
    args = parser.parse_args()
    openai.api_key = args.openai_api_key
    main(args.openai_model)
