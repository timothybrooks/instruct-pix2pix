import gradio as gr
import numpy as np
from PIL import Image

import torch
from diffusers import StableDiffusionInstructPix2PixPipeline


def image_processing_app(prompt, num_inference_steps, guidance_scale, image_guidance_scale, input_image):
    if input_image is not None:
        output_image = run_function(input_image, num_inference_steps, guidance_scale, image_guidance_scale)
        output_image = np.array(output_image).astype('uint8')

        return input_image, output_image
    else:
        return None

def main():
    model_id = "timbrooks/instruct-pix2pix"
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    def generate(input_image, instruction, steps, randomize_seed, seed, randomize_cfg, text_cfg_scale, image_cfg_scale, **kwargs):
        run = True
        img = input_image.resize((512, 512), resample=Image.Resampling.LANCZOS).convert("RGB")
        while run == True:
        # Add your custom processing here
            image = pipe(prompt=instruction, image=img, num_inference_steps=steps, guidance_scale=text_cfg_scale, image_guidance_scale=image_cfg_scale).images
            return [seed, text_cfg_scale, image_cfg_scale, image[0]]
            #yield target.right_column.image(image[0], use_column_width=True)

    with gr.Blocks(css="footer {visibility: hidden}") as demo:
            with gr.Row():
                with gr.Column(scale=1, min_width=100):
                    generate_button = gr.Button("Generate")
                with gr.Column(scale=1, min_width=100):
                    load_button = gr.Button("Load Example")
                with gr.Column(scale=1, min_width=100):
                    reset_button = gr.Button("Reset")
                with gr.Column(scale=3):
                    instruction = gr.Textbox(lines=1, label="Edit Instruction", interactive=True)

            with gr.Row():
                input_image = gr.Image(label="Input Image", type="pil", interactive=True)
                edited_image = gr.Image(label=f"Edited Image", type="pil", interactive=False)
                input_image.style(height=512, width=512)
                edited_image.style(height=512, width=512)

            with gr.Row():
                steps = gr.Number(value=100, precision=0, label="Steps", interactive=True)
                randomize_seed = gr.Radio(
                    ["Fix Seed", "Randomize Seed"],
                    value="Randomize Seed",
                    type="index",
                    show_label=False,
                    interactive=True,
                )
                seed = gr.Number(value=1371, precision=0, label="Seed", interactive=True)
                randomize_cfg = gr.Radio(
                    ["Fix CFG", "Randomize CFG"],
                    value="Fix CFG",
                    type="index",
                    show_label=False,
                    interactive=True,
                )
                text_cfg_scale = gr.Number(value=7.5, label=f"Text CFG", interactive=True)
                image_cfg_scale = gr.Number(value=1.5, label=f"Image CFG", interactive=True)

            gr.Markdown("help")

            generate_button.click(
                fn=generate,
                inputs=[
                    input_image,
                    instruction,
                    steps,
                    randomize_seed,
                    seed,
                    randomize_cfg,
                    text_cfg_scale,
                    image_cfg_scale,
                ],
                outputs=[seed, text_cfg_scale, image_cfg_scale, edited_image],
            )

    demo.queue(concurrency_count=1)
    demo.launch(share=False)


if __name__ == "__main__":
    main()
