import argparse
import json
import sys
from pathlib import Path

import k_diffusion
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

sys.path.append("./")
sys.path.append("./stable_diffusion")

from ldm.modules.attention import CrossAttention
from ldm.util import instantiate_from_config
from metrics.clip_similarity import ClipSimilarity


################################################################################
# Modified K-diffusion Euler ancestral sampler with prompt-to-prompt.
# https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


def get_ancestral_step(sigma_from, sigma_to):
    """Calculates the noise level (sigma_down) to step down to and the amount
    of noise to add (sigma_up) when doing an ancestral sampling step."""
    sigma_up = min(sigma_to, (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5)
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def sample_euler_ancestral(model, x, sigmas, prompt2prompt_threshold=0.0, **extra_args):
    """Ancestral sampling with Euler method steps."""
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        prompt_to_prompt = prompt2prompt_threshold > i / (len(sigmas) - 2)
        for m in model.modules():
            if isinstance(m, CrossAttention):
                m.prompt_to_prompt = prompt_to_prompt
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1])
        d = to_d(x, sigmas[i], denoised)
        # Euler method
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            # Make noise the same across all samples in batch.
            x = x + torch.randn_like(x[:1]) * sigma_up
    return x


################################################################################


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cfg_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cfg_scale


def to_pil(image: torch.Tensor) -> Image.Image:
    image = 255.0 * rearrange(image.cpu().numpy(), "c h w -> h w c")
    image = Image.fromarray(image.astype(np.uint8))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output dataset directory.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to prompts .jsonl file.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt",
        help="Path to stable diffusion checkpoint.",
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="stable_diffusion/models/ldm/stable-diffusion-v1/vae-ft-mse-840000-ema-pruned.ckpt",
        help="Path to vae checkpoint.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of sampling steps.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to generate per prompt (before CLIP filtering).",
    )
    parser.add_argument(
        "--max-out-samples",
        type=int,
        default=4,
        help="Max number of output samples to save per prompt (after CLIP filtering).",
    )
    parser.add_argument(
        "--n-partitions",
        type=int,
        default=1,
        help="Number of total partitions.",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="Partition index.",
    )
    parser.add_argument(
        "--min-p2p",
        type=float,
        default=0.1,
        help="Min prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--max-p2p",
        type=float,
        default=0.9,
        help="Max prompt2prompt threshold (portion of denoising for which to fix self attention maps).",
    )
    parser.add_argument(
        "--min-cfg",
        type=float,
        default=7.5,
        help="Min classifier free guidance scale.",
    )
    parser.add_argument(
        "--max-cfg",
        type=float,
        default=15,
        help="Max classifier free guidance scale.",
    )
    parser.add_argument(
        "--clip-threshold",
        type=float,
        default=0.2,
        help="CLIP threshold for text-image similarity of each image.",
    )
    parser.add_argument(
        "--clip-dir-threshold",
        type=float,
        default=0.2,
        help="Directional CLIP threshold for similarity of change between pairs of text and pairs of images.",
    )
    parser.add_argument(
        "--clip-img-threshold",
        type=float,
        default=0.7,
        help="CLIP threshold for image-image similarity.",
    )
    opt = parser.parse_args()

    global_seed = torch.randint(1 << 32, ()).item()
    print(f"Global seed: {global_seed}")
    seed_everything(global_seed)

    model = load_model_from_config(
        OmegaConf.load("stable_diffusion/configs/stable-diffusion/v1-inference.yaml"),
        ckpt=opt.ckpt,
        vae_ckpt=opt.vae_ckpt,
    )
    model.cuda().eval()
    model_wrap = k_diffusion.external.CompVisDenoiser(model)

    clip_similarity = ClipSimilarity().cuda()

    out_dir = Path(opt.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    with open(opt.prompts_file) as fp:
        prompts = [json.loads(line) for line in fp]

    print(f"Partition index {opt.partition} ({opt.partition + 1} / {opt.n_partitions})")
    prompts = np.array_split(list(enumerate(prompts)), opt.n_partitions)[opt.partition]

    with torch.no_grad(), torch.autocast("cuda"), model.ema_scope():
        uncond = model.get_learned_conditioning(2 * [""])
        sigmas = model_wrap.get_sigmas(opt.steps)

        for i, prompt in tqdm(prompts, desc="Prompts"):
            prompt_dir = out_dir.joinpath(f"{i:07d}")
            prompt_dir.mkdir(exist_ok=True)

            with open(prompt_dir.joinpath("prompt.json"), "w") as fp:
                json.dump(prompt, fp)

            cond = model.get_learned_conditioning([prompt["caption"], prompt["output"]])
            results = {}

            with tqdm(total=opt.n_samples, desc="Samples") as progress_bar:

                while len(results) < opt.n_samples:
                    seed = torch.randint(1 << 32, ()).item()
                    if seed in results:
                        continue
                    torch.manual_seed(seed)

                    x = torch.randn(1, 4, 512 // 8, 512 // 8, device="cuda") * sigmas[0]
                    x = repeat(x, "1 ... -> n ...", n=2)

                    model_wrap_cfg = CFGDenoiser(model_wrap)
                    p2p_threshold = opt.min_p2p + torch.rand(()).item() * (opt.max_p2p - opt.min_p2p)
                    cfg_scale = opt.min_cfg + torch.rand(()).item() * (opt.max_cfg - opt.min_cfg)
                    extra_args = {"cond": cond, "uncond": uncond, "cfg_scale": cfg_scale}
                    samples_ddim = sample_euler_ancestral(model_wrap_cfg, x, sigmas, p2p_threshold, **extra_args)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                    x0 = x_samples_ddim[0]
                    x1 = x_samples_ddim[1]

                    clip_sim_0, clip_sim_1, clip_sim_dir, clip_sim_image = clip_similarity(
                        x0[None], x1[None], [prompt["caption"]], [prompt["output"]]
                    )

                    results[seed] = dict(
                        image_0=to_pil(x0),
                        image_1=to_pil(x1),
                        p2p_threshold=p2p_threshold,
                        cfg_scale=cfg_scale,
                        clip_sim_0=clip_sim_0[0].item(),
                        clip_sim_1=clip_sim_1[0].item(),
                        clip_sim_dir=clip_sim_dir[0].item(),
                        clip_sim_image=clip_sim_image[0].item(),
                    )

                    progress_bar.update()

            # CLIP filter to get best samples for each prompt.
            metadata = [
                (result["clip_sim_dir"], seed)
                for seed, result in results.items()
                if result["clip_sim_image"] >= opt.clip_img_threshold
                and result["clip_sim_dir"] >= opt.clip_dir_threshold
                and result["clip_sim_0"] >= opt.clip_threshold
                and result["clip_sim_1"] >= opt.clip_threshold
            ]
            metadata.sort(reverse=True)
            for _, seed in metadata[: opt.max_out_samples]:
                result = results[seed]
                image_0 = result.pop("image_0")
                image_1 = result.pop("image_1")
                image_0.save(prompt_dir.joinpath(f"{seed}_0.jpg"), quality=100)
                image_1.save(prompt_dir.joinpath(f"{seed}_1.jpg"), quality=100)
                with open(prompt_dir.joinpath(f"metadata.jsonl"), "a") as fp:
                    fp.write(f"{json.dumps(dict(seed=seed, **result))}\n")

    print("Done.")


if __name__ == "__main__":
    main()
