import math
import re

import torch

import comfy.model_management
import comfy.utils
import node_helpers
from comfy_api.latest import io


def parse_apply_to_images(spec, count):
    # "1(1.00),2(0.50)" -> {image 1: 1.0, image 2: 0.5}; a single entry applies to every image,
    # with several entries each listed image gets its own share and unlisted images get none
    entries = [(int(m.group(1)), float(m.group(2)))
               for m in re.finditer(r"(\d+)\s*\(\s*([0-9]*\.?[0-9]+)\s*\)", spec or "")]
    if not entries:
        return [1.0] * count
    if len(entries) == 1:
        return [max(0.0, entries[0][1])] * count
    mults = [0.0] * count
    for idx, mult in entries:
        if 1 <= idx <= count:
            mults[idx - 1] = max(0.0, mult)
    return mults


class QwenImage21EditOptions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21EditOptions",
            display_name="Qwen Image 2.1 Edit Options (QQ)",
            category="model/conditioning/qwen image",
            description=(
                "Tuning for how strongly reference images condition Qwen Image 2.1: the model sees each reference both as VAE latents spliced into "
                "the token sequence (pixel level structure) and through the text encoder's understanding of it (semantics). Both at once tends to "
                "overfixate on the reference; these options let you weaken or drop either channel."
            ),
            inputs=[
                io.AnyType.Input("options", optional=True,
                                 tooltip="Options from another Qwen Image 2.1 Edit Options node, to chain settings. Later nodes override the values they set."),
                io.Combo.Input("mode", options=["latent + llm", "llm only"], default="latent + llm",
                               tooltip="How reference images reach the model. 'latent + llm' is the default: VAE latents in the DiT sequence plus the text encoder's understanding of the image. "
                                       "'llm only' drops the latents: the reference conditions purely through the text encoder, for loose edits where the prompt should dominate."),
                io.String.Input("apply_to_images", default="1(1.00)",
                                tooltip="Per-image share of latent_noise as 'index(multiplier)': '1(1.00),2(0.50)' gives image 2 half the latent_noise of image 1. "
                                        "A single entry applies to every image; with several, unlisted images get no noise."),
                io.Float.Input("latent_noise", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend the reference latents toward noise (matching scale) before they enter the DiT sequence. 0 keeps the reference intact, higher values "
                                       "destroy fine detail while keeping overall layout: an adherence dial between 'llm only' and the default. Needs the vae connected."),
                io.Int.Input("noise_seed", default=0, min=0, max=0x7fffffff,
                             tooltip="Seed for the latent noise, so results are reproducible."),
                io.Int.Input("latent_resolution", default=0, min=0, max=4096, step=32,
                             tooltip="Resize the reference latents independently of the text encoder's view: 0 uses the encoder node's resolution. Lower values mean fewer latent "
                                       "tokens in the DiT sequence, a coarser structural anchor with less detail to copy."),
            ],
            outputs=[
                io.AnyType.Output(display_name="options",
                                  tooltip="Connect to the 'options' input of Text Encode Qwen Image 2.1 Options, or to another Edit Options node to chain."),
            ],
        )

    @classmethod
    def execute(cls, options=None, mode="latent + llm", apply_to_images="1(1.00)", latent_noise=0.0, noise_seed=0, latent_resolution=0) -> io.NodeOutput:
        out = dict(options) if isinstance(options, dict) else {}
        out.update({"mode": mode, "apply_to_images": apply_to_images, "latent_noise": latent_noise,
                    "noise_seed": noise_seed, "latent_resolution": latent_resolution})
        return io.NodeOutput(out)


class TextEncodeQwenImage21Options(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TextEncodeQwenImage21Options",
            display_name="Text Encode Qwen Image 2.1 Options (QQ)",
            category="model/conditioning/qwen image",
            description=(
                "Text Encode Qwen Image 2.1 with an options input: connect a Qwen Image 2.1 Edit Options node to control how strongly the reference "
                "images condition the model (latent noise, per-image noise shares, independent latent resolution, or LLM-only conditioning)."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.AnyType.Input("options", optional=True,
                                 tooltip="Settings from a Qwen Image 2.1 Edit Options node. Default behavior when disconnected."),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32,
                             tooltip="Reference images are resized to about resolution x resolution pixels, at multiples of 32, preserving aspect ratio. 0 keeps each reference at its own size, rounded to a multiple of 32. "),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference images, seen by the text encoder and spliced into the sequence as VAE latents.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent",
                                 tooltip="Empty latent on the first reference image's size, to match with sampling as any other size shifts the edit."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, negative_prompt, vae=None, resolution=1024, images: io.Autogrow.Type = None, options=None) -> io.NodeOutput:
        options = options if isinstance(options, dict) else {}
        mode = options.get("mode", "latent + llm")
        use_latents = vae is not None and mode != "llm only"
        # the latents can be resized independently of the vision tower: fewer latent tokens is a coarser structural anchor
        latent_resolution = options.get("latent_resolution", 0)

        def scale_image(image, target):
            samples = image[:1].movedim(-1, 1)
            if target > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = round(math.sqrt(target * target * ratio) / 32) * 32
                height = round(math.sqrt(target * target / ratio) / 32) * 32
            else:
                width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
            width, height = max(32, width), max(32, height)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                return image[:1]
            return comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)

        ref_latents = []
        images_vl = []
        images = images or {}
        latent_w = latent_h = resolution or 1024
        latent_size_set = False
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            # same resize for the text encoder and the VAE, so every vision slot covers 2x2 latents; one image per input
            s = scale_image(image, resolution)
            if use_latents:
                s_l = scale_image(image, latent_resolution) if latent_resolution > 0 else s
                if not latent_size_set:
                    latent_w, latent_h = s_l.shape[2], s_l.shape[1]
                    latent_size_set = True
            elif not latent_size_set:
                latent_w, latent_h = s.shape[2], s.shape[1]
                latent_size_set = True
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])  # the vision tower sees alpha over white, the vae keeps all four
            images_vl.append(rgb)
            if use_latents:
                ref_latents.append(vae.encode(s_l))

        latent_noise = options.get("latent_noise", 0.0)
        if latent_noise > 0.0 and len(ref_latents) > 0:
            # blend each reference latent toward noise of matching scale, per its apply_to_images share
            mults = parse_apply_to_images(options.get("apply_to_images", "1(1.00)"), len(ref_latents))
            generator = torch.Generator().manual_seed(int(options.get("noise_seed", 0)))
            noisy = []
            for l, mult in zip(ref_latents, mults):
                amount = latent_noise * mult
                if amount <= 0.0:
                    noisy.append(l)
                    continue
                # variance preserving blend: corr(result, original) = 1 - amount, std unchanged
                noise = torch.randn(l.shape, generator=generator) * (float(l.float().std()) * math.sqrt(1.0 - (1.0 - amount) ** 2))
                noisy.append(l * (1.0 - amount) + noise.to(device=l.device, dtype=l.dtype))
            ref_latents = noisy

        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True))
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16], device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})
