from __future__ import annotations

import math


SEMANTIC_STYLE_INSTRUCTIONS = {
    "轻微": (
        "Use the attached reference image as a subtle style guide. Borrow only the broad color palette, "
        "surface texture, lighting mood, and visual rhythm. Do not copy the depicted subject."
    ),
    "平衡": (
        "Use the attached reference image as a style guide. Transfer its visual language, color palette, "
        "material texture, brushwork, lighting, composition rhythm, and overall mood. Do not copy the "
        "depicted subject unless the target prompt asks for it."
    ),
    "强烈": (
        "Use the attached reference image as a strong style guide. Strongly adopt its visual language, "
        "color palette, material texture, brushwork, lighting, composition rhythm, and overall mood, while "
        "keeping the target prompt as the subject and scene."
    ),
}


def _input_meta(display_name, tooltip, **extra):
    meta = {"display_name": display_name, "tooltip": tooltip}
    meta.update(extra)
    return meta


def _resize_image_for_vision(style_image, vision_resolution):
    samples = style_image.movedim(-1, 1)
    total_pixels = max(1, int(vision_resolution) * int(vision_resolution))
    source_pixels = max(1, int(samples.shape[2]) * int(samples.shape[3]))
    scale_by = math.sqrt(total_pixels / source_pixels)
    width = max(1, round(samples.shape[3] * scale_by))
    height = max(1, round(samples.shape[2] * scale_by))
    try:
        import comfy.utils

        samples = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    except ModuleNotFoundError:
        import torch.nn.functional as F

        samples = F.interpolate(samples, size=(height, width), mode="area")
    return samples.movedim(1, -1)[:, :, :, :3]


def _build_style_text(prompt, style_strength, custom_instruction="", fusion=False):
    instruction = (custom_instruction or "").strip()
    if not instruction:
        instruction = SEMANTIC_STYLE_INSTRUCTIONS.get(style_strength, SEMANTIC_STYLE_INSTRUCTIONS["轻微"])

    if fusion:
        instruction = (
            f"{instruction} Keep the target image latent as the source of subject, spatial structure, "
            "pose, and composition; use the attached reference image only as the style guide."
        )
    return f"{instruction}\n\nTarget prompt: {prompt}"


def _encode_style_conditioning(
    clip,
    style_image,
    prompt,
    style_strength="轻微",
    vision_resolution=384,
    custom_instruction="",
    fusion=False,
):
    image_for_clip = _resize_image_for_vision(style_image, vision_resolution)
    text = _build_style_text(prompt, style_strength, custom_instruction, fusion=fusion)
    tokens = clip.tokenize(text, images=[image_for_clip])
    return clip.encode_from_tokens_scheduled(tokens)


def _vae_downscale_ratio(vae):
    if hasattr(vae, "spacial_compression_encode"):
        try:
            return max(1, int(vae.spacial_compression_encode()))
        except Exception:
            pass
    return 8


def _crop_image_for_vae(target_image, vae):
    downscale_ratio = _vae_downscale_ratio(vae)
    height = (target_image.shape[1] // downscale_ratio) * downscale_ratio
    width = (target_image.shape[2] // downscale_ratio) * downscale_ratio
    if height <= 0 or width <= 0:
        raise ValueError(f"target_image is too small for VAE downscale ratio {downscale_ratio}")

    y_offset = (target_image.shape[1] - height) // 2
    x_offset = (target_image.shape[2] - width) // 2
    return target_image[:, y_offset : y_offset + height, x_offset : x_offset + width, :3]


class Krea2StyleSemanticConditioning:
    DESCRIPTION = (
        "Krea2 风格语义条件节点：把参考图通过 Krea2/Qwen3-VL 的图像 token 路径送进 CONDITIONING。"
        "它不修改 diffusion 模型，也不会注入 latent token。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", _input_meta("CLIP", "加载类型应为 krea2 的文本编码器。节点会调用 clip.tokenize(..., images=[参考图])。")),
                "style_image": ("IMAGE", _input_meta("风格参考图", "进入 Qwen3-VL 图像编码路径的参考图。适合提供色彩、材质、笔触、光影和整体视觉语言。")),
                "prompt": ("STRING", _input_meta("正面提示词", "目标图像的文字描述。参考图只作为风格语义输入，不建议在这里重复描述参考图主体。", multiline=True, dynamic_prompts=True)),
                "style_strength": (["轻微", "平衡", "强烈"], _input_meta("语义风格强度", "通过提示词措辞控制参考图影响：轻微更保守，强烈更主动；不是采样器里的数学强度。", default="轻微")),
                "vision_resolution": ("INT", _input_meta("视觉编码分辨率", "参考图送入 Qwen3-VL 前按总像素缩放到此边长平方。384 较稳，512 保留更多细节但更慢。", default=384, min=128, max=1024, step=32)),
            },
            "optional": {
                "custom_instruction": ("STRING", _input_meta("自定义风格指令", "可选。留空使用上方强度预设；填写后会替代预设指令，仍会附加目标提示词。", default="", multiline=True)),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("条件",)
    FUNCTION = "encode"
    CATEGORY = "Krea2/风格"

    def encode(
        self,
        clip,
        style_image,
        prompt,
        style_strength="轻微",
        vision_resolution=384,
        custom_instruction="",
    ):
        conditioning = _encode_style_conditioning(
            clip,
            style_image,
            prompt,
            style_strength,
            vision_resolution,
            custom_instruction,
        )
        return (conditioning,)


class Krea2StyleSemanticFusion:
    DESCRIPTION = (
        "Krea2 风格融合节点：参考图进入 Krea2/Qwen3-VL 语义条件，目标图通过 VAE 编码为初始 latent。"
        "KSampler 使用较低 denoise 时可保留目标图主体、结构和内容，同时迁移参考图风格。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", _input_meta("CLIP", "加载类型应为 krea2 的文本编码器。节点只会把风格参考图送入 clip.tokenize(..., images=[参考图])。")),
                "vae": ("VAE", _input_meta("VAE", "Krea2/Qwen 图像 VAE。节点会用它把目标结构图编码为 KSampler 初始 latent。")),
                "style_image": ("IMAGE", _input_meta("风格参考图", "提供色彩、材质、笔触、光影和整体视觉语言。它会进入 Qwen3-VL 图像编码路径。")),
                "target_image": ("IMAGE", _input_meta("目标结构图", "提供主体、构图、姿势和空间结构。它只会通过 VAE 编码成 latent，不会进入 CLIP 图像条件。")),
                "prompt": ("STRING", _input_meta("正面提示词", "目标图像的文字描述。建议描述目标主体和期望结果，而不是重复参考图主体。", multiline=True, dynamic_prompts=True)),
                "style_strength": (["轻微", "平衡", "强烈"], _input_meta("语义风格强度", "通过提示词措辞控制参考图影响：轻微更保守，强烈更主动；结构保留主要由 KSampler denoise 控制。", default="平衡")),
                "vision_resolution": ("INT", _input_meta("视觉编码分辨率", "风格参考图送入 Qwen3-VL 前按总像素缩放到此边长平方。384 较稳，512 保留更多细节但更慢。", default=384, min=128, max=1024, step=32)),
            },
            "optional": {
                "custom_instruction": ("STRING", _input_meta("自定义风格指令", "可选。留空使用上方强度预设；填写后会替代预设指令，仍会附加目标提示词。", default="", multiline=True)),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("条件", "目标Latent")
    FUNCTION = "encode"
    CATEGORY = "Krea2/风格"

    def encode(
        self,
        clip,
        vae,
        style_image,
        target_image,
        prompt,
        style_strength="平衡",
        vision_resolution=384,
        custom_instruction="",
    ):
        conditioning = _encode_style_conditioning(
            clip,
            style_image,
            prompt,
            style_strength,
            vision_resolution,
            custom_instruction,
            fusion=True,
        )
        target_pixels = _crop_image_for_vae(target_image, vae)
        latent = {"samples": vae.encode(target_pixels)}
        return (conditioning, latent)


NODE_CLASS_MAPPINGS = {
    "Krea2StyleSemanticConditioning": Krea2StyleSemanticConditioning,
    "Krea2StyleSemanticFusion": Krea2StyleSemanticFusion,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2StyleSemanticConditioning": "Krea2 风格语义条件",
    "Krea2StyleSemanticFusion": "Krea2 风格融合",
}
