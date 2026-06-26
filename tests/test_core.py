import importlib.util
import json
import pathlib
import sys
import unittest


class Krea2StyleSemanticConditioningTests(unittest.TestCase):
    def load_custom_node_module(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        module_name = str(root).replace(".", "_x_")
        spec = importlib.util.spec_from_file_location(module_name, root / "__init__.py")
        module = importlib.util.module_from_spec(spec)
        old_module = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
        return module

    def test_package_only_registers_semantic_conditioning_node(self):
        module = self.load_custom_node_module()

        self.assertEqual(set(module.NODE_CLASS_MAPPINGS), {"Krea2StyleSemanticConditioning"})
        self.assertEqual(
            module.NODE_DISPLAY_NAME_MAPPINGS,
            {"Krea2StyleSemanticConditioning": "Krea2 风格语义条件"},
        )
        self.assertFalse(hasattr(module, "WEB_DIRECTORY"))

    def test_semantic_conditioning_schema_uses_chinese_labels(self):
        module = self.load_custom_node_module()
        node = module.NODE_CLASS_MAPPINGS["Krea2StyleSemanticConditioning"]
        input_types = node.INPUT_TYPES()

        self.assertEqual(node.RETURN_TYPES, ("CONDITIONING",))
        self.assertEqual(node.RETURN_NAMES, ("条件",))
        self.assertEqual(node.CATEGORY, "Krea2/风格")

        required = input_types["required"]
        self.assertEqual(required["clip"][1]["display_name"], "CLIP")
        self.assertEqual(required["style_image"][1]["display_name"], "风格参考图")
        self.assertEqual(required["prompt"][1]["display_name"], "正面提示词")
        self.assertEqual(required["style_strength"][1]["display_name"], "语义风格强度")
        self.assertEqual(required["vision_resolution"][1]["display_name"], "视觉编码分辨率")
        self.assertEqual(required["style_strength"][1]["default"], "轻微")
        self.assertIn("custom_instruction", input_types["optional"])

    def test_semantic_conditioning_passes_reference_image_to_krea2_clip(self):
        import torch

        module = self.load_custom_node_module()

        class FakeClip:
            def __init__(self):
                self.text = None
                self.images = None

            def tokenize(self, text, images=None):
                self.text = text
                self.images = images
                return {"tokens": "ok"}

            def encode_from_tokens_scheduled(self, tokens):
                return [("conditioning", {"tokens": tokens})]

        clip = FakeClip()
        style_image = torch.zeros(1, 32, 16, 4)
        node = module.NODE_CLASS_MAPPINGS["Krea2StyleSemanticConditioning"]()

        (conditioning,) = node.encode(
            clip,
            style_image,
            "a cinematic portrait",
            "轻微",
            384,
            "",
        )

        self.assertEqual(conditioning, [("conditioning", {"tokens": {"tokens": "ok"}})])
        self.assertIn("style guide", clip.text)
        self.assertIn("a cinematic portrait", clip.text)
        self.assertEqual(len(clip.images), 1)
        self.assertEqual(tuple(clip.images[0].shape[-1:]), (3,))

    def test_example_workflow_uses_semantic_node_only(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        workflow_dir = root / "workflows"
        workflows = sorted(path.name for path in workflow_dir.glob("*.json"))
        self.assertEqual(workflows, ["krea2_style_reference_semantic_example.json"])

        workflow = json.loads((workflow_dir / workflows[0]).read_text(encoding="utf-8"))
        node_types = [node.get("type") for node in workflow.get("nodes", [])]

        self.assertIn("Krea2StyleSemanticConditioning", node_types)
        self.assertNotIn("Krea2StyleReference", node_types)

        semantic_node = next(node for node in workflow["nodes"] if node.get("type") == "Krea2StyleSemanticConditioning")
        self.assertEqual(semantic_node["widgets_values"][1], "轻微")
        self.assertEqual(semantic_node["widgets_values"][2], 768)


if __name__ == "__main__":
    unittest.main()
