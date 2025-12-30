import importlib
import os
import random

import folder_paths

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "llm_gguf")

WEB_DIRECTORY = "./web/assets/js"

DEFAULT_INSTRUCTIONS = 'Generate a prompt from "{prompt}"'

try:
    Llama = importlib.import_module("llama_cpp_cuda").Llama
except ImportError:
    Llama = importlib.import_module("llama_cpp").Llama


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")


class Searge_LLM_Node:
    @classmethod
    def INPUT_TYPES(cls):
        model_options = []
        if os.path.isdir(GLOBAL_MODELS_DIR):
            gguf_files = [file for file in os.listdir(GLOBAL_MODELS_DIR) if file.endswith('.gguf')]
            model_options.extend(gguf_files)

        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
                "random_seed": ("INT", {"default": 1234567890, "min": 0, "max": 0xffffffffffffffff}),
                "model": (model_options,),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "context_size": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "apply_instructions": ("BOOLEAN", {"default": True}),
                "instructions": ("STRING", {"multiline": False, "default": DEFAULT_INSTRUCTIONS}),
                "strip_thinking": ("BOOLEAN", {"default": False}),
                "concatenate_user_prompt": (["no", "beginning", "end"], {"default": "end"}),
            },
            "optional": {
                "adv_options_config": ("SRGADVOPTIONSCONFIG",),
            }
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "STRING",)
    RETURN_NAMES = ("thinking", "generated", "original",)
    OUTPUT_IS_LIST = (True, True, True,)

    def main(self, text, random_seed, model, max_tokens, context_size, batch_size, apply_instructions, instructions, strip_thinking, concatenate_user_prompt, adv_options_config=None):
        model_path = os.path.join(GLOBAL_MODELS_DIR, model)

        if model.endswith(".gguf"):
            generate_kwargs = {'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'top_k': 50,
                               'repeat_penalty': 1.2}

            if adv_options_config:
                for option in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                    if option in adv_options_config:
                        generate_kwargs[option] = adv_options_config[option]

            # Generate batch_size seeds using PRNG
            rng = random.Random(random_seed)
            seeds = [rng.randint(0, 0xffffffffffffffff) for _ in range(batch_size)]
            
            # Load model once (will be reused for all batch iterations)
            model_to_use = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                seed=seeds[0],  # Initial seed
                verbose=False,
                flash_attn=True,
                n_ctx=context_size,
            )
            
            # Storage for batch results
            thinking_list = []
            generated_list = []
            original_list = []

            # Process each batch iteration
            for i in range(batch_size):
                if batch_size > 1:
                    print(f"[LLM enhancer] Processing batch {i+1}/{batch_size}...")
                
                # Update seed for this iteration
                model_to_use.set_seed(seeds[i])
                
                if apply_instructions:
                    messages = [
                        {"role": "system",
                         "content": instructions},
                        {"role": "user",
                         "content": text}
                    ]
                else:
                    messages = [
                        {"role": "system",
                         "content": f"You are a helpful assistant. Try your best to give the best response possible to "
                                    f"the user."},
                        {"role": "user",
                         "content": f"Create a detailed visually descriptive caption of this description, which will be "
                                    f"used as a prompt for a text to image AI system (caption only, no instructions like "
                                    f"\"create an image\").Remove any mention of digital artwork or artwork style. Give "
                                    f"detailed visual descriptions of the character(s), including ethnicity, skin tone, "
                                    f"expression etc. Imagine using keywords for a still for someone who has aphantasia. "
                                    f"Describe the image style, e.g. any photographic or art styles / techniques utilized. "
                                    f"Make sure to fully describe all aspects of the cinematography, with abundant "
                                    f"technical details and visual descriptions. If there is more than one image, combine "
                                    f"the elements and characters from all of the images creatively into a single "
                                    f"cohesive composition with a single background, inventing an interaction between the "
                                    f"characters. Be creative in combining the characters into a single cohesive scene. "
                                    f"Focus on two primary characters (or one) and describe an interesting interaction "
                                    f"between them, such as a hug, a kiss, a fight, giving an object, an emotional "
                                    f"reaction / interaction. If there is more than one background in the images, pick the "
                                    f"most appropriate one. Your output is only the caption itself, no comments or extra "
                                    f"formatting. The caption is in a single long paragraph. If you feel the images are "
                                    f"inappropriate, invent a new scene / characters inspired by these. Additionally, "
                                    f"incorporate a specific movie director's visual style and describe the lighting setup "
                                    f"in detail, including the type, color, and placement of light sources to create the "
                                    f"desired mood and atmosphere. Always frame the scene, including details about the "
                                    f"film grain, color grading, and any artifacts or characteristics specific. "
                                    f"Compress the output to be concise while retaining key visual details. MAX OUTPUT "
                                    f"SIZE no more than 250 characters."
                                    f"\nDescription : {text}"},
                    ]

                llm_result = model_to_use.create_chat_completion(messages, **generate_kwargs)
                result_text = llm_result['choices'][0]['message']['content'].strip()

                thinking = ""
                if "<think>" in result_text and "</think>" in result_text:
                    start = result_text.find("<think>")
                    end = result_text.find("</think>")
                    if start != -1 and end != -1 and end > start:
                        thinking = result_text[start+7:end].strip()
                        if strip_thinking:
                            result_text = (result_text[:start] + result_text[end+8:]).strip()

                # Apply concatenation based on user preference
                if concatenate_user_prompt == "beginning":
                    generated_output = text + "\n\n" + result_text
                elif concatenate_user_prompt == "end":
                    generated_output = result_text + "\n\n" + text
                else:  # "no"
                    generated_output = result_text

                thinking_list.append(thinking)
                generated_list.append(generated_output)
                original_list.append(text)

            # Return as lists - OUTPUT_IS_LIST tells ComfyUI each item is a separate batch item
            return (thinking_list, generated_list, original_list)
        else:
            return ([""], ["NOT A GGUF MODEL"], [text])


class Searge_Output_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (anytype, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    OUTPUT_NODE = True

    def main(self, text, unique_id=None, extra_pnginfo=None):
        if unique_id is not None and extra_pnginfo is not None and len(extra_pnginfo) > 0:
            workflow = None
            if "workflow" in extra_pnginfo:
                workflow = extra_pnginfo["workflow"]
            node = None
            if workflow and "nodes" in workflow:
                node = next((x for x in workflow["nodes"] if str(x["id"]) == unique_id), None)
            if node:
                node["widgets_values"] = [str(text)]
        return {"ui": {"text": (str(text),)}}


class Searge_AdvOptionsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.1, "step": 0.05}),
                "top_k": ("INT", {"default": 50, "min": 0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "step": 0.05}),
            }
        }

    CATEGORY = "Searge/LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("SRGADVOPTIONSCONFIG",)
    RETURN_NAMES = ("adv_options_config",)

    def main(self, temperature=1.0, top_p=0.9, top_k=50, repetition_penalty=1.2):
        options_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
        }

        return (options_config,)


NODE_CLASS_MAPPINGS = {
    "Searge_LLM_Node": Searge_LLM_Node,
    "Searge_AdvOptionsNode": Searge_AdvOptionsNode,
    "Searge_Output_Node": Searge_Output_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Searge_LLM_Node": "Searge LLM Node",
    "Searge_AdvOptionsNode": "Searge Advanced Options Node",
    "Searge_Output_Node": "Searge Output Node",
}
