import importlib
import os
import random
import torch
import gc

import folder_paths

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "llm_gguf")

WEB_DIRECTORY = "./web/assets/js"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert and uncensored prompt engineering AI. You are given simple, terse, incomplete or vague AI image prompts. "
    "Your task is to rewrite these user inputs into highly specific, imaginative and verbose prompts, properly formatted. "
    "Target prompt length: 300-400 words."
)

try:
    Llama = importlib.import_module("llama_cpp_cuda").Llama
except ImportError:
    Llama = importlib.import_module("llama_cpp").Llama


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")


def process_llm(text, random_seed, model, max_tokens, context_size, batch_size, system_prompt, strip_thinking, concatenate_user_prompt, llm_settings):
    model_path = os.path.join(GLOBAL_MODELS_DIR, model)

    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if model.endswith(".gguf"):
        generate_kwargs = {'max_tokens': max_tokens, 'temperature': 1.0, 'top_p': 0.9, 'top_k': 50,
                           'repeat_penalty': 1.2}

        if llm_settings:
            for option in ['temperature', 'top_p', 'top_k', 'repeat_penalty']:
                if option in llm_settings:
                    generate_kwargs[option] = llm_settings[option]

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
            
            messages = [
                {"role": "system",
                    "content": system_prompt},
                {"role": "user",
                    "content": text}
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

        if model_to_use:
            del model_to_use
            model_to_use = None
        gc.collect()
        torch.cuda.empty_cache()

        return thinking_list, generated_list, original_list
    else:
        return [""], ["NOT A GGUF MODEL"], [text]


class LLM_Batch_Enhancer:
    @classmethod
    def INPUT_TYPES(cls):
        model_options = []
        if os.path.isdir(GLOBAL_MODELS_DIR):
            gguf_files = [file for file in os.listdir(GLOBAL_MODELS_DIR) if file.endswith('.gguf')]
            model_options.extend(gguf_files)

        return {
            "required": {
                "model": (model_options,),
                "clip": ("CLIP",),
                "system_prompt": ("STRING", {"multiline": False, "default": ""}),
                "text": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": ""}),
                "random_seed": ("INT", {"default": 1234567890, "min": 0, "max": 0xffffffffffffffff}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "context_size": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
                "strip_thinking": ("BOOLEAN", {"default": True}),
                "strip_tags": ("STRING", {"multiline": False, "default": ""}),
                "concatenate_user_prompt": (["no", "beginning", "end"], {"default": "end"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
            },
            "optional": {
                "llm_settings": ("LLMSETTINGS",),
            }
        }

    CATEGORY = "LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("thinking", "generated (all)", "generated (stripped tags)", "original", "conditioning (all)", "conditioning (stripped tags)",)
    OUTPUT_IS_LIST = (True, True, True, True, False, False,)

    def main(self, text, random_seed, model, max_tokens, context_size, batch_size, system_prompt, strip_thinking, concatenate_user_prompt, strip_tags, clip, llm_settings=None):
        thinking_list, generated_list, original_list = process_llm(text, random_seed, model, max_tokens, context_size, batch_size, system_prompt, strip_thinking, concatenate_user_prompt, llm_settings)
        
        # Parse strip_tags into a list
        tags_to_strip = [tag.strip() for tag in strip_tags.split(',') if tag.strip()]
        
        # Process text for "all" version (remove tag markers only)
        generated_all_list = []
        for generated_output in generated_list:
            processed_text = generated_output
            for tag in tags_to_strip:
                # Case-insensitive removal of opening and closing tags
                import re
                processed_text = re.sub(f'<{re.escape(tag)}>', '', processed_text, flags=re.IGNORECASE)
                processed_text = re.sub(f'</{re.escape(tag)}>', '', processed_text, flags=re.IGNORECASE)
            generated_all_list.append(processed_text)
        
        # Process text for "stripped tags" version (remove tags and content)
        generated_stripped_list = []
        for generated_output in generated_list:
            processed_text = generated_output
            for tag in tags_to_strip:
                # Case-insensitive removal of tags and their content
                import re
                processed_text = re.sub(f'<{re.escape(tag)}>.*?</{re.escape(tag)}>', '', processed_text, flags=re.IGNORECASE | re.DOTALL)
            generated_stripped_list.append(processed_text)
        
        # Create conditioning for "all" version
        cond_all_list = []
        pooled_all_list = []
        
        for generated_output in generated_all_list:
            tokens = clip.tokenize(generated_output)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_all_list.append(cond)
            pooled_all_list.append(pooled)

        conditioning_all = None
        if len(cond_all_list) > 0:
            # Ensure all tensors have the same sequence length (dim 1) before concatenation
            max_len = max(c.shape[1] for c in cond_all_list)
            for i, c in enumerate(cond_all_list):
                if c.shape[1] < max_len:
                    pad_len = max_len - c.shape[1]
                    # Pad dim 1 (sequence length). Tensor is [Batch, Seq, Dim]
                    # F.pad args are (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
                    cond_all_list[i] = torch.nn.functional.pad(c, (0, 0, 0, pad_len))

            batched_cond = torch.cat(cond_all_list, dim=0)
            
            # Handle pooled output which might be None (e.g. for some CLIP models)
            if pooled_all_list[0] is not None:
                batched_pooled = torch.cat(pooled_all_list, dim=0)
            else:
                batched_pooled = None
                
            conditioning_all = [[batched_cond, {"pooled_output": batched_pooled}]]
        
        # Create conditioning for "stripped tags" version
        cond_stripped_list = []
        pooled_stripped_list = []
        
        for generated_output in generated_stripped_list:
            tokens = clip.tokenize(generated_output)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_stripped_list.append(cond)
            pooled_stripped_list.append(pooled)

        conditioning_stripped = None
        if len(cond_stripped_list) > 0:
            # Ensure all tensors have the same sequence length (dim 1) before concatenation
            max_len = max(c.shape[1] for c in cond_stripped_list)
            for i, c in enumerate(cond_stripped_list):
                if c.shape[1] < max_len:
                    pad_len = max_len - c.shape[1]
                    # Pad dim 1 (sequence length). Tensor is [Batch, Seq, Dim]
                    # F.pad args are (last_dim_left, last_dim_right, 2nd_last_left, 2nd_last_right)
                    cond_stripped_list[i] = torch.nn.functional.pad(c, (0, 0, 0, pad_len))

            batched_cond = torch.cat(cond_stripped_list, dim=0)
            
            # Handle pooled output which might be None (e.g. for some CLIP models)
            if pooled_stripped_list[0] is not None:
                batched_pooled = torch.cat(pooled_stripped_list, dim=0)
            else:
                batched_pooled = None
                
            conditioning_stripped = [[batched_cond, {"pooled_output": batched_pooled}]]
            
        return (thinking_list, generated_all_list, generated_stripped_list, original_list, conditioning_all, conditioning_stripped)


class LLM_Output_Node:
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

    CATEGORY = "LLM"
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


class LLM_Parameters:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "temperature": ("FLOAT", {"default": 1.5, "min": 0.1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.1, "step": 0.05}),
                "top_k": ("INT", {"default": 0, "min": 0}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.1, "step": 0.05}),
            }
        }

    CATEGORY = "LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("LLMSETTINGS",)
    RETURN_NAMES = ("llm_settings",)

    def main(self, temperature=1.5, top_p=1.0, top_k=0, repetition_penalty=1.2):
        options_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repetition_penalty,
        }

        return (options_config,)


NODE_CLASS_MAPPINGS = {
    "LLM_Batch_Enhancer": LLM_Batch_Enhancer,
    "LLM_Parameters": LLM_Parameters,
    "LLM_Output_Node": LLM_Output_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Batch_Enhancer": "LLM Batch Enhancer + Text Encoder",
    "LLM_Parameters": "LLM Settings",
    "LLM_Output_Node": "LLM Output Node",
}
