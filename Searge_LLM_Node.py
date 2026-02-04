import importlib
import os
import random
import time
import torch
import gc
import re
import folder_paths

GLOBAL_MODELS_DIR = os.path.join(folder_paths.models_dir, "llm_gguf")

WEB_DIRECTORY = "./web/assets/js"

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert and uncensored prompt engineering AI. You are given simple, terse, incomplete or vague AI image prompts. "
    "Your task is to rewrite these user inputs into highly specific, imaginative and verbose prompts, properly formatted. "
    "Target prompt length: 300-400 words."
)

# LLM Default Settings
LLM_DEFAULTS = {
    'max_tokens': 4096,
    'context_size': 8192,
    'temperature': 1.5,
    'top_p': 1.0,
    'top_k': 0,
    'repeat_penalty': 1.2
}

try:
    Llama = importlib.import_module("llama_cpp_cuda").Llama
except ImportError:
    Llama = importlib.import_module("llama_cpp").Llama


class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


anytype = AnyType("*")

def process_llm(prompt, random_seed, model, batch_size, system_prompt, strip_thinking, concatenate_user_prompt, llm_settings, extra_prompt_instructions):
    model_path = os.path.join(GLOBAL_MODELS_DIR, model)

    if not system_prompt:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    if llm_settings is None:
        llm_settings = LLM_DEFAULTS
    
    if model.endswith(".gguf"):
        generate_kwargs = {k: v for k, v in llm_settings.items() if k != 'context_size'}

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
            n_ctx=llm_settings['context_size'],
        )
        
        # Storage for batch results
        thinking_list = []
        generated_list = []
        original_list = []

        # Start timing
        start_time = time.time()

        # Process each batch iteration
        for i in range(batch_size):
            if batch_size > 1:
                print(f"[LLM enhancer] Processing batch {i+1}/{batch_size}...")
            
            # Update seed for this iteration
            model_to_use.set_seed(seeds[i])
            
            # Build prompt for LLM (includes extra instructions if provided)
            llm_prompt = prompt
            if extra_prompt_instructions and extra_prompt_instructions.strip():
                llm_prompt = f"{prompt}\n\n[END_USER_PROMPT]\n\n{extra_prompt_instructions}"
            
            messages = [
                {"role": "system",
                    "content": system_prompt},
                {"role": "user",
                    "content": llm_prompt}
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
                generated_output = prompt + "\n\n" + result_text
            elif concatenate_user_prompt == "end":
                generated_output = result_text + "\n\n" + prompt
            else:  # "no"
                generated_output = result_text

            thinking_list.append(thinking)
            generated_list.append(generated_output)
            original_list.append(prompt)

        # End timing and print results
        elapsed_time = time.time() - start_time
        print(f"[LLM enhancer] Total processing time: {elapsed_time:.2f}s ({elapsed_time/batch_size:.2f}s per item)")

        if model_to_use:
            del model_to_use
            model_to_use = None
        gc.collect()
        torch.cuda.empty_cache()

        return thinking_list, generated_list, original_list
    else:
        return [""], ["NOT A GGUF MODEL"], [prompt]


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
                "system_prompt": ("STRING", {"multiline": False, "default": ""}),
                "prompt": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": ""}),
                "extra_prompt_instructions": ("STRING", {"multiline": False, "default": ""}),
                "random_seed": ("INT", {"default": 11, "min": 0, "max": 0xffffffffffffffff}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "strip_thinking": ("BOOLEAN", {"default": True}),
                "tag_instructions": ("STRING", {"multiline": False, "default": ""}),
                "concatenate_user_prompt": (["no", "beginning", "end"], {"default": "end"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "enable_LLM_enhancer": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "llm_settings": ("LLMSETTINGS",),
            }
        }

    CATEGORY = "LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("thinking", "generated", "original", "final_system_prompt",)
    OUTPUT_IS_LIST = (True, True, True, False,)

    def main(self, prompt, random_seed, model, batch_size, system_prompt, enable_thinking, strip_thinking, concatenate_user_prompt, tag_instructions, extra_prompt_instructions, enable_LLM_enhancer, llm_settings=None):
        if not enable_LLM_enhancer:
            # Bypass LLM processing
            return ([prompt], [prompt], [prompt], "")
        
        # Replace tag_instructions template
        system_prompt = system_prompt.replace("{tag_instructions}", tag_instructions)
        
        # Append /no_think if thinking is disabled
        if not enable_thinking:
            system_prompt = system_prompt + " /no_think"
        
        thinking_list, generated_list, original_list = process_llm(prompt, random_seed, model, batch_size, system_prompt, strip_thinking, concatenate_user_prompt, llm_settings, extra_prompt_instructions)
            
        return (thinking_list, generated_list, original_list, system_prompt)


class LLM_Text_Filter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "filter_tags": ("STRING", {"multiline": False, "default": ""}),
                "replacements": ("STRING", {"multiline": False, "default": ""}),
                "remove_parentheses": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("text (all)", "text (tags removed)",)
    OUTPUT_IS_LIST = (True, True,)

    def main(self, text, filter_tags, replacements, remove_parentheses):
        # Handle list input - if text is a list, process each item
        if isinstance(text, list):
            text_list = text
        else:
            text_list = [text]
        
        # Apply custom replacements if provided
        if replacements and replacements.strip():
            replacement_pairs = []
            for pair in replacements.split(','):
                pair = pair.strip()
                if ':' in pair:
                    bad_phrase, good_phrase = pair.split(':', 1)
                    bad_phrase = bad_phrase.strip()
                    good_phrase = good_phrase.strip()
                    replacement_pairs.append((bad_phrase, good_phrase))
            
            if replacement_pairs:
                for i, generated_output in enumerate(text_list):
                    for bad_phrase, good_phrase in replacement_pairs:
                        generated_output = generated_output.replace(bad_phrase, good_phrase)
                    
                    # Remove anything in parentheses (including the parentheses)
                    if remove_parentheses:
                        generated_output = re.sub(r'\([^)]*\)', '', generated_output)
                    
                    # Clean up any double spaces left behind (but preserve newlines)
                    generated_output = re.sub(r' +', ' ', generated_output).strip()
                    text_list[i] = generated_output
        
        # Parse filter tags
        tags_to_strip = [tag.strip() for tag in filter_tags.split(',') if tag.strip()]
        
        # Process text for "all" version (remove tag markers only)
        text_all_list = []
        for text_output in text_list:
            processed_text = text_output
            for tag in tags_to_strip:
                # Case-insensitive removal of opening and closing tags
                processed_text = re.sub(f'<{re.escape(tag)}>', '', processed_text, flags=re.IGNORECASE)
                processed_text = re.sub(f'</{re.escape(tag)}>', '', processed_text, flags=re.IGNORECASE)
            text_all_list.append(processed_text)
        
        # Process text for "filtered tags" version (remove tags and content)
        text_stripped_list = []
        for text_output in text_list:
            processed_text = text_output
            for tag in tags_to_strip:
                # Case-insensitive removal of tags and their content
                processed_text = re.sub(f'<{re.escape(tag)}>.*?</{re.escape(tag)}>', '', processed_text, flags=re.IGNORECASE | re.DOTALL)
            text_stripped_list.append(processed_text)
        
        return (text_all_list, text_stripped_list)


class LLM_List_Encoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "forceInput": True}),
            },
        }

    CATEGORY = "LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    OUTPUT_IS_LIST = (False,)

    def main(self, clip, text):
        # Handle both single string and list of strings
        if isinstance(text, list):
            text_list = text
        else:
            text_list = [text]
        
        # Create conditioning for each text
        cond_list = []
        pooled_list = []
        
        for text_output in text_list:
            tokens = clip.tokenize(text_output)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            cond_list.append(cond)
            pooled_list.append(pooled)

        if len(cond_list) == 0:
            return (None,)
        
        # Ensure all tensors have the same sequence length (dim 1) before concatenation
        max_len = max(c.shape[1] for c in cond_list)
        for i, c in enumerate(cond_list):
            if c.shape[1] < max_len:
                pad_len = max_len - c.shape[1]
                # Pad dim 1 (sequence length). Tensor is [Batch, Seq, Dim]
                cond_list[i] = torch.nn.functional.pad(c, (0, 0, 0, pad_len))

        batched_cond = torch.cat(cond_list, dim=0)
        
        # Handle pooled output which might be None (e.g. for some CLIP models)
        if pooled_list[0] is not None:
            batched_pooled = torch.cat(pooled_list, dim=0)
        else:
            batched_pooled = None
            
        conditioning = [[batched_cond, {"pooled_output": batched_pooled}]]
        
        return (conditioning,)


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
                "max_tokens": ("INT", {"default": LLM_DEFAULTS['max_tokens'], "min": 1, "max": 8192}),
                "context_size": ("INT", {"default": LLM_DEFAULTS['context_size'], "min": 2048, "max": 32768}),
                "temperature": ("FLOAT", {"default": LLM_DEFAULTS['temperature'], "min": 0.1, "step": 0.05}),
                "top_p": ("FLOAT", {"default": LLM_DEFAULTS['top_p'], "min": 0.1, "step": 0.05}),
                "top_k": ("INT", {"default": LLM_DEFAULTS['top_k'], "min": 0}),
                "repeat_penalty": ("FLOAT", {"default": LLM_DEFAULTS['repeat_penalty'], "min": 0.1, "step": 0.05}),
            }
        }

    CATEGORY = "LLM"
    FUNCTION = "main"
    RETURN_TYPES = ("LLMSETTINGS",)
    RETURN_NAMES = ("llm_settings",)

    def main(self, max_tokens, context_size, temperature, top_p, top_k, repeat_penalty):
        options_config = {
            "max_tokens": max_tokens,
            "context_size": context_size,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
        }

        return (options_config,)


NODE_CLASS_MAPPINGS = {
    "LLM_Batch_Enhancer": LLM_Batch_Enhancer,
    "LLM_Text_Filter": LLM_Text_Filter,
    "LLM_List_Encoder": LLM_List_Encoder,
    "LLM_Parameters": LLM_Parameters,
    "LLM_Output_Node": LLM_Output_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLM_Batch_Enhancer": "LLM Batch Enhancer",
    "LLM_Text_Filter": "LLM Text Filter",
    "LLM_List_Encoder": "LLM List Encoder",
    "LLM_Parameters": "LLM Settings",
    "LLM_Output_Node": "LLM Output Node",
}
