from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    ReasoningEffort,
)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def is_thinking_model(model_path):
    return ("qwen" in model_path.lower()) or ("deepseek" in model_path.lower())

def load_chat_lm(model_path: str, **kwargs):
    model, tokenizer = load_model_and_tokenizer(model_path)
    reasoning_effort = kwargs.pop('reasoning_effort', ReasoningEffort.LOW)
    if reasoning_effort == 'No':
        disable_reasoning = True
    else:
        disable_reasoning = False
    if 'openai/gpt-oss' in model_path:
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        def lm(messages, model_identity=None,
               return_reasoning=False,):
            return harmony_chat_completion(model, enc, messages, model_identity=model_identity,
                                           reasoning_effort=reasoning_effort,
                                           disable_reasoning=disable_reasoning,
                                           return_reasoning=return_reasoning, **kwargs)
    elif 'qwen' in model_path.lower() or 'deepseek' in model_path.lower():
        def lm(messages, model_identity=None,
               return_reasoning=False,):
            return qwen_chat_completion(model, tokenizer, messages, model_identity=model_identity,
                                       disable_reasoning=disable_reasoning,
                                       return_reasoning=return_reasoning, **kwargs)
    else:
        default_gen_kwargs = dict(kwargs)
    
        ppl = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )
        # tokenizer.apply
        if is_thinking_model(model_path):
            def lm(messages, return_reasoning=False, model_identity=None, **call_kwargs):
                if model_identity is not None:
                    if messages[0]["role"] == "system":
                        messages[0]["content"] = f"{model_identity}\n\n" + messages[0]["content"]
                    else:
                        messages.insert(0, {"role": "system", "content": model_identity})
                
                gen_kwargs = {**default_gen_kwargs, **call_kwargs}
                r = ppl(messages, **gen_kwargs)[0]["generated_text"][-1]["content"]
                if "</think>" in r:
                    reasoning, final = r.split("</think>")
                else:
                    reasoning, final = "", r
                if return_reasoning:
                    return [{"analysis": reasoning.strip(), "final": final.strip()}]
                else:
                    return [final.strip()]
        else:
            def lm(messages, return_reasoning=False, model_identity=None, **call_kwargs):
                if model_identity is not None:
                    if messages[0]["role"] == "system":
                        messages[0]["content"] = f"{model_identity}\n\n" + \
                            messages[0]["content"]
                    else:
                        messages.insert(
                            0, {"role": "system", "content": model_identity})
                        
                gen_kwargs = {**default_gen_kwargs, **call_kwargs}
                r = ppl(messages, **gen_kwargs)[0]["generated_text"][-1]["content"]
                if return_reasoning:
                    return [{"analysis": "", "final": r}]
                else:
                    return [r]
        
    return lm

def load_model_and_tokenizer(model_path):
    """Load model and tokenizer, handling LoRA adapters if present."""
    # Try loading as LoRA adapter first
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto"
    )
    print(f"Loaded model from {model_path}")

    # fix_mistral_regex=False can fix the tokenizer offline loading issue for Qwen models
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, fix_mistral_regex=False)
    return model, tokenizer


# --------------------------------------------------
# Setup Harmony encoding (ONCE)
# --------------------------------------------------

def harmony_chat_completion(
    model,
    enc,
    messages,
    max_new_tokens=256,
    temperature=1.,
    do_sample=True,
    top_p=0.9,
    model_identity=None,
    reasoning_effort=ReasoningEffort.LOW,
    disable_reasoning=False,
    return_reasoning=False,
):
    """
    Args:
        model: torch.nn.Module
            The language model to use for generation.
        enc: HarmonyEncoding
            The Harmony encoding to use for encoding/decoding messages.
        messages: List[Dict[str, str]]
        reasoning_effort: ReasoningEffort
            The level of reasoning effort to request from the model. Could be test Low, Medium, High.
    Returns:
        If return_reasoning is False:
            List of assistant final responses (str)
        If return_reasoning is True:
            List of dicts with 'analysis' and 'final' keys
            
    Example: 
        enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        messages = [
            {"role": "system", "content": "Talk like a pirate."},
            {"role": "user", "content": "How are you?"}
        ]
        res = harmony_chat_completion(model, enc, messages, model_identity="You are ChatGPT", return_reasoning=True)
        print(f"reasoning: {res[0]['analysis']}")
        print(f"final: {res[0]['final']}")
    """

    # --------------------------------------------------
    # 1. Convert messages → Harmony Conversation
    # --------------------------------------------------
    harmony_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        channels = ['analysis', 'final'] if not disable_reasoning else ['final']
        
        assert model_identity is not None, "model_identity must be provided"

        harmony_messages.append(
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new()\
                    .with_reasoning_effort(reasoning_effort)\
                        .with_required_channels(channels)\
                            .with_model_identity(model_identity),
            )
        )

        if role == "system":
            harmony_messages.append(
                Message.from_role_and_content(
                    Role.DEVELOPER,
                    DeveloperContent.new().with_instructions(content),
                )
            )
        elif role == "user":
            harmony_messages.append(
                Message.from_role_and_content(Role.USER, content)
            )
        elif role == "assistant":
            harmony_messages.append(
                Message.from_role_and_content(Role.ASSISTANT, content)
            )
        else:
            raise ValueError(f"Unknown role: {role}")

    convo = Conversation.from_messages(harmony_messages)

    # --------------------------------------------------
    # 2. Encode conversation → tokens
    # --------------------------------------------------
    input_ids_raw = enc.render_conversation_for_completion(
        convo,
        Role.ASSISTANT,
    )

    input_ids = torch.tensor([input_ids_raw], device=model.device)

    # --------------------------------------------------
    # 3. Generate
    # --------------------------------------------------
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            eos_token_id=enc.stop_tokens_for_assistant_actions(),
        )

    # --------------------------------------------------
    # 4. Decode assistant messages via Harmony
    # --------------------------------------------------
    generated_tokens = outputs[0][len(input_ids_raw):].tolist()

    parsed_messages = enc.parse_messages_from_completion_tokens(
        generated_tokens,
        role=Role.ASSISTANT,
    )
    # print(r[0].content[0].text)
    if len(parsed_messages) > 1:  # has reasoning
        if return_reasoning:
            return [
                {msg.channel: msg.content[0].text for msg in parsed_messages}
            ]
        else:
            return [msg.content[0].text for msg in parsed_messages if msg.channel == 'final']
    else:
        if return_reasoning:
            return [
                {
                    **{"analysis": None},
                    **{msg.channel: msg.content[0].text for msg in parsed_messages if msg.channel == 'final'}
                }
            ]
        else:
            return [parsed_messages[0].content[0].text]


def qwen_chat_completion(
    model,
    tokenizer,
    messages,
    max_new_tokens=32768,
    temperature=1.,
    do_sample=True,
    top_p=0.9,
    model_identity=None,
    disable_reasoning=False,
    return_reasoning=False,
):
    """
    A simple chat completion function using HuggingFace transformers.
    """
    if model_identity is not None:
        if messages[0]["role"] == "system":
            messages[0]["content"] = f"{model_identity}\n\n" + \
                messages[0]["content"]
        else:
            messages.insert(
                0, {"role": "system", "content": model_identity})
    
    # --------------------------------------------------
    # 2. Encode conversation → tokens
    # --------------------------------------------------
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Switches between thinking and non-thinking modes. Default is True.
        enable_thinking=not disable_reasoning
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    if disable_reasoning:
        content = tokenizer.decode(
            output_ids, skip_special_tokens=True).strip("\n")
        thinking_content = ""
    else:
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        thinking_content = tokenizer.decode(
            output_ids[:index], skip_special_tokens=True).strip("\n")

        content = tokenizer.decode(
            output_ids[index:], skip_special_tokens=True).strip("\n")
    if return_reasoning:
        return [{"analysis": thinking_content, "final": content}]
    else:
        return [content]