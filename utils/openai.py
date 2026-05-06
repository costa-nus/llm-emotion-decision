from openai import OpenAI
import time


def load_openai_lm(model_path: str, **kwargs):
    client = OpenAI()
    reasoning_effort = kwargs.pop('reasoning_effort', 'low')
    # specific to 'gpt-5'
    if reasoning_effort == 'No':
        reasoning_effort = 'minimal'

    def lm(messages, return_reasoning=False, model_identity=None, **ignored_kwargs):
        if model_identity is not None:
            if messages[0]["role"] == "system":
                messages[0]["content"] = f"{model_identity}\n\n" + \
                    messages[0]["content"]
            else:
                messages.insert(
                    0, {"role": "system", "content": model_identity})
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.responses.create(
                    model=model_path,
                    reasoning={"effort": reasoning_effort.lower(),
                                'summary': 'auto'},
                    input=messages,
                    # max_output_tokens=2048,
                    **kwargs,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    print(f"All {max_retries} attempts failed.")
                    raise
        if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
            print("Ran out of tokens")
            if response.output_text:
                print("Partial output:", response.output_text)
            else:
                print("Ran out of tokens during reasoning")
        if return_reasoning:
            return [{"analysis": response.output[0].summary[0].text if len(response.output[0].summary) > 0 else "", "final": response.output_text}]
        else:
            return [response.output_text]
    return lm