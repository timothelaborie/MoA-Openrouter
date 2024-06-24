from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from loguru import logger
from utils import generate_with_references, generate_together_stream, DEBUG
from rich.console import Console
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()

app = FastAPI()
console = Console()

default_reference_models = [
    "openai/gpt-4o",
    "qwen/qwen-2-72b-instruct",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-pro-1.5",
    "meta-llama/llama-3-70b-instruct",
]

class QueryRequest(BaseModel):
    instruction: str
    temperature: float = 0.7
    max_tokens: int = 2048
    rounds: int = 1

class QueryResponse(BaseModel):
    output: str

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_fn(
    instruction: str,
    temperature: float,
    max_tokens: int,
    rounds: int,
    reference_models: List[str] = default_reference_models,
) -> str:
    """
    Processes a single instruction using specified model parameters to generate a response.

    Args:
        instruction (str): The user's input or prompt for which the response is to be generated.
        temperature (float): Controls the randomness and creativity of the generated response. A higher temperature
                             results in more varied outputs. Default is 0.7.
        max_tokens (int): The maximum number of tokens to generate. This restricts the length of the model's response.
                          Default is 2048.
        rounds (int): The number of processing rounds to refine the responses.
        reference_models (List[str]): A list of model identifiers that are used as references in the initial rounds of generation.

    Returns:
        str: The generated response.
    """

    data = {
        "instruction": [[{"role": "user", "content": instruction}] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    for i_round in range(rounds):
        references = []

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    generate_with_references,
                    model=model,
                    messages=[{"role": "user", "content": instruction}],
                    references=data["references"],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ): model for model in reference_models
            }

            for future in as_completed(futures):
                model = futures[future]
                try:
                    output = future.result()
                    print(f"Round {i_round+1} - Model: {model}")
                    references.append(output)
                except Exception as e:
                    print(f"Model {model} generated an exception: {e}")
                    references.append("")

        data["references"] = references

    final_model = reference_models[0]
    print(f"Final Round - Model: {final_model}")
    final_output = generate_with_references(
        model=final_model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": instruction}],
        references=references,
        generate_fn=generate_together_stream,
    )

    all_output = ""
    for chunk in final_output:
        out = chunk.choices[0].delta.content
        all_output += out

    if DEBUG:
        logger.info(f"model: {final_model}, instruction: {instruction}, output: {all_output[:20]}")

    return all_output



@app.post("/generate", response_model=QueryResponse)
async def generate_response(query: QueryRequest):
    try:
        output = process_fn(
            instruction=query.instruction,
            temperature=query.temperature,
            max_tokens=query.max_tokens,
            rounds=query.rounds,
        )
        return QueryResponse(output=output)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=1234)