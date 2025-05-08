import asyncio
import os
from dotenv import load_dotenv # Your import
load_dotenv() # Your call
import pandas as pd
from typing import List, Tuple # For type hinting

# Correct imports from deepeval
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric, BaseMetric # Added BaseMetric for typing
from deepeval.models import GeminiModel

# Definition of evaluate_one_pair_async (remains unchanged from your provided code)
async def evaluate_one_pair_async(
    input_text: str, 
    actual_output: str, 
    metrics_for_this_case: List[BaseMetric], 
    pair_idx: int
) -> float:
    test_case = LLMTestCase(input=input_text, actual_output=actual_output)
    try:
        await test_case.a_evaluate(metrics=metrics_for_this_case)
        if test_case.metrics_metadata and len(test_case.metrics_metadata) > 0:
            score = test_case.metrics_metadata[0].score 
            return float(score)
        else:
            print(f"Warning: No metrics metadata found for pair {pair_idx + 1} after evaluation.")
            return 0.0
    except Exception as e:
        print(f"Error evaluating pair {pair_idx + 1} (Input: '{input_text[:50]}...'): {e}")
        return 0.0

# Definition of summarize_many_pairs_async (remains unchanged from your provided code)
async def summarize_many_pairs_async(
    input_output_pairs: List[Tuple[str, str]],
    gemini_model_name: str = "gemini-1.5-flash",
    api_key: str = os.getenv("GOOGLE_API_KEY")
) -> List[float]:
    eval_model = GeminiModel(
        model=gemini_model_name,
        temperature=0.0,
        api_key=api_key
    )
    tasks = []
    for idx, (inp, out) in enumerate(input_output_pairs):
        per_task_summarization_metric = SummarizationMetric(
            threshold=0.5, 
            model=eval_model, 
            n=10 
        )
        tasks.append(evaluate_one_pair_async(inp, out, [per_task_summarization_metric], idx))
    print(f"Starting evaluation of {len(tasks)} pairs concurrently using model '{gemini_model_name}' for the metric...")
    list_of_scores = await asyncio.gather(*tasks)
    print("All evaluations finished.")
    return list_of_scores

# MODIFIED Main execution block
async def main():
    # load_dotenv() is already called at the top of your script.

    # Prioritize GOOGLE_API_KEY as DeepEval's GeminiModel expects it.
    # If not found, try GEMINI_API_KEY and then explicitly set GOOGLE_API_KEY in the environment.
    
    api_key_to_use = os.getenv("GOOGLE_API_KEY")
    source_env_variable = "GOOGLE_API_KEY"

    if not api_key_to_use:
        api_key_to_use = os.getenv("GEMINI_API_KEY")
        source_env_variable = "GEMINI_API_KEY"
        if api_key_to_use:
            print(f"API key found in {source_env_variable}. Setting GOOGLE_API_KEY environment variable for DeepEval.")
            os.environ["GOOGLE_API_KEY"] = api_key_to_use # CRITICAL STEP FOR DEEPEVAL
        else:
            print("Error: Neither GOOGLE_API_KEY nor GEMINI_API_KEY found in environment variables.")
            print("Please ensure GOOGLE_API_KEY is set in your .env file or system environment.")
            return
    else:
        print(f"API key found in {source_env_variable}.")

    # CSV loading and processing (remains unchanged from your provided code)
    csv_path = "data/gemini_summaries.csv" 
    try:
        print(f"Attempting to load data from '{csv_path}'...")
        df_full = pd.read_csv(csv_path)
        if len(df_full) == 0:
            print("CSV file is empty. Using dummy data instead.")
            raise FileNotFoundError 
        sample_size = min(10, len(df_full)) 
        df = df_full.sample(sample_size, random_state=42) 
        print(f"Loaded and sampled {len(df)} pairs from '{csv_path}'.")
    except FileNotFoundError:
        print(f"Warning: CSV file '{csv_path}' not found. Using dummy data for demonstration (5 pairs).")
        data = {
            'body': [
                "This is a long document about artificial intelligence. It discusses various concepts such as large language models, natural language processing, and machine learning. The goal is to provide a comprehensive overview.",
                "The history of space exploration is vast, starting from early rockets to ambitious missions to Mars and beyond. Key milestones include the first human in space and the moon landing.",
                "A detailed recipe for baking a classic chocolate cake from scratch, including ingredients, step-by-step instructions, and tips for a perfect bake.",
                "Climate change significantly impacts polar bears and their arctic habitat. Rising temperatures lead to melting sea ice, which is crucial for their hunting and survival.",
                "An analysis of the global economy in the post-pandemic era reveals shifts in consumer behavior, supply chain disruptions, and the rise of remote work."
            ],
            'generated': [
                "AI document summary: covers LLMs, NLP, and ML.",
                "Space exploration history: from rockets to Mars missions, including moon landing.",
                "Classic chocolate cake recipe with instructions.",
                "Climate change affects polar bears due to melting arctic sea ice.",
                "Post-pandemic global economy: changes in behavior, supply chains, remote work."
            ]
        }
        df = pd.DataFrame(data)
    except Exception as e:
        print(f"Error loading or sampling data: {e}. Exiting.")
        return
    
    if df.empty:
        print("No data to process after loading/sampling. Exiting.")
        return

    print(f"\nProcessing {len(df)} input-output pairs for summarization evaluation...")
    input_output_data: List[Tuple[str, str]] = [
        (str(row_body), str(row_generated)) 
        for row_body, row_generated in df[["body", "generated"]].itertuples(index=False, name=None)
    ]

    evaluation_gemini_model = "gemini-1.5-flash" 
    
    scores = await summarize_many_pairs_async(
        input_output_pairs=input_output_data,
        gemini_model_name=evaluation_gemini_model,
        api_key=api_key_to_use
    )

    print("\n--- Evaluation Results ---")
    if len(scores) == len(df):
        df['evaluation_score'] = scores
        for i in range(len(df)):
            # original_text_preview = df['body'].iloc[i][:70].replace("\n", " ") + "..."
            # generated_summary_preview = df['generated'].iloc[i][:70].replace("\n", " ") + "..."
            score_value = df['evaluation_score'].iloc[i]
            print(f"Pair {i+1}: Score = {score_value:.4f}")
        print("\nDataFrame with evaluation_score:")
        print(df[['body', 'generated', 'evaluation_score']].head())
    else:
        print("Mismatch between the number of scores returned and DataFrame rows.")
        print("Scores:", scores)

if __name__ == "__main__":
    # This try-except block is for the RuntimeError if an event loop is already running (e.g. in Jupyter)
    # The ValueError for API key should be handled by the logic within main().
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print(
                "---------------------------------------------------------------------------\n"
                "Failed to run asyncio.run(main()) because an event loop is already running.\n"
                "This is common in environments like Jupyter Notebooks or IPython.\n\n"
                "You have a couple of options:\n\n"
                "1. IF YOU ARE IN A JUPYTER NOTEBOOK OR IPYTHON:\n"
                "   You can typically await the main function directly in a cell.\n"
                "   First, make sure all function definitions (like main, summarize_many_pairs_async, etc.)\n"
                "   have been executed. Then, in a new cell, simply run:\n\n"
                "       await main()\n\n"
                "2. USE 'nest_asyncio' (for environments where you need nested loops):\n"
                "   This library allows asyncio event loops to be nested.\n"
                "   a. Install it if you haven't: pip install nest_asyncio\n"
                "   b. Add these lines at the very top of your script/notebook (before other asyncio imports):\n"
                "      import nest_asyncio\n"
                "      nest_asyncio.apply()\n"
                "   After applying nest_asyncio, the original asyncio.run(main()) in the script MIGHT work directly.\n"
                "---------------------------------------------------------------------------"
            )
        else:
            raise # Re-raise any other RuntimeError