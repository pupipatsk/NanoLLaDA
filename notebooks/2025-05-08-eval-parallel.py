import asyncio
import concurrent.futures
import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import SummarizationMetric
from deepeval.models import GeminiModel
import os
import dotenv
import sys

# Reconfigure stdout to use UTF-8 encoding
sys.stdout.reconfigure(encoding='utf-8')

dotenv.load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

# Configurable input file list
# INPUT_FILES = [
#     "data/test-100-1024-generated-base-08-05-2025.csv",
#     "data/test-100-1024-generated-fine-tuned-08-05-2025.csv"
# ]

INPUT_FILES = [
    "data/langchain-refine.csv",
]

# Generate output file paths with 'score-' prefix in the score directory
OUTPUT_FILES = [
    f"data/score/score-{os.path.basename(file)}"
    for file in INPUT_FILES
]

def evaluate_pair(pair, api_key):
    """Evaluates a single input-output pair synchronously."""
    inp, out = pair
    model = GeminiModel(
        model_name="gemini-2.0-flash",
        api_key=api_key,
        temperature=0,
    )
    metric = SummarizationMetric(
        threshold=0.5,
        model=model,
        n=10
    )
    try:
        test_case = LLMTestCase(input=inp, actual_output=out)
        result = evaluate(test_cases=[test_case], metrics=[metric])
        return result.test_results[0].metrics_data[0].score
    except Exception as e:
        print(f"Error evaluating pair: {e}")
        return 0

async def evaluate_pair_async(pair, api_key, executor, loop):
    """Wraps the synchronous evaluation in an async function using a thread pool."""
    return await loop.run_in_executor(executor, evaluate_pair, pair, api_key)

async def progress_reporter(tasks, file_names, interval=5):
    """Periodically prints progress for all files."""
    total = [len(task_list) for task_list in tasks]
    while True:
        done = [sum(task.done() for task in task_list) for task_list in tasks]
        for i, file_name in enumerate(file_names):
            print(f"{file_name}: {done[i]}/{total[i]} completed")
        if all(done[i] == total[i] for i in range(len(tasks))):
            break
        await asyncio.sleep(interval)

async def main():
    """Main function to process all files concurrently."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    loop = asyncio.get_running_loop()

    # Load data from all input files
    dfs = [pd.read_csv(file) for file in INPUT_FILES]

    # Prepare input-output pairs for each file
    pairs = [list(df[["body", "generated"]].itertuples(index=False, name=None)) for df in dfs]

    # Create tasks for each pair in each file
    tasks = [[asyncio.create_task(evaluate_pair_async(pair, API_KEY, executor, loop)) for pair in pair_list] for pair_list in pairs]

    # Start progress reporter
    progress_task = asyncio.create_task(progress_reporter(tasks, INPUT_FILES))

    # Wait for all evaluations to complete
    await asyncio.gather(*[task for task_list in tasks for task in task_list])
    await progress_task

    # Collect results
    results = [[task.result() for task in task_list] for task_list in tasks]

    # Print results
    for i, file_name in enumerate(INPUT_FILES):
        print(f"Scores for {file_name}:", results[i])

    # Append scores to DataFrames and save to output files
    for i, df in enumerate(dfs):
        df['score'] = results[i]
        df.to_csv(OUTPUT_FILES[i], index=False)
        print(f"Scores appended to {OUTPUT_FILES[i]}")

if __name__ == "__main__":
    asyncio.run(main())