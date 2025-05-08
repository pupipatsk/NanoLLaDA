import asyncio
import concurrent.futures
import time
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

async def progress_reporter(tasks1, tasks2, interval=5):
    """Periodically prints progress for both files."""
    total1 = len(tasks1)
    total2 = len(tasks2)
    while True:
        done1 = sum(task.done() for task in tasks1)
        done2 = sum(task.done() for task in tasks2)
        print(f"gemini_summaries.csv: {done1}/{total1} completed")
        print(f"test-100-1024-generated-ft.csv: {done2}/{total2} completed")
        if done1 == total1 and done2 == total2:
            break
        await asyncio.sleep(interval)

async def main():
    """Main function to process both files concurrently."""
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    loop = asyncio.get_running_loop()

    # Load data from both files
    df1 = pd.read_csv("data/gemini_summaries.csv")
    df2 = pd.read_csv("data/test-100-1024-generated-ft.csv")

    # Prepare input-output pairs
    pairs1 = list(df1[["body", "generated"]].itertuples(index=False, name=None))
    pairs2 = list(df2[["body", "generated"]].itertuples(index=False, name=None))

    # Create tasks for each pair
    tasks1 = [asyncio.create_task(evaluate_pair_async(pair, API_KEY, executor, loop)) for pair in pairs1]
    tasks2 = [asyncio.create_task(evaluate_pair_async(pair, API_KEY, executor, loop)) for pair in pairs2]

    # Start progress reporter
    progress_task = asyncio.create_task(progress_reporter(tasks1, tasks2))

    # Wait for all evaluations to complete
    await asyncio.gather(*tasks1, *tasks2)
    await progress_task

    # Collect results
    results1 = [task.result() for task in tasks1]
    results2 = [task.result() for task in tasks2]

    # Print or process results as needed
    print("Scores for gemini_summaries.csv:", results1)
    print("Scores for test-100-1024-generated-ft.csv:", results2)

if __name__ == "__main__":
    asyncio.run(main())