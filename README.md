# NanoLLaDA: Fine-tuning LLaDA
This project involves fine-tuning LLaDA (*L*arge *La*nguage *D*iffusion with m*A*sking) as the final project for the Natural Language Processing (NLP) Systems course (2025) at Chulalongkorn University.

Dataset: [nakhun/thaisum](https://huggingface.co/datasets/nakhun/thaisum)

## Setup
This project uses:
- **Python 3.11**
- **[Poetry](https://python-poetry.org/)**: for dependency management and packaging

### Install Poetry
Poetry Installation Docs: https://python-poetry.org/docs/#installation
- If you don't have `pipx` yet, install it first: https://pipx.pypa.io/stable/installation/
```bash
pipx install poetry
```

### Install dependencies
From the root of the project (`cd NanoLLaDA`):
```bash
poetry install
```
This will:
- Create a virtual environment (in `.venv/`)
- Install all dependencies from `pyproject.toml` and `poetry.lock`

### Activate the virtual environment
```bash
source .venv/bin/activate
```
> You can verify the environment is active with: `which python` → should point to `.venv/bin/python`

### Register poetry environment as a Jupyter kernel
To use your Poetry-managed environment in Jupyter Notebooks (including within VS Code), you need to register it as a kernel.
> Note: If VS Code was open before running this command, you may need to restart VS Code to see the new kernel.
```bash
poetry run ipython kernel install --user --name=nanollada --display-name "Python (nanollada)"
```
> You can verify the kernel is registered with: `jupyter kernelspec list` → should show `nanollada` in the list.


## References
- [LLaDA Paper (arXiv:2502.09992)](https://arxiv.org/abs/2502.09992)
- [Official LLaDA Repository](https://github.com/ML-GSAI/LLaDA)
- [ThaiSum Dataset](https://huggingface.co/datasets/nakhun/thaisum)

## Contributors
- [Kawin Rattanapun](https://github.com/athensclub)
- [Chanatip Pattanapen](https://github.com/demonstem)
- [Pupipat Singkhorn](https://github.com/pupipatsk)
- [Chotpisit Adunsehawat](https://github.com/Nacnano)
