import subprocess

def run(nb):
    subprocess.run([
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute", nb,
        "--ExecutePreprocessor.timeout=-1",
        "--output", nb.replace(".ipynb", "_out.ipynb")
    ], check=True)

run("transformer_akkadian_english.ipynb")