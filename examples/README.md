## generate.py

This program takes a JSON list of strings from standard input, sends each input string to an LLM as a prompt, and returns a list of dictionaries, with a 'prompt' and 'answer' key for each original prompt.

To execute it, run
```bash
python generate.py
```
And then type some prompts in list form, e.g. 
```bash
["What is spaCy?", "What is Rust?"]
```
followed by an EOF-marker (Ctrl-d on Linux or Mac, Ctrl-z on Windows) and an enter.
The output will look something like this:
> [{"prompt": "What is spaCy?", "answer": "spaCy is a Python library for natural language processing. It is designed to be easy to use and highly customizable, making it a great tool for developers and researchers."}, {"prompt": "What is Rust?", "answer": "Rust is a p
rogramming language that is designed to be a safe, concurrent, and efficient replacement for C++. It is a statically-typed language that is designed to be memory-safe and thread-safe, which means that it can be used to write high-performance, low-latency application
s. Rust is designed to be a modern, high-performance programming language that is designed to be easy to use and easy to learn."}]

To get the full list of options and defaults for this command, run
```bash
python generate.py --help
```