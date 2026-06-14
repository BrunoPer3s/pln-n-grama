from datasets import load_dataset
dataset = load_dataset("TucanoBR/GigaVerbo", split='train', streaming=True)
with open("saida.txt", "w", encoding="utf-8") as arquivo:
    for i, example in enumerate(dataset):
        if i >= 2000:  # Para após 'x' exemplos
            break
        arquivo.write(f"{example['text']}\n")
        arquivo.write("-" * 50 + "\n")
