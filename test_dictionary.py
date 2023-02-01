from nlp.datasets.data_helper import create_or_load_tokenizer

ko_vocab = create_or_load_tokenizer(
    file_path="data/sample/valid_ko",
    save_path="dictionary/sample_valid",
    vocab_size=3000,
    language="ko",
)
