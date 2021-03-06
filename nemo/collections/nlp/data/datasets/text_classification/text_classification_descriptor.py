from nemo import logging
from nemo.collections.nlp.data.datasets.datasets_utils import calc_class_weights, get_label_stats, if_exist

__all__ = ['TextClassificationDataDesc']


class TextClassificationDataDesc:
    def __init__(self, data_dir, modes=['train', 'test', 'dev']):
        self.data_dir = data_dir

        max_label = 0
        for mode in modes:
            if not if_exist(self.data_dir, [f'{mode}.tsv']):
                logging.info(f'Stats calculation for {mode} mode is skipped as {mode}.tsv was not found.')
                continue

            input_file = f'{self.data_dir}/{mode}.tsv'
            with open(input_file, 'r') as f:
                input_lines = f.readlines()[1:]  # Skipping headers at index 0

            try:
                int(input_lines[0].strip().split()[-1])
            except ValueError:
                logging.warning(f'No numerical labels found for {mode}.tsv.')
                raise

            queries, raw_sentences = [], []
            for input_line in input_lines:
                parts = input_line.strip().split()
                label = int(parts[-1])
                if label > max_label:
                    max_label = label
                raw_sentences.append(label)
                queries.append(' '.join(parts[:-1]))

            infold = input_file[: input_file.rfind('/')]

            logging.info(f'Three most popular classes in {mode} dataset')
            total_sents, sent_label_freq = get_label_stats(raw_sentences, infold + f'/{mode}_sentence_stats.tsv')

            if mode == 'train':
                self.class_weights = calc_class_weights(sent_label_freq)
                logging.info(f'Class weights are - {self.class_weights}')

            logging.info(f'Total Sentences - {total_sents}')
            logging.info(f'Sentence class frequencies - {sent_label_freq}')

        self.num_labels = max_label + 1
