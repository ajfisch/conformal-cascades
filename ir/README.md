# Information Retrieval for Fact Verification

The goal of IR for fact verification is to retrieve a sentence that can be used to support or refute a given claim. We use the FEVER dataset ([Thorne et al., 2018](https://arxiv.org/abs/1803.05355)), in which evidence is sourced from a set of ∼40K sentences collected from Wikipedia. A sentence that provides enough evidence for the correct verdict (true/false) is considered to be acceptable (multiple are labeled in the dataset). Our cascade consists of (1) a fast, non-neural BM25 similarity score between a given claim and sentence, and (2) the score of an ALBERT model ([Lan et al., 2020](https://arxiv.org/abs/1909.11942)) trained to directly predict if a given claim and sentence are related.

Training the FEVER model was done using the HuggingFace transformers library, while BM25 scores were computed using Gensim. We provide the outputs of our scoring steps for val and test in the `data/ir` directory created after following the download instructions in the root directory.

For every split, we provide the "gold" dataset file and a file of our predictions, where each entry (in `jsonlines` format) is structured like the following: 

```python
{
  'annotation_id': '100030',
  'classification': 'REFUTES',
  'docid': 'Steve_Wozniak',
  'evidence': 'He primarily designed the 1977 Apple II , known as one of the first highly successful mass-produced microcomputers , while Jobs oversaw the development of its unusual case and Rod Holt developed the unique power supply .',
  'gold_docid': 'Steve_Wozniak',
  'label': 'Correct',
  'query': 'Steve Wozniak designed homes.',
  'sent_bm25': 17.344631996194426,
  'sent_logit': -2.0979771614074707,
  'sent_prob': 0.013705157674849033,
  'sentence_ind': 3
}
```

The `conformal_ir.py` script then transforms these input files into the appropriate input formats and runs the conformal experiments.
