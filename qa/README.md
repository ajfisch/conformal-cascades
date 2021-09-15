# Open-Domain Question Answering

Open-domain question answering focuses on using a large-scale corpus D to answer arbitrary questions via search combined with reading comprehension.
We use the open-domain setting of the Natural Questions dataset (Kwiatkowski et al., 2019). Following Chen et al. (2017), we first retrieve relevant passages from Wikipedia using a document retriever, and then select an answer span from the considered passages using a document reader. We use a Dense Passage Retriever model (Karpukhin et al., 2020) for the retriever, and a BERT model (Devlin et al., 2019) for the reader. The BERT model yields several score variants—we use multiple in our cascade (relevance CLS logit, start logit, and end_logit). Any span from any retrieved passage that matches any of the annotated answer strings when lower-case and stripped of articles and punctuation is considered to be correct.

For our predictions, we used results from the [DPR repository](https://github.com/facebookresearch/DPR). 

For every split, we provide a `json` file of our predictions, where each entry is structured like the following: 

```python
[
    {
        "question": "who is the killer in season 1 of broadchurch",                                                           
        "gold_answers": [
            "Joe"
        ],
        "predictions": [
            {
                "text": "joe miller",
                "start_score": 13.041522979736328,                                                                            
                "end_score": 13.02441692352295,                                                                               
                "score": 26.065939903259277,                                                                                  
                "relevance_score": 10.888565063476562,                                                                        
                "passage_idx": 9,
                "passage_id": "18417956",                                                                                     
                "passage_score": 80.00108                                                                                     
            }, 
            ...
```

The `conformal_open_qa.py` script then transforms these input files into the appropriate input formats and runs the conformal experiments.
