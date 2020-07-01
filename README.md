# TASLP_2020_ERPLearning
This is the repository fot the paper Event Representation Learning Enhanced With Knowledege

## Requirements:  
`Python 3.7`   
`Java 1.7`  
`Scala 2.9.0` (for preprocessing NYT corpus)  
`Huggingface Transformer`  

## Data:
This paper uses following data:  
`Transitive Similarity Data`  
`Yago Knowledge Base`, which we only take the Yago_Facts part, you can access it at the [home page](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/yago)    
`Hard Similarity Dataset`  
`ATOMIC Dataset`  
`NYT Dataset`  
`ROC Dataset`  

The pretrained word embedding is glove embedding, for the 'PerosonX' and 'PersonY' embedding , we take average of 20 most used people's name as its' embedding,the embedding could be downloaded at [google drive](https://drive.google.com/file/d/1Jw-X-mVci5VbKg0Gl0ZRRwhgfgyiZ7Vc/view?usp=drive_open).   

## Processing:  
For NYT corpus preprocessing , see [this](https://github.com/MagiaSN/CommonsenseERL_EMNLP_2019/blob/master/preproc/README.md).  
You also need to run event_process.py to have a data that combines NYT data with Yago data, since Yago data is huge, this process may take a while to run.  
### Train:  
Firstly, you should train the event representation model NTN using NYT dataset. This model is described in  the AAAI 2018 *paper Event Representations with Tensor-based Compositions* .    

* `train_event_prediction_on_nyt.py`,this code pretrains the event representation model on the NYT corpus with "event prediction" objective.  

* `train_word_prediction_on_nyt.py`,this code pretrains the event representation model on the NYT corpus with "word prediction" objective.   

After pretraining on the NYT corpus, train the event representation model on the ATOMIC dataset, using the following script:  
`joint_train_on_atomic_stack_prop.py`,this code trains the event representation model on the ATOMIC dataset, with additional intent prediction and sentiment classification objective.  

## Test:
### For the hard similarity task, run this script:  
`eval_hard_similarity.py `
### For the transitive sentence similarity task, run this script:  
`eval_transitive_sentence_similarity.py`  
### For the script event prediction task:  
see this [repository](https://github.com/MagiaSN/ConstructingNEEG_IJCAI_2018).  
### For Running the ROC dataset:
1.Download the [reverb](https://github.com/knowitall/reverb) tool which is used to extract events from sentences , you need to use this event extraction tools which will require Java as building tools.  
2.Using this tool to process the ROC dataset of each yaer,which will give you the extracted events for each sentence.  
3.Running `process.py`,`justify.py`,`reindex.py`,`reform.py`,`selectcloze.py` to collect the data.  
4.To run next step, you still need the transfomer package which is the `huggingface transformer`, install it following [this](https://huggingface.co/).  
5.Run the `train_bert_with_event.py` and `train_bert_without_event.py` to have the results of different models.  




