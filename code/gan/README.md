# Data Preparation

1. Download [glove.840B.300d.txt](https://nlp.stanford.edu/projects/glove/) to folder `data_embeddings`

2. Convert json format data to txt format data

```
./dataConvertDistractor.sh
```

# Train and Test
```
./train_distractor.sh
```
```
./test_distractor.sh
```

# Credits
We thank the Wang et al. 2017 for their [IRGAN implementation](https://github.com/geek-ai/irgan).
