# Online speaker recognition and clustering


This study focuses on multi-enrollment speaker recognition with varying number of enrollment utterances. This task naturally occurs in the task of online speaker clustering, where speech segments arrive sequentially, and the speaker recognition system has to identify previously encountered speakers and detect new speakers.

## Data

Pre-computed embeddings: 

* [VoxCeleb](https://drive.google.com/drive/folders/1BX6_IO_-trAIiDYDrcLgOx-zYFXiYiZ3) cropped to 2 seconds; should be located in "./cache/voxceleb".
* [Datasets](https://drive.google.com/drive/folders/1GisDvp8LpMygELNqoFj9IwvH8msLSXIi) for speaker diarization; should be located in "./cache/diarization".


## Embeddings extractors

We consider several embeddings extractors to ensure generalization of the results across different embedding spaces

* [CLOVA](http://www.robots.ox.ac.uk/~joon/data/baseline_v2_ap.model)
* [SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)
* [BUT](https://github.com/BUTSpeechFIT/VBx/tree/master/VBx/models)


Speaker verification evaluation results on the commonly adopted [VoxCeleb1-O (cleaned)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt) protocol:

|  Embeddings | EER, % |
|:-----------:|:------:|
| CLOVA       |  1.19  |
| SpeechBrain |  0.90  |
| BUT         |  0.65  |


# Many-to-many verification

First, we compared different scoring backends for multi-enrollment speaker recognition. We considered the general case of *set-to-set* comparisons where trials may have arbitrary *and* varying number of enrollment and test segments.

We created several custom evaluation protocols from the VoxCeleb1 test set. 
Specifically, we generated four trial lists with configurations (1, 1), (3, 1), (10, 1), and (3, 3), where the notation (#enrollments, #tests) represents the number of enrollment or test segments in a single trial. 
In addition, we combined all the trial lists to get the [pooled](data/meta/voxceleb/multienroll) protocol. The idea behind it is to reveal the robustness of scoring back-ends to the number of enrollment segments. 
To exclude the effect of utterance duration, the recordings were cropped to 2 seconds before extracting embeddings.

Four different scoring backends are compared:
* CSEA - cosine similarity w/ embeddings averaging
* CSSA - cosine similarity w/ scores averaging
* sph-PLDA - spherical PLDA
* PSDA - probabilistic spherical discriminant analysis

Results for multi-enrollment speaker verification with three different embeddings extractors: CLOVA, SpeechBrain, BUT.
*Equal error rates* (EER, %) with *minDCF* ($P_\text{target}=0.01$) in the last column (lower is better).

CLOVA
|  Scoring | (1, 1) | (3, 1) | (10, 1) | (3, 3) |    pooled    |
|:--------:|:------:|:------:|:-------:|:------:|:------------:|
| CSEA     |  4.06  |  1.48  |   0.77  |  0.19  | 3.86 / 0.298 |
| CSSA     |  4.06  |  1.60  |   0.87  |  0.36  | 1.78 / 0.246 |
| sph-PLDA |  4.06  |  1.48  |   0.78  |  0.20  | 1.72 / 0.201 |
| PSDA     |  4.06  |  1.50  |   0.76  |  0.20  | 1.62 / 0.211 |

SpeechBrain
|  Scoring | (1, 1) | (3, 1) | (10, 1) | (3, 3) |    pooled    |
|:--------:|:------:|:------:|:-------:|:------:|:------------:|
| CSEA     |  4.98  |  1.65  |   0.83  |  0.17  | 2.85 / 0.206 |
| CSSA     |  4.98  |  1.79  |   1.02  |  0.37  | 2.05 / 0.228 |
| sph-PLDA |  4.98  |  1.60  |   0.78  |  0.14  | 1.99 / 0.170 |
| PSDA     |  4.85  |  1.55  |   0.78  |  0.13  | 2.08 / 0.172 |

BUT
|  Scoring | (1, 1) | (3, 1) | (10, 1) | (3, 3) |    pooled    |
|:--------:|:------:|:------:|:-------:|:------:|:------------:|
| CSEA     |  3.60  |  1.11  |   0.61  |  0.05  | 2.23 / 0.167 |
| CSSA     |  3.60  |  1.27  |   0.66  |  0.16  | 1.44 / 0.183 |
| sph-PLDA |  3.60  |  1.08  |   0.59  |  0.04  | 1.33 / 0.126 |
| PSDA     |  3.48  |  1.05  |   0.59  |  0.04  | 1.32 / 0.128 |

PSDA and sph-PLDA have lower of comparable error rates in all the tests.


# Online speaker diarization

Next, we compared several (online) clustering algorithms with different underlying scoring backends for the online speaker diarization task. These algorithms also include the proposed clustering algorithm which is based on variational Bayesian (VB) inference. Roughly, it can be seen as an online version of the [VBx](https://github.com/BUTSpeechFIT/VBx). See the [paper](https://arxiv.org/abs/2302.09523) for details.

For the evaluation metrics, we used the *diarization error rate* ([DER](https://github.com/nryant/dscore#diarization-error-rate)) and *Jaccard error rate* ([JER](https://github.com/nryant/dscore#jaccard-error-rate)).

We used two popular datasets of multi-speaker recordings: 
* [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) 
* [VoxConverse](https://mm.kaist.ac.kr/datasets/voxconverse/) (version 0.2).

The results below are reported for the oracle speech activity labels. Note that there might be an insignificant difference between the metrics computed with dscore and pyannote (presented below) packages.


## AMI corpus

We followed the experimental setup in [AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup)

Results for the ```dev``` and ```test``` splits with ```collar=0.25, skip_overlap=True``` 


#### CLOVA
| Back-end    | DER, % | JER,% | DER, % | JER,% |
|-------------|:------:|:-----:|:------:|:-----:|
| CSEA        |  3.30  | 23.23 |  4.23  | 25.50 |
| CSSA        |  5.68  | 26.08 |  5.88  | 27.58 |
| sph-PLDA    |  4.22  | 23.19 |  6.00  | 26.63 |
| PSDA        |  4.02  | 23.97 |  5.19  | 26.74 |
| VB sph-PLDA |  3.42  | 24.03 |  3.17  | 24.94 |
| VB PSDA     |  3.29  | 23.95 |  3.32  | 24.68 |

#### SpeechBrain
| Back-end    | DER, % | JER,% | DER, % | JER,% |
|-------------|:------:|:-----:|:------:|:-----:|
| CSEA        |  2.75  | 22.01 |  3.63  | 25.20 |
| CSSA        |  4.21  | 23.91 |  3.67  | 26.33 |
| sph-PLDA    |  4.91  | 25.40 |  6.32  | 28.33 |
| PSDA        |  3.57  | 22.96 |  4.68  | 26.10 |
| VB sph-PLDA |  2.73  | 22.76 |  3.32  | 25.21 |
| VB PSDA     |  2.59  | 22.61 |  3.34  | 24.47 |

#### BUT
| Back-end    | DER, % | JER,% | DER, % | JER,% |
|-------------|:------:|:-----:|:------:|:-----:|
| CSEA        |  3.23  | 22.19 |  4.04  | 24.88 |
| CSSA        |  3.91  | 23.69 |  4.96  | 26.05 |
| sph-PLDA    |  4.82  | 24.67 |  6.94  | 27.03 |
| PSDA        |  3.98  | 23.33 |  5.29  | 25.58 |
| VB sph-PLDA |  2.65  | 22.42 |  3.21  | 23.97 |
| VB PSDA     |  2.88  | 22.52 |  3.06  | 24.01 |



## VoxConverse

Results for the ```dev``` and ```test``` splits [ver0.2](https://github.com/joonson/voxconverse/tree/ver0.2) with ```collar=0.25, skip_overlap=True```


#### CLOVA
| Back-end    | DER, % | JER,% | DER, % | JER,% |
|-------------|:------:|:-----:|:------:|:-----:|
| CSEA        |  1.64  | 12.97 |  4.12  | 20.39 |
| CSSA        |  2.59  | 16.57 |  6.10  | 24.11 |
| sph-PLDA    |  1.79  | 13.92 |  4.14  | 20.92 |
| PSDA        |  2.00  | 12.89 |  4.45  | 20.76 |
| VB sph-PLDA |  1.76  | 14.07 |  3.60  | 21.94 |
| VB PSDA     |  1.62  | 13.21 |  3.94  | 22.76 |

#### SpeechBrain
| Back-end    | DER, % | JER,% | DER, % | JER,% |
|-------------|:------:|:-----:|:------:|:-----:|
| CSEA        |  1.84  | 12.91 |  3.60  | 20.01 |
| CSSA        |  2.98  | 17.34 |  5.50  | 24.67 |
| sph-PLDA    |  2.23  | 16.63 |  5.07  | 24.06 |
| PSDA        |  2.21  | 10.20 |  4.52  | 24.37 |
| VB sph-PLDA |  1.49  | 13.03 |  3.51  | 19.85 |
| VB PSDA     |  1.62  | 13.35 |  3.39  | 20.37 |

#### BUT
| Back-end    | DER, % | JER,% | DER, % | JER,% |
|-------------|:------:|:-----:|:------:|:-----:|
| CSEA        |  1.86  | 14.29 |  3.61  | 18.73 |
| CSSA        |  2.86  | 21.07 |  5.80  | 27.40 |
| sph-PLDA    |  2.27  | 18.98 |  4.81  | 23.31 |
| PSDA        |  2.08  | 18.35 |  4.43  | 24.13 |
| VB sph-PLDA |  1.50  | 12.97 |  2.95  | 19.02 |
| VB PSDA     |  1.46  | 13.27 |  3.10  | 19.64 |


The proposed online clustering algorithm (VB) demonstrates the lowest test DERs for both sph-PLDA and PSDA models.
