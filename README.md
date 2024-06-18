<p align="center">
 <br>
 <img src="assets/logo_tax.png" style="height: 80px;">
 <h2 align="center">Beyond Boundaries: Learning a Universal Entity Taxonomy <br> across Datasets and Languages for Open Named Entity Recognition <br> (B<sup>2</sup>NER) </h2>
</p>

<p align="center">
 <a href="https://github.com/UmeanNever/B2NER/blob/main/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/UmeanNever/B2NER"></a>
 <a href="http://arxiv.org/abs/2406.11192"><img alt="Paper" src="https://img.shields.io/badge/📖-Paper-orange"></a>
 <a href="https://drive.google.com/file/d/11Wt4RU48i06OruRca2q_MsgpylzNDdjN/view"><img alt="Blog" src="https://img.shields.io/badge/📀-Data-blue"></a>
</p>

We present B2NERD, a cohesive and efficient dataset that can improve LLMs' generalization on the challenging Open NER task, refined from 54 existing English or Chinese datasets. 
Our B2NER models, trained on B2NERD, outperform GPT-4 by 6.8-12.0 F1 points and surpass previous methods in 3 out-of-domain benchmarks across 15 datasets and 6 languages.

 - 📖 Paper: [Beyond Boundaries: Learning a Universal Entity Taxonomy across Datasets and Languages for Open Named Entity Recognition](http://arxiv.org/abs/2406.11192)
 - 📀 Data: See below data section.

# Release
 - **[ETA: June 30]** We plan to release our codes and models supporting the training and inference of our B2NER models.
 - **[June 18]** We release our papar and data. Our B2NERD dataset is highly suitable for training out-of-domain / zero-shot NER models.

# Data
One of the paper's core contribution is the construction of B2NERD dataset. It's a cohesive and efficient collection refined from 54 English and Chinese datasets and designed for Open NER model training.  
We provide 3 versions of our dataset.  
 - `B2NERD`: Contain ~52k samples from 54 Chinese or English datasets. This is the final version of our dataset suitable for out-of-domain / zero-shot NER model training. It has standardized entity definitions and pruned diverse data.   
 - `B2NERD_all`: Contain ~1.4M samples from 54 datasets. The full-data version of our dataset suitable for in-domain supervised evaluation. It has standardized entity definitions but does not go through any data selection or pruning.  
 - `B2NERD_raw`: Raw collected datasets with raw entity labels. It goes through basic format preprocessing but without further standardization.

You can download the data from [Here](https://drive.google.com/file/d/11Wt4RU48i06OruRca2q_MsgpylzNDdjN/view?usp=drive_link).  
Please also make sure you have got proper license to access the raw datasets in our collection.

# Model & Code Usage 
On the way. ETA June 30th

# Cite
