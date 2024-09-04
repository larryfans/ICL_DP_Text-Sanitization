# ICL_DP_Text-Sanitization

## Introduction
This repo contains source code for "In-Context Learning with Differentially Private Text Sanitization in Large Language Models" (accepted to the Findings of DSPP 2024 )

With the increasing popularity of cloud-based large language models, users often inadvertently input text containing personal information during interactions, leading to significant privacy concerns. To address this challenge, we propose an in-context learning(ICL) based on differential privacy(DP) to protect users' instances and context information formally. The core idea entails enhancing local differentially private text sanitization and using token mapping relationships to remap private responses effectively. We conduct the experimental evaluation by comparing our method with Custext, two-shot, and zero-shot. The findings indicate that our method can attain a competitive edge while maintaining robust privacy protections.

# Set environment
1. In models.py, set you openai key:
```
openai.api_key = '/api_key/'
openai.base_url = "/base_url/"
```

2. Install dependency
```shell
pip install -r requirements.txt
```

The first time of using stopwords from the NLTK package, you need to execute the following code in python
```
nltk.download('stopwords')
```
# How to run our code

```shell
bash run.sh
```



