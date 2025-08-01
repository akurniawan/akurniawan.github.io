---
title: "Indonesian Language Model"
date: 2018-12-11T00:00:00+07:00
draft: false
tags: ["advanced-nlp", "language-model", "representation-learning"]
categories: ["nlp", "research"]
description: "Building Indonesian language models using AWD-LSTM for natural language processing applications at Traveloka"
summary: "A deep dive into building state-of-the-art Indonesian language models using AWD-LSTM architecture. This post covers the challenges of working with low-resource languages, implementation details, and experimental results from production systems at Traveloka."
---

# Lingua

NLP (Natural Language Processing) has been proven useful for many industrial practitioners to gain insight and automate human-intensive labor in order to bring a better experience for their customers. Chatbot to quickly reply customers' inquiries and free text search engine to help customers express their intent towards our product in more flexible way are a few examples of the use cases.

When dealing with text data, the representation of the text itself is one of the central components to build NLP applications. Recent state-of-the-art of text representations have been powered by word2vec and its variants that were first popularised by [Mikolov et al. 2013](https://arxiv.org/abs/1301.3781). However, as an Indonesian-based technology company, Traveloka deals with a high volume of Indonesian text. Due to the fact that Indonesian is considered as one of the low-resource languages, not much work has been done in text representation for this language.

## Language Model

It is tempting for us to build word2vec embedding from our own corpus that later can be used by the whole company. However, recent study from [Wendlandt et al.2018](https://arxiv.org/pdf/1804.09692.pdf) and [Antoniak et al.2018](https://mimno.infosci.cornell.edu/papers/antoniak-stability.pdf) show that word dense representations may suffer from instabilities measured by their closeness of their neighbours.

Following [Howard, J and Ruder, S., 2018](https://arxiv.org/abs/1801.06146), we chose [AWD LSTM](https://arxiv.org/abs/1708.02182) as the current state-of-the-art in language model for this experiment.

![AWD LSTM Architecture](/images/posts/language-model/awd-arch.png)

## Final Thoughts

This experiment demonstrates the effectiveness of AWD-LSTM for Indonesian language modeling, providing a solid foundation for downstream NLP tasks in Indonesian text processing.