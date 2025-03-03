# **Transformer Large Language Models**

**Deeplearning.ai**

**Andrew Ng - CEO, Deeplearning.ai**

**Jay Alammar - Director & Engg Fellow, Cohere**

**Maarten Grootendorst - Senior Clinical Data Scientist, IKNL**

**Sunday, 02/23/2025**

## ***1 - Introduction***

- In this course, we will learn about the main components of the LLM Transformer architecture.
- The Transformer architecture was, of course, introduced in the 2017 paper - Attention is All You Need, originally for Machine Translation tasks (for ex: English input to German output).
- But this same architecture turned out to be great for inputting a prompt (like an instruction) and outputting a response to that prompt / instruction, like a Q&A. 
- The original Transformer architecture consisted of 2 parts - an Encoder and a Decoder.

<img src="https://drive.google.com/uc?export=view&id=17RQl0vwEDuiC8ytLFWMIyHwYhi8-JxNE">

- The original English text would be passed to the Encoder, it would generate a certain context, and this context would be utilized by the Decoder to generate the German text. The Encoder and the Decoder form the basis of many models used today.
- The Encoder converts input text into rich, context-sensitive representations. Encoders are the basis of the BERT model and most embedding models used today in LLM RAG applications.
- The Decoder on the other hand, is used for Text Generation tasks, such as Text Summarization, Code Writing and Question Answering. It is the basis for most of today’s popular LLMs from orgs like OpenAI, Anthropic, xAI, Meta, Mistral & Cohere.
- In this course, we will dive into recent developments in LLMs, and understand how a sequence of increasingly sophisticated building blocks eventually led to the Transformer. We will then learn about Tokenization, which breaks down text into individual tokens that can then be fed to  Transformers. This will be followed by intuition about how transformers work, focusing on Decoder-only models, which take in a text prompt and generate text one token at a time.
- The model starts by mapping each input token into an embedding vector that captures the meaning of that token. After that, the model parses these token embeddings through a stack of Transformer blocks, where each block is a specific Neural Network architecture, designed to learn flexibly and also scale well in parallel across multiple GPUs.  
- Each Transformer block consists of a Self-Attention module and a Feedforward Neural Network.
- The model then uses the output vectors of the Transformer blocks and passes them to the last component, the Language Modeling head, which generates the output token.
- The magic of the Transformer experience in LLMs comes from two parts - the Transformer architecture as well as the rich datasets that LLMs learn from. However, it’s still important to have a solid intuition about what the Transformer architecture is doing, so that you develop intuitions about why these models behave the way they do, as well as how to use them.

## ***2 - Understanding Language Models: Language as a Bag-of-Words***

- In the next few videos, we will learn about the evolution of how language has been represented numerically.
- We’ll start with Bag-of-Words, an algorithm that represents words as large sparse vectors, or arrays of numbers, which simply record the presence of words.
- Then we’ll move to Word2Vec, whose vectors capture the meaning of words in the context of a few neighboring words.
- Finally we’ll move to Transformers, whose dense vectors capture the meaning of words in the context of a sentence or a paragraph.
- Although Bag-of-Words and Word2Vec lack contextualized representations, they are a good baseline to start with.
- Encoder-based Transformer models (starting from BERT, DistilBERT and RoBERTA), have typically been very good at converting language into numerical representations.
- In contrast, Decoder-based Transformer models are generative in nature, and their main objective is to generate high-quality text.
- We also have Encoder-Decoder Transformer models, such as T5, Switch and Flan-T5, which attempt to get the best of both worlds.

<img src="https://drive.google.com/uc?export=view&id=19YL6Lca_EA3Vw_QIzPAaAtmrSaypQa9f">

- Language is a tricky concept for computers. Text is unstructured, and loses its meaning when it’s represented by 1s and 0s.
- As a result, in the history of language AI, there has been a focus on representing language in a structured manner, so that it can be more easily used by computers.
- Some of the tasks that have attempted to bring this structure are Text Generation, Text Embedding Creation and Text Classification.
- The first method was by representing language as a Bag of Words.






