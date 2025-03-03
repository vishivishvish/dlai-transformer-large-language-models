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

<img src="https://drive.google.com/uc?export=view&id=1ZmjoRqY_5a4sky0U_xmURntjnej_LdpC">

- Dense vector embeddings, however, are more nuanced than the above simple vector representations of 1s and 0s, and cannot be intuited so easily - they rather have values between 0 and 1. We will look into these Vector Embeddings in the next section.

## ***3 - Understanding Language Models: (Word) Embeddings***

- In this section, let us explore how these complex vector embeddings are created. 
- Although Bag-of-Words is simple and easy to understand, it has a major flaw - it does not consider the semantic nature of words / text, and it considers language to be nothing more than a literal bag of words. 
- Word2Vec, released in 2013 by Google, was one of the first successful attempts at capturing the meaning of text in embeddings. To do so, Word2Vec learned semantic representations of words by training on vast amounts of text data, such as the entirety of Wikipedia. To do this, Word2Vec leverages Neural Networks.
- We start by assigning every word in the vocabulary a vector (of say 5 values), initialized with random values. 
- Then, we take any pair of words from the training data, and a supervised objective is given to the Neural Network, where for example, it has to predict whether the two words are neighbors in a sentence or not.

<img src="https://drive.google.com/uc?export=view&id=1d879aylRVlTIHwWxMZGayzV5yqMrdZ_Y">

- In this training process, Word2Vec learns the relationship between words, and distills that information into the embedding for that word.
- If two words tend to have the same neighbor, their embeddings will be closer to each other, and vice-versa.
- The resulting embeddings after this training process, tend to capture some amount of semantic meaning in words.

<img>

- Each vector embedding has a certain fixed number of values in the vector (between -1 and 1), this number of values is called the dimensionality (or number of dimensions) of the vector.
- We can interpret each of these values as the value that word scores on a particular property of that object - for example, if the first property captures whether the object is an animal or not, the word “cats” may have a high score (0.91) for that value in its embedding.
- The number of dimensions in these vector embeddings is quite large, to give enough scope to capture semantic nuance of the word - it’s common to see dimension sizes in the thousands (ex: 1024 or 2048).
- In practice, of course, we do not actually know what properties each value of the vector embedding actually represents. What we can do is, plot the vector embeddings of different words together, and we should be able to see that similar words must appear closer to each other in vector embedding space than they do to other words.

<img>

- Although we only covered Word Embeddings, there are actually many types of embedding models we can use.
- When we talk about a model like Word2Vec, that converts textual input to embeddings, we refer to it as a representation model, as it attempts to represent text as values.
- Now, there are Tokenizers available that have a fixed vocabulary - these don’t convert every word into its own token, but rather, sometimes split apart a single word (such as “vocalization” into two or more tokens, such as “vocal” and “ization”)

<img>

- When we work with an input sentence such as “Her vocalization was melodic”, it is first tokenized by a Tokenizer, and the individual tokens are all passed to a Representation Model. This then converts each token into its corresponding Vector Embedding. To get Word Embeddings, we just have to ensure that for those multiple tokens which belong to the same word (such as “vocal” and “ization”), the embeddings of these two tokens are averaged out to get the Word Embedding for “vocalization”.
- In the same way, we can average out Vector Embeddings for all the tokens in a sentence to get Sentence Embeddings, and similarly for a document to get Document Embeddings.
H- owever, the above method is still static in nature, and doesn’t take into account the contextualized & dynamic nature of how language is actually used. We shall look into contextualized representations in the next section.

## ***4 - Understanding Language Models: Encoding and Decoding Context with Attention***
