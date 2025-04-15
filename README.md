# **Transformer Large Language Models** 

**[Deeplearning.ai](https://www.deeplearning.ai/short-courses/how-transformer-llms-work/)**

**Andrew Ng - CEO, Deeplearning.ai**

**Jay Alammar - Director & Engg Fellow, Cohere**

**Maarten Grootendorst - Senior Clinical Data Scientist, IKNL**

***Notes by Vishnu Subramanian***

## ***1 - Introduction***

- In this course, we will learn about the main components of the LLM Transformer architecture.
- The Transformer architecture was, of course, introduced in the 2017 paper - Attention is All You Need, originally for Machine Translation tasks (for ex: English input to German output).
- But this same architecture turned out to be great for inputting a prompt (like an instruction) and outputting a response to that prompt / instruction, like a Q&A. 
- The original Transformer architecture consisted of 2 parts - an Encoder and a Decoder.

<img src="https://drive.google.com/uc?export=view&id=17RQl0vwEDuiC8ytLFWMIyHwYhi8-JxNE">

- The original English text would be passed to the Encoder, it would generate a certain context, and this context would be utilized by the Decoder to generate the German text. The Encoder and the Decoder form the basis of many models used today.
- The Encoder converts input text into rich, context-sensitive representations. Encoders are the basis of the BERT model and most embedding models used today in LLM RAG applications.
- The Decoder on the other hand, is used for Text Generation tasks, such as Text Summarization, Code Writing and Question Answering. It is the basis for most of today’s popular LLMs from orgs like OpenAI, Google, Anthropic, xAI, Meta, Mistral and DeepSeek.
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
- In contrast, Decoder-based Transformer models are generative in nature (which goes one step further than just converting language into numerical representations, as these representations then need to be converted back into new text). The main objective of generative Decoder-based Transformer models is to go that one step further and generate high-quality text.
- We also have Encoder-Decoder Transformer models, such as T5, Switch and Flan-T5, which attempt to get the best of both worlds, but which are currently not very popular (as of early 2025).

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

<img src="https://drive.google.com/uc?export=view&id=1SmHim-t4UN0UA6TujSNhuzGkT5qIPePM">

- Each vector embedding has a certain fixed number of values in the vector (between -1 and 1), this number of values is called the dimensionality (or number of dimensions) of the vector.
- We can interpret each of these values as the value that word scores on a particular property of that object - for example, if the first property captures whether the object is an animal or not, the word “cats” may have a high score (0.91) for that value in its embedding.
- The number of dimensions in these vector embeddings is quite large, to give enough scope to capture semantic nuance of the word - it’s common to see dimension sizes in the thousands (ex: 1024 or 2048).
- In practice, of course, we do not actually know what properties each value of the vector embedding actually represents. What we can do is, plot the vector embeddings of different words together, and we should be able to see that similar words must appear closer to each other in vector embedding space than they do to other words.

<img src="https://drive.google.com/uc?export=view&id=1EkqUekBMdlQZOjdxE0mhmveG3D1NiIGp">

- Although we only covered Word Embeddings, there are actually many types of embedding models we can use.
- When we talk about a model like Word2Vec, that converts textual input to embeddings, we refer to it as a representation model, as it attempts to represent text as values.
- Now, there are Tokenizers available that have a fixed vocabulary - these don’t convert every word into its own token, but rather, sometimes split apart a single word (such as “vocalization” into two or more tokens, such as “vocal” and “ization”)

<img src="https://drive.google.com/uc?export=view&id=19uTtb_8D26Odp7ltwJHPc-auXAHW2y78">

- When we work with an input sentence such as “Her vocalization was melodic”, it is first tokenized by a Tokenizer, and the individual tokens are all passed to a Representation Model. This then converts each token into its corresponding Vector Embedding. To get Word Embeddings, we just have to ensure that for those multiple tokens which belong to the same word (such as “vocal” and “ization”), the embeddings of these two tokens are averaged out to get the Word Embedding for “vocalization”.
- In the same way, we can average out Vector Embeddings for all the tokens in a sentence to get Sentence Embeddings, and similarly for a document to get Document Embeddings.
- However, the above method is still static in nature, and doesn’t take into account the contextualized & dynamic nature of how language is actually used. We shall look into contextualized representations in the next section.

## ***4 - Understanding Language Models: Encoding and Decoding Context with Attention***

- In the previous section, we explored Word & Token embedding techniques, but with limited contextual capabilities.
- In this section, we will explore how we can encode and decode context with attention.
- Word2Vec and other such methods create static embeddings - the same embedding is generated for “apple” whether it appears in “Eat an apple every day” or “Apple launched a new iPhone yesterday” - which is obviously not ideal since "apple" refers to 2 different things above. 
- For such words (and for all words really), their embeddings should have scope to change depending on the context (and the words they are surrounded by). Capturing this text context is in fact critical to perform certain tasks, such as Language Translation or Part-of-Speech Tagging.
- The first kind of Neural Network architecture that was used to encode text in this manner was the Recurrent Neural Network (RNN) model. RNNs can be used to model sequences such as text.
- To do this, typically RNNs were used to perform two tasks for text-to-text objectives such as Machine Translation - Encoding the input language text into representations, and Decoding the representations back into text in the output language. The Encoder would convert the entire sentence into an embedding, and the embedding would be used by the Decoder to generate the sentence in the output language.

<img src="https://drive.google.com/uc?export=view&id=1Nto1QN9VQV42B5dKgc3-IAgK9mKLF8WH">

- Each step in this architecture is autoregressive - when generating the next words, the architecture needs to consume all previously generated words.

<img src="https://drive.google.com/uc?export=view&id=1nCCZomdCTtZId6VlryFVcVrsUL5LazJ6">

- The autoregressive nature of the generation means the models generate one token at a time, dependent on the input tokens and all previously generated tokens.
- To recap the Encoding-Decoding process, we first start with the input sentence, tokenized into tokens.
- An embedding model like Word2Vec will be used to embed those tokens into embeddings, which will be the actual inputs to the model.
- Although each of these embeddings is static by itself, the Encoder processes the entire input sequence, including all the embeddings, and takes into account the context of the embeddings.
- The Encoder aims to represent the input as well as possible, and generates the context in the form of an embedding (a context embedding).
- The Decoder leverages this context embedding and uses that to finally generate the outputs.

<img src="https://drive.google.com/uc?export=view&id=1epHN4P9q6v3anv1xNPIK6SOEvsp9Uq0m">

- This context embedding however, makes it difficult to deal with longer sequences, since it is merely a single embedding representing the entire input. So this single embedding might fail to capture the entire context of a long, complicated sequence.
- In 2014, the idea of Attention was introduced, which was a big improvement over the original architecture.
- Attention allows the model to focus on those parts of the input sequence which are relevant / attend to one another and amplify their signal. Attention determines which words are most important in a given sentence.

<img src="https://drive.google.com/uc?export=view&id=1QV2uUeBIkMcTeIR3EBuaeHLCco8_RJXC">

- By adding these Attention mechanisms to the Decoder step, the RNN can generate signals for each input word in the sequence related to the potential outputs.
- In this Attention Encoder-Decoder process, we would again pass the Word2Vec embeddings of each input token to the Encoder, but from the Encoder, we would pass the hidden states of each input token now to the Decoder, as opposed to before when we passed only a single context embedding to the Decoder.
- A stateful word is an internal vector from a hidden layer of an RNN that contains information about previous words.
- The Decoder then uses the Attention mechanism to look at the entire sequence, and then generates the final language.
- Due to this Attention mechanism, the output tends to be much better since you now look at the entire sequence using embeddings for each token or words instead of the smaller, more limited Context Embedding.

<img src="https://drive.google.com/uc?export=view&id=1kQUrGv7a6fHGudDG8PM7HwdV6dtkXFp2">

- The sequential nature of this architecture precludes parallelization during training of the model.

## ***5 - Understanding Language Models: Transformers***

- In the previous section, we explored the main idea behind Attention.
- In this section, we will understand how this technique was further developed and as of Feb’25, still powers modern-day Large Language Models.
- The true power of Attention, in the way that it currently drives the abilities of LLMs, was explored in the paper “Attention is All You Need” in 2017. This paper introduced the “Transformer” architecture, a new architecture which was solely based on Attention and Feedforward Neural Networks, and did not require Recurrent Neural Networks.
- This architecture also had the additional advantage that it could be trained in a parallel manner, which enabled a significant speedup of calculations compared to RNN-based models, that precluded parallelization.

<img src="https://drive.google.com/uc?export=view&id=1BdKkB4_HTHw1SXfYkGwZ-wkLd-H_XL6H">

- Let’s look at a high-level picture of how the Transformer works.
- Assume you have the same input and output sequences as before.
- The Transformer consists of stacked Encoder and Decoder blocks. These blocks have the same attention mechanism that we have previously seen. It’s just that by stacking these blocks, we amplify the strength of the Encoders and Decoders.
- Let’s take a closer look at the Encoder.
- The input “I love llamas” is converted into embeddings, but instead of Word2Vec, we can just initialize these embeddings with random values, because of the power of the eventual training process and the quality & quantity of training data in finding the right values for the embeddings.
- Then, Self-Attention, which is Attention focused only on the input, processes these embeddings and updates them. These updated embeddings contain more contextualized information as a result of the Attention mechanism. They are passed to a Feedforward Neural Network, to finalize the contextual token word embeddings. 
- All this is of course, only part of the Encoder, whose job is to find good representations for input word embeddings.

<img src="https://drive.google.com/uc?export=view&id=1ubiSfQz67LSfiMRn-siX9kXoG7vG6_JH">

- Self-Attention is an Attention mechanism, and instead of processing two separate sequences (input and output sequences), it processes only one sequence (only the input sequence), by comparing it to itself.
- Now, after the Encoders are done processing the information, the next step is for the Decoder.

<img src="https://drive.google.com/uc?export=view&id=1TdfLYVFAFgbfu_VpTUCFZSbaagZUF4TJ">

- The Decoder can take any previously generated words, and pass it to the masked self-attention, similar to the Encoder to process these embeddings.
- Intermediate embeddings are generated and passed to another attention network together with the embeddings of the encoder, thus processing both what has been generated and what you already have. 
- This output is passed to a Feedforward Neural Network, and finally generates the next word in the sequence. 

<img src="https://drive.google.com/uc?export=view&id=1PFvmYPT3xQSSpbm5VJLRbOFhb4gVvDYT">

- Masked self-attention is similar to self-attention, but removes all values from the upper diagonal.
- Therefore it masks future positions, so that any given token can only attend to tokens that came before it.
- The original Transformer architecture is an Encoder-Decoder architecture, which was suited towards Translation tasks, which was its original purpose, but cannot be used easily for other purposes such as Text Classification.
- In 2018, a new architecture called Bidirectional Encoder Representations from Transformers (BERT) was introduced that could be leveraged for a wide variety of Encoder-oriented tasks, such as Classification and Semantic Search.
- BERT was an Encoder-only architecture that focused on representing language well by generating contextual word embeddings.

<img src="https://drive.google.com/uc?export=view&id=144FWX_s8xNuA_BYr1tLCk0yTX-jeylzZ">

- These Transformer Encoder blocks are the same as what we saw before. Self-Attention blocks, followed by Feedforward Neural Networks.
- The input also contains a CLS token - classification token, which is used as a representation for the entire input.
- Often, we use this CLS token as the input embedding for fine-tuning the model on specific tasks like Classification.
- To train a BERT-like model, we can use a technique called Masked Language Modeling (MLM).

<img src="https://drive.google.com/uc?export=view&id=1zt8iRhz0MQGtI-t6RZTcvhnToVEZ11LG">

- First randomly mask a number of words from your input sequence. Then the training objective is for the model to predict these masked words.
- By doing this, the model learns to predict language as it attempts to deconstruct the masked words. The process of learning to predict language teaches the BERT model how to represent language in a high-quality numerical representation (embedding).
- Training BERT-like models is typically a two-step process.

<img src="https://drive.google.com/uc?export=view&id=1OiDIBdadyINjLuLrWK8EZaE8Um7dmHjU">

- First, we need to Pre-train it using MLM on a large dataset (such as the whole of Wikipedia). Then we need to fine-tune it on a number of downstream tasks such as Classification, Named Entity Recognition, Paraphrase Identification.
- Autoregressive generative models, in contrast, use a slightly different architecture, where only the Decoder block is used, and the Encoder is not required.

<img src="https://drive.google.com/uc?export=view&id=1h7WK9CxIGup6KvvjaiHxR529y9r8Jf9V">

- Assume that you have an input sequence “I love” and randomly initialized embeddings. 
- These input embeddings are passed to a stack of Transformer Decoders. Each Decoder consists of Masked Self-Attention and a Feedforward Neural Network. Finally the next word is generated “llamas”.
- GPT-1 (Generative Pre-trained Transformer) from OpenAI was the first big implementation of this architecture, and scaling it up to GPT-2, GPT-3 and GPT-3.5 is what brought about the LLM revolution.
- Decoder-only Autoregressive Generative Models and Encoder-only Representation Models are the two main flavors of Generative AI currently seen in Language AI.

<img src="https://drive.google.com/uc?export=view&id=1Do2JvPGRLU-EUJ5gFt7rKVhTE1HThWTE">

- These two flavors of Generative AI do have a common limitation - the Context Length.
- We start from an input sequence, let’s say “Tell me something about Llamas.”, that we prompt the Generative model with in this instance. Let’s say the model has already started generating tokens in response - “Llamas are”. The original query plus the already-generated text equals the current context length (let’s say 8 tokens).

<img src="https://drive.google.com/uc?export=view&id=1etGzXfdAhbJlYiaw3eDuGuThR0xSNVnl">

- But any Generative LLM (such as OpenAI’s models) or even a representation model (such as OpenAI’s Ada models) can only have a maximum context length (such as 128k tokens). That means, the model can only process that many tokens at a given time.
- This context length also includes every new token that is being generated at any given point of time, as well as all the input tokens and previously generated tokens that led to this new token generation.

<img src="https://drive.google.com/uc?export=view&id=16--LJx-IcRwpf4bg4rhmkzkm3jP8Xsbr">

- In terms of the number of parameters, GPT-1 already had > 100 million parameters.
- GPT-2 was released in 2019 with around 1.5 billion parameters, and GPT-3 came out in 2020 with a massive 175 billion parameters.
- As the number of parameters grew, so did the capabilities of OpenAI’s models, with some new, emergent capabilities only seemingly unlocked at a certain order of magnitude of scale.

<img src="https://drive.google.com/uc?export=view&id=1Ekb7znVTzkOJcxl5g5L3VID1p94hB5WC">

- OpenAI’s release of GPT-3.5 as the ChatGPT product at the end of 2022 kicked off an LLM race, with plenty of both Proprietary and Open-source / Open-weight competitors releasing very capable LLMs.
- In the next lesson, we will learn about Tokenizers and Embeddings.

## ***6 - Tokenizers***

- In the previous lessons, we learned how word meaning was represented in vectors.
- We briefly introduced “tokens” as words or word pieces.
- In this lesson, we will illustrate what these tokens are, and how they help Transformers do their job.
- Imagine you have a given input sentence “Have the bards who”.
- For a language model to process that input text, it will first break down that text into smaller pieces.

<img src="https://drive.google.com/uc?export=view&id=1EU-zn_B2Z2Q1WXNhCuGMpanIE5CUDwTr">

- Each piece is called a token, and the process of breaking down the text is called Tokenization.
- Each token is then turned into a numerical representation, called an embedding. These are vector values that represent the semantic nature of a given text.
- These embeddings are static, and each embedding is created independently from all other embeddings and tokens.

<img src="https://drive.google.com/uc?export=view&id=18rX82y_ylrJ1vxeKEEWg4RkrBnHWqO5B">

- These embeddings are processed by a Large Language Model, and converted into Contextualized Embeddings. 
- These Contextualized Embeddings are still one for each input token, but they have been processed in such a way that all other tokens are also considered.

<img src="https://drive.google.com/uc?export=view&id=1eoSGRj9qkMfuQvb20WXrtahoAwMMVhsd">

- These embeddings can themselves be the output of a model (such as an Embedding Model like OpenAI’s Ada series), or they can be used by a model to then create outputs (such as for typical Autoregressive LLMs).
- In the case of generative models, this can be another token.
- Let’s explore in detail how the Tokenization process works.
- Given an input sentence, it’s tokenized or encoded into smaller pieces.
- Tokens can be entire words, or pieces of a word. 
- This process is necessary as tokenizers have a limited vocabulary, so whenever it encounters an unknown word, it can potentially still be represented by these sub-word tokens.

<img src="https://drive.google.com/uc?export=view&id=1DPy3q4HEbc5AkDZ-aRDnfi8V32aifMrg">

- Each token has an associated fixed ID to easily encode and decode the tokens.
- These are fed to the language model that internally creates the token embeddings.
- The output of a generative model would then be another token ID, which is decoded to output an actual token.

<img src="https://drive.google.com/uc?export=view&id=16giVUOV61QUlWt_IiAX4n_LZAlWKzihp">

- There are many different tokenization levels that we can explore.
- Let’s say we have a text sequence “Have the ♫ bards who preceded”.
- Words like “bards” and “preceded” might be represented by the tokenizer not as full words themselves (due to the relative rarity of their occurrence in the corpus), but rather as sub-word tokens, such as “b” + “ards” and “preced” + “ed”.
- Although it’s possible to go right down to the character level itself to do tokenization, that leads to extremely long sequence lengths, which means in practice, most modern LLM tokenizers work at the sub-word level.

<img src="https://drive.google.com/uc?export=view&id=13oF8IJIW3gLAkljxWME-qcqgH7d9gPp-">

- To explore different tokenizers, we first need to install the transformers package with:

`pip install transformers`

- The transformers library is used not only to interact with Tokenizers but also to use LLMs.
- First, we will need to import the AutoTokenizer sub-package from transformers, using:
 
`from transformers import AutoTokenizer;`

- This is used to interact with any tokenizer.
- The tokenizer that we will be using is called the “BERT-base-cased” model (which basically means this is a version of BERT Base which recognizes case - the difference between uppercase and lowercase. This is a pre-trained Encoder model from the BERT family, first introduced by Google.
- The tokenizer is loaded with the following command:
  
`tokenizer = AutoTokenizer.from_pretrained(‘bert-base-cased’);`

- Now, once the tokenizer is loaded, it can be used to process the input sentence and extract Token IDs like so:
  
`sentence = ‘Hello world!’;`

  ` token_ids = tokenizer(sentence).input_ids;`

- If you were to print these Token IDs, it would output a list of numerical values, like so:

`[101, 8667, 1362, 106, 102]`

- We would need to decode these Token IDs in order to get the actual tokens.
- There is a decode method in the tokenizer object that will allow us to loop through this list and decode the individual Token IDs:
  
`for token_id in token_ids:`

  ` print(tokenizer.decode(token_id));`
  
- This gives us:
  
`[CLS]`

`Hello`

`world`

`!`

`[SEP]`

- [CLS] and [SEP] are special tokens unique to the BERT tokenizer - they were arbitrarily chosen by BERT’s developers. [CLS] or the Classification Token represents the start of the entire input, and [SEP] or the Separation Token represents the end of a sentence.
- Visually, it’s good to be able to use color highlights on our input sentence to see the split of how the tokenizer converts the input into individual tokens - this can be done with some simple functions.
- First, let’s create a list of colors in RGB:
  
`colors = [‘102;194;165’, ‘252;141;98’, ‘141;160;203’, ‘231;138;195’, ‘166;216;84’, ‘255;217;47’];`

- These RGB colors will help us differentiate the individual tokens in the text sequence.
- Now, to print the list of colored tokens, this is the function we will create:

<img src="https://drive.google.com/uc?export=view&id=1Txr96dy1on67hYKPMNsy0almCo6a_gDJ">

- The section on printing the colored list of tokens is the crux of the matter.
- ‘idx’ and ‘t’ in the “for loop” means that we can loop over the lists of both the indices and the Token IDs.
- While decoding each Token ID, we’re using ANSI escape codes to color the foreground and background of each token. The 0 represents the normal text style, the 30 represents the foreground color (black), the background color type is 48;2; and the specific background color comes from cycling through the earlier list of colors by using the modulo % operator, which cycles through a list of remainders of diving by the length of the colors list.
- The piece of text we’ll run through the tokenizers is:

`text = """`

`English and CAPITALIZATION`

`🎵 鸟`

`show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "`

`12.0*50=600`

`""";`

- We can now run this text through the show_tokens() function and see how each tokenizer splits this text into tokens.
- For BERT-base-cased, we would do:

`show_tokens(text, ‘bert-base-cased’)`

<img src="https://drive.google.com/uc?export=view&id=1RXBsm9x4Sau1l74OOByZRklVme_I_NLD">

- We can see that the vocab size of this tokenizer is quite low, below 30k unique tokens.
- This is why it needs as many as 49 tokens to represent this input - tokenizer vocabulary size is inversely proportional to the likely number of tokens required to represent a piece of text.
- As we saw before, it starts with the [CLS] token and ends the sentence with [SEP].
- This tokenizer, as we see, has a lot of difficulty representing the word “CAPITALIZATION” - it has to split it into 8 different tokens.
- The hashtags at the start of some tokens simply represent that they, together with the token before, are part of the same word.
- The [UNK] token is the unknown token, and is used when the Tokenizer doesn’t know how to represent any token.
- Now let’s see how a more recent tokenizer differs from BERT-base-cased. Let’s use Xenova/gpt-4.

`show_tokens(text, ‘Xenova/gpt-4’)`

<img src="https://drive.google.com/uc?export=view&id=1J3uS8L4JLz55O2vBeGcp9b5_gzVitUGW">

- The GPT-4 Tokenizer has a vocabulary size more than 3 times as large as the BERT-base-cased Tokenizer.
- That means it should clearly need fewer tokens to represent the same string, and this is evident in the way the word “Capitalization” now needs only 2 tokens to represent it as opposed to 8.
- This Tokenizer also doesn’t need any [CLS] and [SEP] tokens, because it’s meant purely for a generative model in GPT-4.
- Because this tokenizer has a large vocabulary length, it’s easier to represent uncommon tokens, but there’s a tradeoff. 
- The larger the tokenizer vocabulary size, the more embeddings need to be calculated for each of the tokens. So there’s a tradeoff between choosing a tokenizer with a million tokens, versus learning the representations for each of those tokens.
- In the next section, we will see how LLMs process tokens in order to generate their output text.

## ***7 - Architectural Overview***

- So far, we’ve learned how language is represented numerically, and how words are converted into tokens and dense embeddings.
- When we think about Transformer LLMs, we know there’s an input prompt to the model, and there’s some output text generated by the model.
- One of the first important intuitions to understand the transformer’s working is that it generates tokens one by one.

<img src="https://drive.google.com/uc?export=view&id=1JCHZgR5aW858cTTiXraVEy1D2LcnIDzm">

- The Transformer is made up of 3 major components - the Tokenizer, the Transformer Blocks & the LM Head.

<img src="https://drive.google.com/uc?export=view&id=1oJK84o_jnljMebjR-43sH1u8lnEGwmzr">

- The Tokenizer is the component that breaks down the text into multiple chunks.
- The output of the Tokenizer goes to a stack of Transformer Blocks - this is obviously where the vast majority of the computing power gets used. The operations of these Transformer Blocks in terms of how they process the chunks of text from the Tokenizer are also quite understandable to a degree.
- The output of these Transformer Blocks goes to a Neural Network called the Language Modeling Head, whose job it is to present the final output in the format desired from the application design.
- We know that the Tokenizer has a vocabulary that breaks down text into independent tokens.
- Let’s assume the Tokenizer has about 50,000 independent tokens in its vocabulary. 
- The model has associated Token Embeddings for each of these tokens, and these are substituted in the beginning once the model is processing its inputs.

<img src="https://drive.google.com/uc?export=view&id=1CU9C411MhkCqOlVLc8Ix-S48k0Jmmu3F">

- Since we’re looking at the high-level overview here, let’s look at the Language Modeling Head.

<img src="https://drive.google.com/uc?export=view&id=1b4POBvVRnGfguuEVgqbbQ4mIGvtSDKVU">

- At the end of the processing of a Language Model, what happens at the end is a kind of scoring or token probability calculation, based on all of the processing that the stack of transformers has done (to make sense of the input context, what is requested in the prompt, and the internal weights of the model), and hence what the next token should be in response to that.
- So the result of the Language Modeling head is this sort of token probability scoring which says, for all the tokens I know, this is how much percent probability each of these tokens has, and all of these probabilities have to add up to 100%.
- So if you have the word “Dear” scoring a 40% probability and becoming the highest probability token, that could possibly be used as your output. Not necessarily, but possibly, since there are also other more nuanced Decoding Strategies than just directly outputting the highest probability token.
- This strategy of only selecting the highest probability token is called “Greedy Decoding” - it is what happens when we set the temperature of the LLM to 0.
- But this is not the only method available.
- Other strategies, such as “Top P Decoding”, are what happen when we set temperature > 0. It adds an element of randomness to the Decoding, it may still select the highest probability token, but in a few cases, it might select the next highest probability tokens too. It definitely looks at the scoring, but it doesn’t always have to pick the Top 1.

<img src="https://drive.google.com/uc?export=view&id=12dUvSPqzZBiATfJBwS6ct8hFY8pZPOux">

- This is important to generate text that sounds human / natural - human text has an element of randomness and an aversion to using the same words repeatedly, and that is what we’re trying to mimic with Top P Decoding.
- This is also the reason why when we use the same prompt repeatedly, we can get multiple answers that use words quite differently from each other.
- Another important intuition about Transformers, which is why they operate better than previous ideas like RNNs, is that they process all of their input tokens in parallel, and this parallelization makes it time-efficient. So we can compute a long context on a lot of different GPUs at the same time.

<img src="https://drive.google.com/uc?export=view&id=1vPCdpe48EPgifHvBMdhUS8i2EddBavoB">

- The way to envision this is to think of multiple tracks flowing through this stack of Transformer blocks, and the number of tracks here is the context size of the model.
- So if a model has a context size of 16,000 tokens, it can process 16,000 tokens at the same time.
- The generated output token in Decoder-only Transformers, is the output of the final token in the model. 
- Remember that every time we generate a new token, that new token together with all earlier tokens within the context size, are used back to generate the next new tokens. Also, that means all the earlier calculations can be cached, since they would be exactly the same anyway, so we can cache them to speed up the generation of every new token
- This is referred to as “KV-Caching”.
- There’s an important metric used to gauge Transformer performance efficiency, called “Time to First Token”, which is how long the model takes to process all of this to generate the first token.
- Generating every other token is a slightly different process.

## ***8 - The Transformer Block***

- We have learned that after tokenization, the tokens are passed through a stack of Transformer blocks. Let's get into the details of those blocks.

<img src="https://drive.google.com/uc?export=view&id=1bOAaG89XwfJivgdE1c1vtRFxtaAOWy_0">

- Since the input words are “The Shawshank”, let’s think about the two tracks that are flowing through the stack of transformer blocks. 
- In the beginning, our tokenizer has broken down the prompt into these two tokens.
- “The” is its own token, and “Shawshank” is its own token.
- We have the associated vector embeddings for both these tokens - that is what we will actually be calculating on.
- Now that we have turned language into numbers, we can apply a lot of interesting calculus on them to try to predict the next word.
- The embeddings first flow to the Transformer Block 1 - this generates vectors of the same size as the embeddings as its output.
- But the Transformer Block 1 has performed some processing in the middle.
- Before we get into that though, it’s useful to understand the general flow of data through the model.
- The same thing happens with Transformer Block 2, which operates on the outputs of Transformer Block 1, in parallel across the multiple tracks.
- This happens down the list of Transformer Blocks all the way to the end.
- In the final layer, the vector for the final token in the prompt is presented to the Language Modeling head, which outputs or generates the next token.
- This is the flow - everything flows from the beginning to the end in one direction - from the tokenizer down the Transformer blocks, one-by-one in sequence up until the Language Modeling head.
- The Transformer block itself is made up of two major components - the Self-Attention Layer and the Feedforward Neural Network. 

<img src="https://drive.google.com/uc?export=view&id=1eXejTASUIxeYxxzZOMru3-cKQP5Pt0cn">

- We can think of the Feedforward Neural Network as a module for storage of information statistics about what words are likely to come next after the input token.
- This contrasts it with the Self-Attention module.

<img src="https://drive.google.com/uc?export=view&id=1ESXESGat-ZFInakCWWMHQjy3cjiuLqlW">

- A Feedforward Neural Network usually looks like this, with layers usually expanding before contracting, to give the network information bandwidth / high-dimensional space to expand its knowledge into, before it has to compress it down to the final output dimensions.
- That’s exactly what happens in the Feedforward Neural Network in the Transformer as well. The connections inside the Dense Model is presumably where all the information that models know is stored and modeled and interpreted and interpolated between to enable the models to do the incredible things they can, to generate code, encode information about the world, and speak to you in a fluent and coherent language of your choosing.

<img src="https://drive.google.com/uc?export=view&id=1OKQhdrLvCc4yA4pbVH_u_6ickJLgzhc1">

- Self-Attention on the other hand, allows models to attend to previous tokens, and incorporate the context in its understanding of the token it’s currently looking at.
- Let’s say we have a prompt: “The dog chased the llama because it” - when the model is processing these words, it has to bake in some information from these words to understand what “it” is referring to - does “it” refer to the dog or the llama?

<img src="https://drive.google.com/uc?export=view&id=1ne8dgEYAjQGD3v-K27ZJBCwYZzYJqu8L">

- It’s important for the model to understand what that “it” refers to, and that’s a little bit of what self-attention refers to.
- It enables the model to bake in some of that representation of the llama tokens, for example. If the previous context points towards “it” referring to the llama, self-attention allows the model to bake in some of that representation of the llama into it.
- This is an NLP task called “Coreference Resolution”, and if these are the only words presented to you, it would be difficult to discern if “it” refers to the dog or to the llama.
- But in this example, let’s assume that previous tokens in the prompt indicate that “it” is the llama.
- So to understand attention at a high level, let’s formulate it like this:

<img src="https://drive.google.com/uc?export=view&id=1oM7K_2tEhPG8V3edmwh7m6jSBvABpTuy">

- So we have other positions in the sequence - previous tokens that we’ve processed in the past.
- Then we have the current position that we’re processing.
- We have its current vector representation (current position information).
- As this current vector representation passes through the Self-Attention module, it gets enriched with context information from other positions - assuming we’re pulling in the right information from the other tracks, that is useful to represent this token in this step.
- This is a high-level formulation of what self-attention does, we’ll get into more of the specifics in the next lesson.
- But what Self-Attention does is two things.

<img src="https://drive.google.com/uc?export=view&id=1jRdOHce8iidDatNK7mG5eEXKQXzuRzXC">

- The first thing is Relevance Scoring - it assigns a score to how relevant each of the input tokens are to the token we’re currently processing.
- The second step is, after scoring that relevance, it combines the relevant information into the representation.
- So at a high level, this is what self-attention boils down to - Relevance Scoring and combining information from the relevant tokens into the current vector that we’re processing and representing.
- In the next section, we’ll look into further details of how that’s done.

## ***9 - Self-Attention***

- Self-Attention is a key component in the Transformer block.
- As we’ve learned, it consists of two steps - Relevance Scoring and Combining Information.
- We’ll now take a closer look at how these are calculated, and how that has evolved in recent years to enable more efficient attention.
- Self-Attention happens within what we call an Attention Head.
- Let’s assume we only have one of these Attention Heads right now, that we’re using to process Self-Attention.

<img src="https://drive.google.com/uc?export=view&id=15OVEF-HPlNlLJJhQMebcoUm2dsAkAeLQ">

- We have the token that we’re currently processing, and then we have the other positions in the sequence. These are the vector representations of the other tokens that precede this token.
- Self-Attention is conducted using three matrices - the Query Projection Matrix, the Key Projection Matrix and the Value Projection Matrix. 

<img src="https://drive.google.com/uc?export=view&id=13B01OLOilPjL6yYBoLk5XOu3Jad4HYuN">

- For Transformers, Keys, Queries and Values are important concepts.
- In this calculation, we will get a sense of what each of them is used for.
- These weight matrices are used to calculate query, key and value matrices.
- And through some interaction that we will discuss, we can go about scoring the relevance and then combining the information.

<img src="https://drive.google.com/uc?export=view&id=1g4Nq2Rv1LdDEh2lan-b8_A9DZZ_hmImt">

- Let’s say on the Query side, we have a vector representation that represents the current token / current position.
- And in the “Previous tokens” portion before the “Current token”, let’s say each row of that matrix represents a vector pertaining to one of the previous tokens before the current token.
- And the same thing is done with Keys and Values.

<img src="https://drive.google.com/uc?export=view&id=1HDpFYPK7TOrgg_kOOdbD-mjwQRfN7Xc4">

- The end goal of Relevance Scoring is something like the above figure.
- Every token we have is assigned a score showing how relevant the token is, to the token we’re currently representing.
- In this case, let’s say we have “The” and “dog” which have the highest Relevance Score. So more of those tokens will be baked into the enriched vector representation that contains the contextual information from other positions, which we see in the end of the above diagram.
- This is the end of the Relevance Scoring stage. We obtain scores for each token and they add up to 100%.

<img src="https://drive.google.com/uc?export=view&id=1BeT_U0NeRNcR1IaX5AEPaOwwWnDVwTqs">

- Technically how this is done, is by Matrix Multiplication.
- We multiply the Queries Vector associated with the current token with the Keys vectors that represent the previous tokens.
- This is a high-level intuition, but to know more about how Attention is calculated and implemented, the follow-up course from Deeplearning.ai and StatQuest on “Attention in Transformers with PyTorch”, by Josh Starmer and Andrew Ng, is entirely devoted to the calculation and implementation of Attention.

<img src="https://drive.google.com/uc?export=view&id=1lhuc5HJNk7A-KSGpGTbDbkrrKM3Bd3W0">

- And then, now that we have the Relevance Scores, we can start with the second step, which is, combining information from the relevant tokens, that is done using the Values vectors associated with each of these tokens. Each token has a Values vector associated with it.
- We just multiply the score of each token, by the Value vector.
- That gives us these Weighted Values, where “The” and “dog” have the highest value.
- The other values in the Weighted Values section will be closer to zero, because we’re multiplying them by smaller numbers.
- Once we have all the Weighted Values, we just sum them up.
- That is the output of this second Information Combination step.

<img src="https://drive.google.com/uc?export=view&id=1sulmYfie5uCXk0oZoxqIHNcAvPiEMIbw">

<img src="https://drive.google.com/uc?export=view&id=1bLgnJXC2lMhsCocRK3jX9tTRAuXit13M">

- As mentioned before, that calculation happens within an attention head.
- But in Self-Attention, that same operation happens in parallel, in multiple attention heads.
- Each attention head has its own set of Key, Query and Value Weight Matrices, so the attention that we assign to the various vectors is different.
- So we can think about two components of Self-Attention - splitting into various Attention Heads, and then combining the information from all the Attention Heads back together to form this output of the Self-Attention layer.
- It’s also important to visualize the Key, Query and Value matrices.
- Each one of the Attention Heads has its own set of Projection Matrices for Keys, Queries and Values.
- And now that we have this visual, we can talk about more recent forms of Attention, that power modern Transformer-based Large Language Models.

<img src="https://drive.google.com/uc?export=view&id=1F8OqzpplOfiFDbKTcapSDU6uIERYTOvx">

- To make self-attention more efficient, as this is one of the steps that takes the longest and takes the most computational time in Transformers, one idea that was proposed to make attention more efficient, was the idea of Multi-Query Attention (MQA), by Noam Shazeer (one of the original inventors of the Transformer) in 2019.
- The idea was that each Attention Head doesn’t need to have its own separate Keys and Values.
- Let’s have them all share the same Keys matrix & Values matrix.
- So we only have one for the entire layer - one set of these two projection matrices.
- We can think of this as some sort of compression - we have a smaller number of parameters now, and this helps models calculate Self-Attention faster during inference decoding. 
- So MQA helps improve the inference speed for Transformers, as it reduces the memory requirements of loading Key-Value Tensors during incremental decoding.
- However MQA’s drawback is that while it improves decoding efficiency, it can lead to quality degradation in the Transformer’s outputs.
- More recently, Grouped Query Attention (GQA), proposed by another team at Google in 2023, is another efficient Attention mechanism that allows us to use multiple Keys and Values, but not as large as the number of Attention Heads (n_attention_heads). This smaller number is what we refer to in the below diagram as n_groups. 

<img src="https://drive.google.com/uc?export=view&id=1ZTDzZ7XYV_Olc2ItGP6U9Hg89rGCTuNo">

- This leads to better results than when we just shared one set of Keys and Values across all Attention Heads.
- GQA was especially important for larger Language Models (post-2022), which required more of those parameters to represent the data that is required to really do Self-Attention on very large training datasets.
- Now when we read papers that describe the architecture of the model that was trained, if they use some form of GQA, they will mention not only the number of Attention Heads they used but also the number of Groups of Keys and Values they used.

<img src="https://drive.google.com/uc?export=view&id=1_UQpwMHQEVic_Mrnjj3_fDl8rz5yYOc4">

- Another important recent idea for improving the efficiency of attention is the idea of Sparse Attention.
- This usually does not happen in all the layers.
- Let’s say the first layer has Self-Attention, in the way that’s displayed in the diagram, where Token Number Seven, as we’re processing it, is able to attend to all of the tokens that preceded it (Global Autoregressive Self-Attention).
- In larger and larger models, that becomes a little bit too expensive, if you allow that to happen at every layer.
- So we start to see that maybe interleaved, like Layers 2, 4 and 6 for example, are not able to attend to all of the tokens in the history, but maybe only to the last 4 or 6 or 32 tokens.
- This idea is referred to as Sparse Attention.

<img src="https://drive.google.com/uc?export=view&id=1faolEhDXVQc2IGt6_VDxzpnueJO5dIJK">

- One way to think about it is - if we were to set up some visual language, let’s say we have the token “the” - we’re processing it now and it’s the first token, so we can’t really attend to any previous tokens.
- But let’s say the second token is “dog”. So “dog” can attend to both “the” and “dog”. 
- Then we have “chased” - the third token. That is able to attend to three tokens.
- Now, this is just to set up the visual language for figures like the below one, which describes what Sparse Attention looks like.

<img src="https://drive.google.com/uc?export=view&id=1ZH5QiK7ePYc9A5N_79MtAomezt-ABF1x">

- In full attention, every token can attend to every previous token.
- So each row can be thought of as a new token, and hence a step in processing.
- But Sparse Attention can be, for example, strided - where at each position, we look back at a fixed number of immediate previous tokens (let’s say 3 or 4), but we also look back at every 4th token before the earliest such immediately previous token. So for example, in the last row in the Strided Sparse Attention, when we’re processing the 16th Token, we attend to the 13th, 14th and 15th Tokens, but we also attend to Tokens 12, 8 and 4.
- Another variant is Fixed Sparse Attention, where for example after you reach every 4th token, you’re only allowed to attend to tokens from that token onwards in future processing steps, but you can still attend to all the 4th previous tokens as well. So for example, when processing Token 8, we can attend to Tokens 4-7. But when processing Token 9, we can only process Token 8 and Token 4. 
- So Fixed Sparse Attention is a combination of only attending to tokens from the fourth token onwards, in addition to allowing solely every fourth previous token as well.
- These Sparse Attention ideas were introduced by a team led by Alec Radford and Ilya Sutskever from OpenAI in 2019.
- More recently, to allow models to go through Long Context inputs of 100k or even 1 million+ tokens, ideas like Ring Attention have been introduced.
- The blog post at [Coconut-Mode](https://coconut-mode.com/posts/ring-attention/) is a good visual explanation for how Ring Attention works.

<img src="https://drive.google.com/uc?export=view&id=1Y8WS2vof-Wir1G3pAHhjU2EGJnNeD4EG">

- Now that we have made it this far, having seen a lot of visual language for representing ideas in Transformers, when we are reading a paper like the [Llama 3.1 Technical Report](https://arxiv.org/abs/2407.21783), we should be able to mentally picture the architecture choices made by the Meta team for their Llama 3.1 models.

<img src="https://drive.google.com/uc?export=view&id=1s13vvITTIDl3kmcOePRUCqtKqaDAp5-d">

- The 8B model, for example, has 32 layers. That basically refers to 32 Transformer blocks.

<img src="https://drive.google.com/uc?export=view&id=1DjZYUhwXvILpj1SM11CmwrRFUduL0XDd">

- It has a 4096 model dimension. That is the dimension length of the vector that flows through the Transformer.

<img src="https://drive.google.com/uc?export=view&id=1pxWx6Ufk5DXKq7pnoNkd9taRGMG_87VK">

- The Feedforward Neural Network has a dimension of 14,336. That refers to how many units there are in the middle layer of the Feedforward Neural Network.

<img src="https://drive.google.com/uc?export=view&id=1eSNCJmOueDwSiUQkiQHr3UXhnsTCquYm">

- The model has 32 Attention Heads, and 8 Key-Value Heads for the Grouped Query Attention.

<img src="https://drive.google.com/uc?export=view&id=19iBez63FqyzT0JUpWiolITcHbxg81K8t">

- The vocabulary size of 128,000 means that the model can output one out of 128k tokens at each generation step, based on the probability scores it generates for them.

<img src="https://drive.google.com/uc?export=view&id=1mV6XRR5FckdeWe39SqTCudZU1VhOHSxd">

## ***10 - Model Example***

- We’ve learned so far about all the different components of the Transformer.
- Now, it’s time for us to explore the architecture of one model example using the Hugging Face Transformers library.
- We will take a quick look at the code of how Hugging Face Transformers allows us to look at the Transformer and its Tokenizer and the process we’ve learnt about happens - how the processing and flow of information happens from the tokenizer to the stack of Transformer Decoders to the element. And we’ll look at how that happens in code.
- We’re going to be loading a language model - Phi-3-mini-4k-instruct, from Microsoft’s Phi-3 family.

<img src="https://drive.google.com/uc?export=view&id=1JR3TtzXy5leNlykkSmaEO91z9c1hTjuR">

- In this code, we will be downloading the model together with its tokenizer.
- When we load it on a CPU, we may get a couple of warnings, but those can be ignored.
- Now we can create a Hugging Face Pipeline, by specifying the tokenizer and the model we’ve downloaded. The “Pipeline” is just a crude abstraction that makes it easier to generate code within the LLM, after we give it up and load the model and tokenizer.
- In the code below, we’re saying that whenever we give the system a prompt, we want it to generate 50 tokens in response to that.
- The do_sample parameter being set to False, means that we’re doing Greedy Decoding.
- So with each token, it would score the probability of the output tokens, and it would simply choose the token with the highest probability - almost exactly like setting the temperature to zero.

<img src="https://drive.google.com/uc?export=view&id=1GZdTLGN1zIXVhSZqgtDuj_Ed7KHqRUyp">

- Now that we’ve done that, we can declare our prompt and pass it to the model.
- This is the prompt we’re giving to the model - “Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.”
- Let’s skip through to when it’s done generating, and talk about what it did in the process.
- Obviously, it’s only allowed to generate 50 tokens, so we will likely only see the start of this email.

<img src="https://drive.google.com/uc?export=view&id=18btWJo72SFavS2mML7P7LCjXVe9RTIeh">

- We can change the prompt to whatever we like, and see how the model responds.
- When the model is running on CPUs, it may take a few minutes to generate the response.
- This is why, for example, in the industry a lot of these models run on inference-optimized GPUs.
- A lot of the efficiency-oriented methods we’ve discussed are important in speeding up the text generation process - hence the focus on efficiency in research and in industry.
- Now that we’ve loaded the model in Hugging face, we can actually just print it out to try and understand its architecture.

<img src="https://drive.google.com/uc?export=view&id=1KaIZ9jySz0E0M0tmr1YSq2K-e5S_HP34">

- We can see that the output is indented, that shows us the hierarchy of the model.
- First we have the model, which is the Phi 3 model for Causal Language Modeling - this is housed in an instance of the Phi3ForCausalLM() object.
- The Causal or Autoregressive nature of the language model means that the attention only focuses on the previous tokens.
- Inside, we have two major components - the (model) itself, which is where all of the (layers) / the Transformer Blocks sit. 
- We have the Tokens - the (embed_tokens) - also referred to as the Tokens Matrix, which is a matrix whose rows are each unique token (the token vocabulary size of 32064 in this model), and the columns are the dimensions of the embedding of each token (3072 dimensions here).
- Inside (layers), we see that there are 32 Decoder Transformer layers, and we can see the exact component of each of them.
- We have Self-Attention - (self_attn), the projection, the Rotary Embeddings, the MLP (Multi-Layer Perceptron or Feedforward Neural Network), we can see that it projects the 3072-dimension embeddings up to 16,384 dimensions, and eventually projects it down to 3072 dimensions again.
- We use the SiLU activation function, which is x(1/(1+e-x)).
- Towards the end of the model, we see the Language Modeling head, which takes in the 3072-dimensional vector that’s come out from the end of the process, and projects it up to a 32,064-dimensional vector, which contains a probability score assigned to each of the 32,064 tokens in the vocabulary on their likelihood of being the next token.
- Hugging Face also makes it possible to do "model.model", so that allows us to go solely into the (model) part of the Phi3ForCausalLM object.
- We can use this to address all of the sub-objects in the same fashion, such as model.model.embed_tokens for example.
- We can access the first layer of the model by doing model.model.layers[0].
- As opposed to the Hugging Face Pipeline method, which abstracted away all the low-level details, let us now try to access the Token Generation process of the model in a more involved manner, where we get to see the mechanics of the information generation process.
- Let’s say our prompt is “The capital of France is”
- First, we need to tokenize the input prompt and get the IDs of the tokens. That is done by:

<img src="https://drive.google.com/uc?export=view&id=1ChbeU9I14QTwiMZiSj192MH4TZRS5yxX">

- This input_ids variable will now contain a Tensor object of the input IDs of the tokens in the prompt “The capital of France is”.
- This input_ids variable can be passed through the model and we can get the model output before the LM head, by using the same Hugging Face functions from before.

<img src="https://drive.google.com/uc?export=view&id=1ArhpEErrwEjM41a2dOBjle64UKSx2Ql4">

- Let’s take a look at how that model_output looks like:

<img src="https://drive.google.com/uc?export=view&id=1O7OVShhjH-dEe4B7zfwbcZabhDyY8uEL">

- On using the .shape attribute, we see that it’s a tensor with dimensions [1,5,3072] - 1 is the batch dimension, it’s 1 because we only passed the model one text input “The capital of France is” - in training, there’s a lot more batches. 
- 5 is the number of tokens in the sequence, as we know.
- 3072 is the dimension of the output vectors. 
- This can be visualized from the following diagram from earlier.

<img src="https://drive.google.com/uc?export=view&id=1VpXHsaXKAHDCPJ9nJvOpPatF1UCLNqUE">

- The Block N Output that we see before the LM Head is the 3072-dimension output vector being referred to earlier.
- There would be 5 such vectors of 3072 dimensions each.
- We can take this tensor of dimensions [1,5,3072], pass this to the Language Modeling Head, and see what the output looks like.

<img src="https://drive.google.com/uc?export=view&id=1nY3mcx8giCeKUQNDF5XzgxXa0tVgi2lw">

- We see that this is now a tensor of shape [1,5,32064], meaning that for each of the 5 tokens (rows of this new matrix), there are 32,064 probability prediction scores - 1 for each token in the vocabulary - on the likelihood of that token in the vocabulary being the next token in the sequence.
- But to get the actual output for the prompt we sent to the model, we are obviously only interested in the predictions for the last token of the input prompt - the 5th row in our above tensor matrix.
- So we address that particular output, and in the manner of Greedy Decoding, we select the highest probability generation token for that last input token in the following manner - getting the Token ID first, and then separately decoding that Token ID to give us the actual token prediction.

<img src="https://drive.google.com/uc?export=view&id=1-IUriEgGkK2AO4BwQGCg9w6iVteuXRoG">

- This is fascinating for a couple of reasons.
- In a sense, this is software that is able to now tell you information about the world.
- But also, there’s another thing here - the models never really saw the text. They only see the lists of Token IDs, and they output these Token IDs as well.
- The models never actually see the text, the level that we operate at, despite being so good at being able to predict the next token of text. So somehow, the models have understood a deeper intrinsic pattern of knowledge inside the text that allows them to operate flawlessly at the text level.
- In the next section, we’ll look at some recent improvements to LLMs.

## ***11 - Recent Improvements***

- Now that we know how Transformer LLMs work, let’s look at a couple of more recent ideas that are part of the latest generation of models.

<img src="https://drive.google.com/uc?export=view&id=1CiAnwMxFBm011fOzigJMs6z89hXZigg9">

- On the left is a simplified view of the Transformer.
- This is a Transformer Decoder - an architecture of the original one from 2017.
- The figure on the right (the diagram in the original Transformer paper from 2017) is a flipped version of the one on the left.
- Positional Encoding is a method of applying positional information. Otherwise, it would be presented to the model as a Bag-of-Words, there would be no order to the words. And the order of words matters a lot in a sequence.
- So Positional Encoding is one method - there are multiple methods of adding that positional information to the representation of vectors.
- The original Transformer model was an Encoder-Decoder model. 
- Most LLMs now in existence used to generate text are purely Decoder models, and do not have the Encoder component.

<img src="https://drive.google.com/uc?export=view&id=1-ftGDeg9s3yTyV7Cka1GGjaTRM3tNiTH">

- Encoder models are also in use, such as BERT and the models used for text embeddings and re-rankers, or other efficient, classification-oriented ways of doing NLP tasks that are not necessarily required to do Text Generation.
- We can compare and contrast the 2017 Transformer Block with the 2024 Transformer Block (such as those used in the Llama 3 Model Family), and we would see that they are quite close to each other, with only a few important differences.

<img src="https://drive.google.com/uc?export=view&id=1eb5cGazMyJvuA0ee6VOOodiW_5A8JOpp">

- One change is that there’s no longer a Positional Encoding step at the beginning of the processing of the model in the 2024 Transformer Block.
- We now use Rotary Embeddings - essentially positional information is now added at the Self-Attention level.
- The Layer Normalization has moved before the Self-Attention and before the Feedforward Neural Network stages in the 2024 Transformer block, when in contrast, the Add and Normalize steps were after these stages in the 2017 version.
- The rationale for these changes is that some experimental results showed the models performing better with this kind of setup.
- In addition, as we see, the 2024 Transformer Block uses Grouped Query Attention in place of vanilla Multi-head Self-Attention.
- One important thing to note, that was there in both the original Transformer architecture as well as the more modern version, is the presence of Residual Self-Connections, that repack the information from the beginning of the processing and add them to the representation at the end.
- Before we get into Rotary Embeddings, let’s first talk about the LLM Training process.

<img src="https://drive.google.com/uc?export=view&id=10WULaXkg9iywTcDmklI5S9nTeBXU3bJA">

- As we know, LLMs are trained in multiple steps.
- The first step is the Base Training, where it’s the next token prediction style generation, also called Language Modeling, which is of course why these models are called Language Modeling.
- When we visualize it in our head, we might think of training in batches as following the format in the first diagram above.
- Each row in the batch would contain one document. But if the model were to have a 16k context window, and Document 1 only takes up a small fraction of that space, so we would normally add padding to the rest of that context window.
- If we were doing it naïvely, this is one of the first ways we might think about doing training - we would have documents like this, where the majority of the context is just padding that’s not really used.
- This would be a very inefficient way to pack the training data - think of it like filling only 20% of a truck for each round trip from a warehouse to a retail store. 
- In reality, a more efficient way to pack these documents for training is like the second figure, where you have multiple short documents which you pack into one row of your batch, and you do that with each row. This would require less padding in each row, meaning we would definitely at least use most of the training compute available, since the GPU is going to be doing that crunching regardless of whether it’s an actual document or it’s just padding.
- This is a high-level side piece of information about how that training is done.
- We say that because this also has an impact on the architecture - specifically with the Positional Encoding.
- In the naïve method displayed in the first figure, we can say that for Token No. 1, I’ll always assign it with a positional vector that encodes that this is Position No. 1. And so on for Position No. 2, 3 and 4. 
- You can have multiple of these static Positional Encoding methods - they can either be learned, or they can be algorithmic in terms of having some combination of Sine and Cosine functions with time, so that the model learns that this kind of information in a vector means that it’s referring to Position No. 1 or 2 or 3 etc.
- That’s what’s called a Static Positional Encoding method.
- There are other methods that are more Dynamic in that they denote that this token is three tokens before another token, for example.
- In the second diagram, because there are multiple documents in the same row, the Self-Attention mechanism when it’s working on Document 2, should not be able to look at Document 1.
- But that’s not a Positional Encoding property.
- The Positional Encoding property that’s needed here is a way to allow it to say - okay, this is the first position of this document. 
- So in the counting of which token, which position we are in the context length, you support a way in which the model can make sense that - okay, this is Token No. 1 in Document No. 2.
- This is a bit of intuition around why a lot of the more recent models use Positional Encoding methods like Rotary Position Embeddings (RoPE).

<img src="https://drive.google.com/uc?export=view&id=10XRtOOyRA4ecKDZCT3mJvOmQrG70VVLw">

- We won’t discuss Rotary Position Embeddings in detail, but we can say that they add the Positional Information directly at the Self-Attention layer of each Transformer block.

<img src="https://drive.google.com/uc?export=view&id=12nfVzH4SZpgLAfnBMQxKZxIKHRCGsrdd">

- RoPE adds the positional embedding in the above step of Self-Attention.
- This is just before the Relevance Scoring step - the first of the two steps that Self-Attention does.
- It basically has a formulation that adds that information to the Queries and Keys vectors.
- And so, the vector rows in the Keys matrix on the right, have some information that tells that one vector comes before another vector, which comes before another vector, etc.
- That information is present on the right, but not on the left, essentially.

<img src="https://drive.google.com/uc?export=view&id=1Q-itSuTmQA1ZxqtKdsjZS7rZ6lSIJL3m">

- One more recent development is the idea of Mixture of Experts (MoE) -  a concept that ensembles multiple sub-models to improve the quality of LLMs.
- That’s not to say all LLMs are become Mixture-of-Experts models - we can think of this as a variant of Transformer Language Models, rather than replacing the Dense models we’ve covered in the program so far.
- The idea behind Mixture-of-Experts (MoE) is that at each layer, you have multiple sub Neural Networks - each of which we call an “Expert”. 
- A router in each of these layers, decides which expert should process this token / vector.
- [Maarten Grootendorst’s visual guide to MoE](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts) is a detailed and highly illustrative explainer on the idea.

<img src="https://drive.google.com/uc?export=view&id=1JClpquifmqUBX4izaWxRsqBl78ls_8al">

- For the intuition of experts, it’s important not to think of each expert as being one monolithic component.
- Each layer has its own set of experts.
- Another important intuition is that these experts are not specialized in specific domains, like a psychology expert or a biology expert. 
- Rather, these experts tend to focus on specific kinds of tokens, and how to process them best - such as punctuation or verbs or otherwise.

<img src="https://drive.google.com/uc?export=view&id=1x29qVgU3ETB5h1TKytigK1rf8miwQCB3">

- In the flow of using a Mixture-of-Experts model, you’re not assigned to one expert.
- So if you’re assigned in Layer 1 to Expert 1, Layer 2 might use Expert 3 or Expert 4.
- This routing happens at every layer. Each layer routes to the proper expert at that layer.
- There are also methods that would route to two different experts in the same layer and merge the outputs together. So there are a couple of different methods of using these.

<img src="https://drive.google.com/uc?export=view&id=1hAEJCJkJvUuUWVrpOso_Zr4srInEawCO">

- But this is a high-level intuition.
- The Experts actually sit at the Feedforward Network stage of the Transformer block - one FFN is basically replaced by many FFNs constituting the Mixture-of-Experts.

<img src="https://drive.google.com/uc?export=view&id=1UyFIgpgBZVTC1yY-_knxqCDu7bkyfIxj">

- In addition to having these experts, the Mixture-of-Experts layer also has the router, which is basically a token-level classifier, that classifies that for this type of token, what is the best expert that is most suited for processing that token.

<img src="https://drive.google.com/uc?export=view&id=1ZbqAzNZA47En9jeHctdP9YPqCXI-7icN">

- We can think of the router as producing a Multi-class classification score, where the router has deemed that to process this token, it knows the Expert FFNN 1 will do the best job.
- And that processing is how the Feedforward Neural Network is applied to this token in this processing step in the layer.
- This was a high-level look at Mixture-of-Experts models - the next section will dive deeper into the mechanics of how they work.

## ***12 - Mixture-of-Experts (MoE)***

- Mixture-of-Experts is an exciting recent improvement to Transformer LLMs.
- This technique extends Transformers by introducing dynamically chosen experts.
- In this section, we will learn its two main components - the experts and the router.
- Mixture-of-Experts changes parts of the Decoder block inside the Transformer model.
- The input of a decoder is typically several factors representing the input tokens.
- These are first Layer-Normalized, before they pass to the Attention mechanism.
- We apply Masked Self-Attention to the inputs to weight tokens based on their relative importance in the context of all other tokens.
- This output is aggregated together with the unprocessed inputs (the residual connection), creating both a direct and an indirect path.
- This concludes one of the most important components of Transformer models - its Attention mechanism.
- It prepares the input in such a way that more contextual information is stored in the vectors.
- It is then Layer Normalized again, before it’s processed by a Feedforward Neural Network.
- This FFNN component is typically one of the largest components of an LLM, since it attempts to find complex relationships in the information processed by the Attention mechanism.
- The FFNN takes the inputs and processes it through one or more hidden layers.
- This entire mechanism is called a Dense Network, since all the parameters of the network are activated and used.

<img src="https://drive.google.com/uc?export=view&id=1HDJLuHRVtjis0_joX7KMZRA3o3IcXQ2V">



## ***13 - Conclusion***

***WIP - More Notes Coming!***
