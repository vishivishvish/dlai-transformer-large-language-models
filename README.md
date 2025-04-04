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


## ***9 - Self-Attention***

## ***10 - Model Example***

## ***11 - Recent Improvements***

## ***12 - Mixture-of-Experts (MoE)***

## ***13 - Conclusion***

***WIP - More Notes Coming!***
