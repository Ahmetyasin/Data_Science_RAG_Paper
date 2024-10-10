# **Enhanced Retrieval-Augmented Generation (RAG) Architecture for Data Science**

In the rapidly evolving field of data science, efficiently navigating the expansive body of academic literature is crucial for informed decision-making and innovation. This repository presents the implementation of an enhanced **Retrieval-Augmented Generation (RAG)** system, specifically designed to assist data scientists in retrieving precise and contextually relevant academic resources. By integrating advanced techniques such as **semantic chunking**, **fine-tuned embedding models**, and an **abstract-first retrieval method**, this system significantly improves the relevance and accuracy of the retrieved information.

Comprehensive evaluation using the **RAGAS** framework demonstrates substantial improvements, particularly in **Context Relevancy**, making this system highly effective in reducing information overload and enhancing decision-making processes. The system shows potential in transforming academic exploration within data science, offering a valuable tool for researchers and practitioners alike.

## **Project Overview**

This repository contains the complete implementation of our enhanced RAG architecture for data science applications. The image below provides a high-level overview of the architecture.

![Enhanced RAG Architecture]([./image.png](https://github.com/Ahmetyasin/Data_Science_RAG_Paper/blob/main/img/Enhanced_Architecture.png))

### **Key Improvements in the Architecture:**
- **Semantic Chunking**: Abstracts and articles are semantically chunked to allow for improved information retrieval.
- **Fine-Tuned Embedding Models**: The embedding models are fine-tuned on academic textbooks to increase the precision of vector representation in the data science domain.
- **Abstract-First Retrieval**: An abstract-based retrieval strategy that enhances the accuracy and contextual relevance of results.
- **Vector Databases**: Custom-built vector databases for content and abstract retrieval.

All the **fine-tuning trials** shown in **Table 1** can be reproduced using the repository linked below:  
**[Fine-Tuning and Data Preparation Repository](https://github.com/Ahmetyasin/DS-Fine-Tuning-Embedding)**

This additional repository contains instructions on how to prepare data for fine-tuning and fine-tune an open-source model, which is then pushed to Hugging Face. All results in **Table 1** are generated using a similar approach.

## **Table 1: Fine-Tuning Trials**

| Model                  | Dataset                                   | Embedding Size | Performance Improvement (%) |
|------------------------|-------------------------------------------|----------------|-----------------------------|
| SentenceTransformer V1  | Textbook Data Science (Preprocessed)      | 768            | 12%                         |
| SentenceTransformer V2  | Grobid-Extracted Academic Papers          | 1024           | 18%                         |
| MPNet Fine-Tuned        | Textbook + Articles                       | 768            | 22%                         |
| OpenAI GPT-4 Embedding  | Textbook (Semantic Chunking)              | 768            | 28%                         |
| GPT-4 Large Embedding   | Textbook + Articles + Grobid-Parsed Data  | 1024           | 35%                         |

## **Repository Structure**

This repository provides all the necessary code to reproduce the results for **Table 2** and **Table 3** from the paper. The main code for these experiments can be found in the `Codes` folder.

### **Folder Structure:**

- **`Vector_DB/`**:
  - This folder contains 5 different Python scripts that form **5 different Vector DBs (Chroma)** using various configurations. The actual vector databases are not included due to their large size (several GBs).
  - All article data used for vector creation is stored in `.pkl` format within this folder.
  - **Files**:
    - `Document_list.pkl`
    - `Document_list_summary.pkl`
    - `Document_list_grobid.pkl`
    
- **`Experiments/`**:
  - This folder contains the 5 staged improvements in the RAG pipeline. Each code corresponds to a different configuration of the pipeline as outlined in the paper.
  - **Results Folder**: Includes Excel files showing results for 50-question trials and 1500-row trials.
  
- **`Results/`**:
  - Contains detailed results for each experiment, including evaluations of 50-question article sets and larger 1500-row datasets. The results are presented in **Excel** format for ease of analysis.
  
- **`Data_Science_RAG_Paper.pdf`**:
  - You can read the full paper [here](https://github.com/Ahmetyasin/Data_Science_RAG_Paper/blob/main/Data_Science_RAG_Paper.pdf).

## **How to Use This Repository**

1. **Vector DB Creation**:
   - To create and configure the vector databases, use the scripts in the `Vector_DB/` folder. These scripts utilize different configurations to store vector representations of articles.

2. **Experimentation**:
   - To reproduce the pipeline experiments, refer to the `Experiments/` folder. Each file represents a different stage in the pipeline with various optimizations.
   
3. **Results Analysis**:
   - All experimental results, including those from **50-question trials** and **1500-row trials**, can be found in the `Results/` folder in Excel format.

## **Full Paper**

For a detailed explanation of the architecture, methodology, and experimental results, you can access the full paper here:  
[Read Full Paper](https://github.com/Ahmetyasin/Data_Science_RAG_Paper/blob/main/Data_Science_RAG_Paper.pdf)
