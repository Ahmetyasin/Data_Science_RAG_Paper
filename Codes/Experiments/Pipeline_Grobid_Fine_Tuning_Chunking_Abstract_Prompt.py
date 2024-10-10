from langchain_community.retrievers import ArxivRetriever
from langchain.chains import ConversationalRetrievalChain
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoConfig
from langchain_community.llms import HuggingFacePipeline
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import ArxivLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
import pandas as pd
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_relevancy,
)
import openai
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import time
import nest_asyncio
nest_asyncio.apply()

# Temporarily adjust display options
pd.set_option('display.max_rows', None)  # No limit on the number of rows
pd.set_option('display.max_columns', None)  # No limit on the number of columns
pd.set_option('display.width', None)  # No limit on the display width to avoid wrapping
pd.set_option('display.max_colwidth', None)  # Display full content of each cell

os.environ["OPENAI_API_KEY"] = "Your-OpenAI-API-Key"


# Set OpenAI API Key
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

    
# Single detailed question
detailed_question = [
"I'm a data scientist in the financial sector, selecting features to predict stock prices. I've used traditional methods like backward elimination. Which advanced feature selection technique should I consider to enhance model performance?",
"In the context of agriculture, I'm predicting crop yields based on soil and weather data. Having tried manual feature selection, what automated feature selection method would help in identifying the most influential factors for yield prediction?",
"As a data scientist in the insurance industry, I'm working on customer churn prediction. I've used correlation-based feature selection but need to account for nonlinear relationships. Which technique would be suitable for this purpose?",
"Working in the healthcare sector, I deal with patient vital signs data that often contains outliers. After using Z-score for outlier detection, what robust outlier elimination technique should I adopt for cleaner datasets?",
"In the manufacturing industry, I'm analyzing sensor data from machinery for predictive maintenance. Having used the IQR method to detect outliers, which advanced technique would you recommend for more accurate anomaly detection and elimination?",
"I'm analyzing customer behavior data in e-commerce, using PCA for dimensionality reduction. Can you suggest a modern technique that might offer better insights while preserving data variance?",
"In environmental science, I deal with high-dimensional climate data. Having used t-SNE for visualization, what other dimensionality reduction method should I consider for more effective data analysis?",
"As a data scientist in the telecom sector, I work with extensive network traffic data. After applying LDA for feature extraction, what newer technique would enhance the interpretability and accuracy of my models?",
"In the energy industry, I'm analyzing sensor data from smart grids. While PCA has helped in reducing dimensions, what alternative should I explore for better performance with large datasets?",
"I'm studying brain imaging data for neurological research. Having tried factor analysis, what other dimensionality reduction technique could help in uncovering hidden patterns more effectively?",
"In the automotive industry, I develop models for autonomous vehicles using real-world driving data. What synthetic data generation technique can I use to augment my dataset while ensuring realistic scenarios?",
"Working with confidential financial data, I need to generate synthetic datasets for model training. Which method would provide a balance between data utility and privacy?",
"As a data scientist in social sciences, I deal with sensitive survey data. What technique should I use to generate synthetic data that maintains the statistical properties of the original data?",
"In robotics, I'm training a robot to navigate complex environments. Having used Q-learning, what advanced reinforcement learning algorithm should I explore for improved performance in dynamic settings?",
"As a game developer, I'm using reinforcement learning to create intelligent agents. Beyond DQN, which algorithm would enhance the adaptability and learning speed of my agents?",
"In the pharmaceutical industry, I'm classifying drug responses based on genetic data. After using SVM, what advanced classification algorithm should I consider for better accuracy and interpretability?",
"As a data scientist in the automotive sector, I'm classifying vehicle types from sensor data. Having used decision trees, which more sophisticated method could improve classification performance?",
"Working in cybersecurity, I classify network traffic for threat detection. Beyond random forests, what advanced classification technique should I employ for more accurate threat identification?",
"In the education sector, I'm classifying student performance based on learning behaviors. After using k-NN, which algorithm would offer better results considering interpretability and accuracy?",
"As a data scientist in the legal domain, I'm classifying legal documents by type. What advanced classification method would enhance my document categorization system?",
"In the energy sector, I'm predicting energy consumption using multiple regression. What advanced regression model could provide better accuracy and handle nonlinearity more effectively?",
"Working in the telecommunications industry, I'm predicting customer call durations. Having tried ridge regression, which sophisticated regression technique should I explore next?",
"In the field of epidemiology, I'm predicting disease spread based on environmental factors. Beyond polynomial regression, what model should I use to improve predictive accuracy?",
"As a data scientist in real estate, I'm predicting property values. After using lasso regression, which advanced method would you recommend for incorporating spatial dependencies?",
"In transportation, I'm predicting traffic flow using regression models. What advanced technique should I consider for capturing complex patterns in traffic data?",
"In the context of smart cities, I'm clustering IoT sensor data. Having used DBSCAN, what other clustering algorithm would be effective for large, diverse datasets?",
"As a data scientist in the food industry, I'm clustering consumer taste preferences. Beyond hierarchical clustering, which algorithm should I explore for better segmentation?",
"Working in the entertainment sector, I'm clustering movie preferences. What advanced clustering method can help in identifying more nuanced audience segments?",
"In environmental monitoring, I'm clustering air quality data. After using agglomerative clustering, which technique would provide better results for large-scale environmental datasets?",
"As a data scientist in sports analytics, I'm clustering player performance metrics. Beyond Gaussian Mixture Models, which advanced clustering method should I consider for improved insights?",
"In medical imaging, I'm detecting abnormalities in MRI scans. Beyond CNNs, what advanced deep learning architecture should I explore for higher accuracy?",
"As a data scientist in agriculture, I'm analyzing drone imagery for crop health. Having used image segmentation, which technique should I adopt for more precise analysis?",
"Working with satellite imagery in environmental science, I'm classifying land use. Beyond traditional CNNs, which model should I explore for improved accuracy and efficiency?",
"In the automotive sector, I'm developing object detection systems for autonomous vehicles. What advanced image processing technique would enhance detection accuracy in real-time scenarios?",
"As a data scientist in fashion, I'm analyzing images for style recognition. Beyond transfer learning, which approach should I use to improve model performance?",
"In legal tech, I'm analyzing contracts using NLP. Having used word embeddings, what advanced technique should I consider for better understanding legal jargon and nuances?",
"As a data scientist in customer service, I'm developing chatbots for automated responses. Beyond basic sequence models, which advanced NLP model should I explore for more natural interactions?",
"Working in publishing, I'm analyzing book reviews for sentiment analysis. What state-of-the-art NLP technique would help in capturing complex sentiments more accurately?",
"In the field of education, I'm developing automated grading systems. Having used traditional NLP methods, which advanced model should I adopt for better accuracy in grading essays?",
"As a data scientist in social media analysis, I'm detecting trends from tweets. Beyond LSTM, which cutting-edge NLP technique should I consider for more insightful trend analysis?",
"In finance, I'm forecasting stock prices using traditional ARIMA models. What advanced time series analysis method should I use to better capture market volatility?",
"As a data scientist in retail, I'm predicting sales trends. Beyond SARIMA, which technique should I explore to account for seasonal variations and promotions?",
"Working in the energy sector, I'm forecasting electricity demand. Having used exponential smoothing, what advanced method should I consider for more accurate demand predictions?",
"In healthcare, I'm analyzing patient vital signs over time. What advanced time series model would enhance the predictive accuracy of health outcomes?",
"As a data scientist in transportation, I'm forecasting passenger flow. Beyond traditional methods, which advanced technique should I adopt to improve forecasting accuracy?",
"In the context of fraud detection, I'm evaluating the performance of my models. Beyond accuracy and precision, what other evaluation metrics should I consider for a more comprehensive assessment?",
"As a data scientist in healthcare, I'm evaluating models for disease prediction. Which evaluation method would help in understanding the trade-offs between sensitivity and specificity more effectively?",
"Working in marketing, I'm assessing the effectiveness of customer segmentation models. Beyond silhouette score, what other evaluation techniques should I use to validate the quality of my clusters?",
"In the field of education, I'm evaluating models for predicting student success. What advanced evaluation metrics should I consider to ensure my models are both accurate and fair?",
"As a data scientist in manufacturing, I'm assessing the performance of predictive maintenance models. Beyond traditional metrics, which evaluation method should I use to measure the real-world impact of my models?"
]

embedding_path = 'AhmetAytar/all-mpnet-base-v2-fine-tuned_17_textbook_grobid_semantic'

encode_kwargs = {'normalize_embeddings': True}

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':'cpu'}


# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_path,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs # Pass the encoding options
)


vectorstore_abstract = Chroma(persist_directory="./Vector_DB/chroma_grobid_fine_tuned_chunking_abstract", embedding_function=embeddings)
vectorstore = Chroma(persist_directory="./Vector_DB/chroma_grobid_fine_tuned_chunking", embedding_function=embeddings)

print("vectorstore is loaded")
# Retrieve and generate using the relevant snippets of the blog.
#retriever = vectorstore.as_retriever(search_kwargs={'k' : 1})


template = """You are the best assistant for question-answering tasks. 
Your role is to answer the question excellently using the provided context.
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise. I will tip you $1000 for a perfect response!

Question: {question} 

Context: {context} 

Answer:"""


prompt = PromptTemplate.from_template(template)

llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def run_evaluation(data_dict):
    results = []
    for item in data_dict:
        attempt = 0
        while attempt < 4:
            try:
                result = evaluate(
                    dataset=Dataset.from_dict(item), 
                    metrics=[
                        context_relevancy,
                        faithfulness,
                        answer_relevancy,
                    ],
                    llm=llm,
                    raise_exceptions = False
                )
                results.append(result.to_pandas())
                time.sleep(2)
                break  # Exit the retry loop on success
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                attempt += 1
                time.sleep(10)  # Wait before retrying

        if attempt == 4:
            print("Maximum retries reached, moving to next item.")
            results.append(pd.DataFrame([{"context_relevancy": None, "faithfulness": None, "answer_relevancy": None}]))
            break

    return pd.concat(results, ignore_index=True)

def run_experiment(run_number):
    answers = []
    contexts = []

    detailed_questions = detailed_question if isinstance(detailed_question, list) else [detailed_question]

    for query in detailed_questions:
        retrieved_docs = vectorstore_abstract.similarity_search(query=query, k=100)
        titles=[]
        for i in retrieved_docs:
            titles.append(i.metadata['Title'])
        retriever = vectorstore.as_retriever(search_kwargs={'k' : 1,'filter': {'Title': {'$in': titles}}})

        rag_chain_from_docs = (
            RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
            | prompt
            | llm
            | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        answers.append(rag_chain_with_source.invoke(query))
        contexts.append([doc.page_content for doc in retriever.get_relevant_documents(query,search_kwargs={'k' : 1})])

        print(answers[-1])

    extracted_answers = [answer_dict['answer'] for answer_dict in answers]

    data_to_evaluate = [
        {
            "question": [detailed_questions[i]],
            "answer": [extracted_answers[i]],
            "contexts": [contexts[i]]
        }
        for i in range(len(detailed_questions))
    ]

    eval_df_list = []
    for item in data_to_evaluate:
        eval_df = run_evaluation([item])
        eval_df_list.append(eval_df)

    eval_df = pd.concat(eval_df_list, ignore_index=True)

    df = pd.DataFrame(data_to_evaluate)
    df = pd.concat([df, eval_df], axis=1)
    df['run_number'] = run_number

    return df

# Initialize an empty DataFrame to store all results
all_results = pd.DataFrame()

# Run the experiment 30 times and collect all results
for i in range(1, 31):
    result_df = run_experiment(i)
    all_results = pd.concat([all_results, result_df], ignore_index=True)
    print(all_results.to_string())
    print(f"Run {i} completed.")

# Save the combined results to a CSV file
all_results.to_csv('./Results/Results_pipeline_grobid_fine_tuning_chunking_abstract_prompt.csv', index=False)
print("All runs completed and results saved to ./Results/Results_pipeline_grobid_fine_tuning_chunking_abstract_prompt.csv")










