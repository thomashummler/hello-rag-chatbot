from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import JoinDocuments, SentenceTransformersRanker
from haystack.pipelines import Pipeline
import pandas as pd
import numpy as np
from openai import OpenAI
import os

openai.api_key = os.environ["API_KEY"]

file_path = 'Rieker_SUMMERANDWINTER_DATA.xlsx'

Rieker_Database = pd.read_excel(file_path)

seed_value = 42
np.random.seed(seed_value)

# Your original code
df_groupByColor_Rieker = Rieker_Database.groupby('Main_Color', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByShoeType_Rieker = Rieker_Database.groupby('main_category', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByGender_Rieker = Rieker_Database.groupby('Warengruppe', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupBySaison_Rieker = Rieker_Database.groupby('Saison_Catch', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))
df_groupByMaterial_Rieker = Rieker_Database.groupby('EAS Material', group_keys=False).apply(lambda x: x.sample(10, replace=True, random_state=seed_value))

result_df = pd.concat([df_groupByColor_Rieker, df_groupByShoeType_Rieker, df_groupByGender_Rieker, df_groupBySaison_Rieker, df_groupByMaterial_Rieker], ignore_index=True)
result_df = result_df.drop_duplicates(subset='ID', keep='first')
Rieker_Database = result_df

docs = []
for index, row in Rieker_Database.iterrows():
    document = {
        'content': ', '.join(str(value) for value in row),
        'meta': {'ID': row['ID'],'Main_Color': row['Main_Color'], 'Main_Category': row['main_category'], 'Gender': row['Warengruppe'], 'Saison': row['Saison_Catch'],'Main_Material': row['EAS Material']}
    }
    docs.append(document)


print(docs)

#Gerade unklar ob preprocessor wirklich benutzt wird
from haystack.nodes import PreProcessor
preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=512,    #Vlt hier die Länge auf 100 setzen um die
    split_overlap=32,
    split_respect_sentence_boundary=True, # Stellt sicher dass das Dokument nicht mitten im Satz aufgeteilt wird, um semantische Bedeutung für semantische Suche aufrecht erhalten werden kann
)


docs_to_index = preprocessor.process(docs)

document_store = InMemoryDocumentStore(use_bm25=True, embedding_dim=384)



sparse_retriever = BM25Retriever(document_store=document_store)
dense_retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",   # Jeder Dense Retriever hat im Normalfall eine begrenzte Anzahl an Tokens die er gleichzeitig verarbeiten kann-> im Normfall 256
                                                                # Welche Dense Retriever Methode wird verwendet? It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
    use_gpu=True,
    scale_score=False,
)

document_store.delete_documents()
document_store.write_documents(docs_to_index)   #Hier evtl docs_to_index einfügen falls preprocessor wirklich nötig
document_store.update_embeddings(retriever=dense_retriever) #Hier Laaaaange Laufzeit Batches die entschlüsselt werden -> dauert hier die Indexierung so lange?

join_documents = JoinDocuments(join_mode="concatenate")     # Dabei werden beim Hybrid Retrieval alle Dokumente einfach an die Liste drangehängt ohne die Reihenfolge zu verändern
rerank = SentenceTransformersRanker(model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")

pipeline = Pipeline()
pipeline.add_node(component=sparse_retriever, name="SparseRetriever", inputs=["Query"])
pipeline.add_node(component=dense_retriever, name="DenseRetriever", inputs=["Query"])
pipeline.add_node(component=join_documents, name="JoinDocuments", inputs=["SparseRetriever", "DenseRetriever"])
pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])

#Ausprobieren der Pipeline
from haystack.utils import print_answers

prediction = pipeline.run(
    query="Ich suche einen Schuh in der Farbe rot",
    params={
        "SparseRetriever": {"top_k": 10},
        "DenseRetriever": {"top_k": 10},
        "JoinDocuments": {"top_k_join": 15},  # comment for debug
        # "JoinDocuments": {"top_k_join": 15, "debug":True}, #uncomment for debug
        "ReRanker": {"top_k": 5},
    },
)

documents = prediction['documents'] 
# Gegebene Liste von Dokumenten

# Schleife durch alle Dokumente
for document in documents:
    # Zugriff auf die relevanten Informationen
  print(document)