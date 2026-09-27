from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

with open('knowledge.txt','r',encoding='utf-8') as f:
    text=f.read()

Chunks=[
    chunk.strip()for chunk in text.split('\n\n')
    if chunk.strip()
]
print('Chunks:')
for i,chunk in enumerate(Chunks):
    print(f'{i}:{chunk}')

#create embeddings
vectorize=TfidfVectorizer()
document_vector=vectorize.fit_transform(Chunks)

## User Query
query=input('\n Ask a Question:')

query_vector=vectorize.transform([query])

similarity=cosine_similarity(query_vector,document_vector)[0]
print(similarity)


#find best chunck
best_index=similarity.argmax()
best_chunk=Chunks[best_index]
print('\n Retrieved Chunk:\n')
print(best_chunk)
print('\n similarity score:')
print(similarity[best_index])
