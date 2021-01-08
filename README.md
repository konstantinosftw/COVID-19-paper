# COVID-19-paper

## Pipeline

### a. New Corpus
	1. Enrich papers with an ontology [SNOMED] using an appropriate tool [MetaMap]
	2. Further enrich the Corpus utilizing the embedded SNOMED IDs. Methods:
		1. Neighbors
		2. Synonyms

### b. New query
	1. Receive a ranked list of documents from the model [BioBERT], most similar to the original query
	2. Enrich the query using the same method as on the Corpus
	3. Further enrich the query with the same methods as we did the Corpus
	4. Utilize the enriched concepts generated for the query and the returned documents to further filter and re-rank the documents. Filtering methods:
		1. count # of concept intersections between qry and doc
		2. create SNOMED embeddings


### c. Evaluation

	1. Use the appropriate evaluation metrics & visualization tools [TREC, ggplot]

