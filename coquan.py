import pickle
import pandas as pd
import numpy as np
import tarfile
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial.distance import cosine
from sklearn.metrics import ndcg_score

# ------------- DOCUMENT IMPORTS -------------

# annotated corpus - filepath string to .pickle
annotated_corpus = '/kaggle/input/ohsuenrich/ohsu-annotated.pickle'
# sonmed words ?
words = '/kaggle/input/snomed-annot-test/snomed_words.csv'
# training data - filepath string to tar file
training_data = '/kaggle/input/ohsuenrich/dset.tar.xz'
# queries dictionary - filepath string to .pickle
queries_dict = '/kaggle/input/ohsuenrich/queries-dict-v2.pickle'

# ------------ /DOCUMENT IMPORTS -------------



# ------------------- CLASS DEFINITION ----------------------


class coquan:
    def __init__(self, annotated_corpus, words, training_data, queries_dict):
        self.annotated_corpus = annotated_corpus
        self.words = words
        self.training_data = training_data
        self.queries_dict = queries_dict
        self.extract_location = 'coquan'

    def load_data(self, extract_tar=True):  # could add flags like "all" or specific files to re-load
        # load corpus
        print("Loading the corpus...")
        with open(self.annotated_corpus, 'rb') as handle:
            self.id_dict = pickle.load(handle)

        # load snomed words
        print("Loading snomed words...")
        self.snomed_words = pd.read_csv(self.words)

        # extract traning data
        if extract_tar:
            print("Extracting the training data...  HODL!⏳")
            opened_tar = tarfile.open(self.training_data)
            if tarfile.is_tarfile(self.training_data):
                opened_tar.extractall(self.extract_location)
            else:
                print("The training_data path is not a tar file")
                raise Exception("The training_data path is not a tar file")
        
        # load training data
        ## SINCE WE DONT HAVE ALL PAPERS WE HAVE TO CLEAN UP :(
        print("Loading the training data...  HOOODLL!⏳")
        with open(self.extract_location + '/dset.pickle', 'rb') as f:
            dset = pickle.load(f)
        dset['tfidf'] = [np.array(dset['tfidf'][i])
                         for i in range(dset.shape[0])]
        self.dset = dset
        print(5*" ", "...dataset has shape of "+str(dset.shape))

        # Import the dictionary
        print("Importing queries...")
        with open(self.queries_dict, 'rb') as f:
            q_dict = pickle.load(f)
            
        # Cleaning up the dictionary
        print("Removing queries with missing answers' documents")
        ids = list(dset['ui_id'])
        total_deleted=0
        total_original =0
        paper_list = list()
        original_paper_list = list()
        for qid in q_dict:
            old_len = len(q_dict[qid]['relevant_papers'])
            total_original+=old_len
            original_paper_list=original_paper_list+q_dict[qid]['relevant_papers']
            q_dict[qid]['relevant_papers'] = [p for p in q_dict[qid]['relevant_papers'] if p[0] in ids]
            paper_list= paper_list+q_dict[qid]['relevant_papers']
            new_len = len(q_dict[qid]['relevant_papers'])
            total_deleted+=old_len-new_len

        print(5*" ","{0:<16}{1}".format("...originally: ",str(len(set(original_paper_list)))))
        print(5*" ","{0:<16}{1}".format("...deleted: ",str(total_deleted)))
        print(5*" ","{0:<16}{1}".format("...remained: ",str(len(set(paper_list)))))
        new_q_dict = dict()
        for k in q_dict:
            if len(q_dict[k]['relevant_papers'])>0:
                new_q_dict[k] = q_dict[k]
        self.q_dict = new_q_dict
        print('\nRemaining Keys : '+str([k for k in q_dict])+'\n')

        # Loading completed - confirmation
        print(21*'-', '\n   Locked & Loaded', '\n'+21*'-')

    def find_relevant_papers(self, search_term):
        # print("Finding relevant snomed IDs to word '"+search_term+"'")
        # Get rows with relevant term in 'snomed_superclasses' column
        relevant_words = self.snomed_words[self.snomed_words['snomed_superclasses'].str.contains(
            search_term, case=False)].reset_index(drop=True)
        # Get rows with relevant term in 'snomed_descriptions' column
        relevant_words = relevant_words.append(self.snomed_words[self.snomed_words['snomed_descriptions'].str.contains(
            search_term, case=False)]).reset_index(drop=True)
        # Get the SNOMED ids contained in retrieved rows
        relevant_ids = [s[2:-2].replace('"', '').replace("'", '').split(', ')
                        for s in relevant_words['snomed_ids']]
        # Flatten list
        relevant_ids = [item for sublist in relevant_ids for item in sublist]
        # Remove duplicates
        relevant_ids = list(set(relevant_ids))
        # print("Found "+str(len(relevant_ids))+" relevant IDs")

        # Search for appearances of found SNOMED ids in our annotation set
        if len(relevant_ids) > 0:
            # print("Finding papers annotated with relevant IDs")
            r = relevant_ids[0]
            relevant_papers = self.id_dict[r]
            for i in range(1, len(relevant_ids)):
                r = relevant_ids[i]
                relevant_papers = relevant_papers + self.id_dict[r]

            # print("Found "+str(len(list(set(relevant_papers))))+" relevant papers.")
        else:
            relevant_papers = []
            # print('No relevant papers')
        return list(set(relevant_papers))

    # Given a query, returns embeddings of papers which contain relevant terms to the query according to SNOMED
    def snomed_filter(self, dset, query, count=1):
        # split_query into words and remove stopwords
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(query)
        words = [w for w in word_tokens if w not in stop_words]

        words = [word.lower() for word in words if word.isalpha()]
        words = [w for w in words if w not in ['is', 'are']]
        # search all words for relevant snomed ids
        rel_ids = [self.find_relevant_papers(word) for word in words]

        # check for intersection of rpaper ids with more than 20 papers
        rel_ids_flat = [item for sublist in rel_ids for item in sublist]
        relevant_papers = list(
            set([i for i in rel_ids_flat if rel_ids_flat.count(i) > count]))

        # if intersection is very small just return the union
        if len(relevant_papers) < 20:
            # print("WARNING: The intersection of snomed ids was too small, searching in union")
            relevant_papers = list(set(rel_ids_flat))

        return relevant_papers
        '''
        ## Finally, if union is still small, relevant embeddings are all papers
        if len(relevant_papers)<20:
            print("WARNING: The union of snomed ids was too small, searching all papers")
            relevant_embeddings = emb_df
        else:
            relevant_embedding_indexes = [i for i in range(emb_df.shape[0]) if emb_df['paper_id_oshu'][i] in relevant_papers]
            relevant_embeddings = emb_df.iloc[relevant_embedding_indexes].reset_index(drop=True)

        return relevant_embeddings
        '''

    def answer_query(self, dset, use_snomed_filter=False, query_id='OHSU1',
                     vectorization='biobert', query_mode='normal',
                     choice_strategy='min', count=1):

        assert vectorization in ['biobert', 'tfidf']
        assert query_mode in ['normal', 'synonyms', 'neighbors']
        assert choice_strategy in ['min', 'mean']

        # Put everything into numpy arrays
        if vectorization == 'biobert':
            paper_embs = np.stack(dset['biobert'])
            paper_embs = paper_embs.reshape(
                paper_embs.shape[0], paper_embs.shape[-1])
            if query_mode == 'normal':
                q_embs = self.q_dict[query_id]['embedding']
            elif query_mode == 'synonyms':
                q_embs = self.q_dict[query_id]['synonym_embeddings']
            elif query_mode == 'neighbors':
                q_embs = self.q_dict[query_id]['neighbor_embeddings']
        elif vectorization == 'tfidf':
            paper_embs = np.stack(dset['tfidf'])
            q_embs = np.array(self.q_dict[query_id]['tfidf'])
            q_embs = q_embs.reshape((1, q_embs.shape[0]))

        # Calculate distances of all query embeddings with all paper embeddings
        dists = np.zeros((q_embs.shape[0], paper_embs.shape[0]))

        for i in range(dists.shape[0]):
            dists[i] = np.array([cosine(q_embs[i], paper_embs[j])
                                 for j in range(paper_embs.shape[0])])

        if choice_strategy == 'min':
            dists = np.min(dists, axis=0)
        elif choice_strategy == 'mean':
            dists = np.mean(dists, axis=0)

        dist_inds = np.argsort(dists)

        if use_snomed_filter:
            relevant_papers = list(self.snomed_filter(
                dset, self.q_dict[query_id]['title']+'. '+self.q_dict[query_id]['description'], count))
            dist_inds = [i for i in list(
                dist_inds) if dset['ohsu_id'][i] in relevant_papers]
            answers = list(dset['ui_id'][dist_inds])
            answers = answers + \
                [a for a in list(dset['ui_id']) if a not in answers]
            zeroes = np.ones((dset.shape[0] - len(dist_inds),))

            return list(answers), np.append(dists[dist_inds], zeroes)
        else:
            answers = list(dset['ui_id'][dist_inds])

        return list(answers), dists[dist_inds]

    def run_experiment(self, parameters, dset=None):
        if dset is None:
            dset = self.dset
        answers = dict()
        for k in parameters:
            print(k+':'+str(parameters[k]))
        for qid in [k for k in self.q_dict][:2]:
            print(qid)
            answers[qid] = dict()
            for s_filter in parameters['use_snomed_filter']:
                for vectorization in parameters['vectorization']:
                    # Use biobert? (different query modes and choices compared to tfidf)
                    if vectorization == 'biobert':
                        for query_mode in parameters['query_mode']:
                            # Use query mode with multiple queries?
                            if query_mode in ['synonyms', 'neighbors']:
                                for choice_strategy in parameters['choice_strategy']:
                                    # Use snomed filter?
                                    if s_filter:
                                        for count in parameters['count']:
                                            run_name = vectorization+'-'+query_mode+'-' + \
                                                choice_strategy+'-filt_' + \
                                                str(s_filter)+'-'+str(count)
                                            answers[qid][run_name] = self.answer_query(dset,
                                                                                       use_snomed_filter=s_filter,
                                                                                       query_id=qid,
                                                                                       vectorization=vectorization,
                                                                                       query_mode=query_mode,
                                                                                       choice_strategy=choice_strategy,
                                                                                       count=count)
                                    # Don't use snomed filter?
                                    else:
                                        run_name = vectorization+'-'+query_mode + \
                                            '-'+choice_strategy + \
                                            '-filt_'+str(s_filter)
                                        answers[qid][run_name] = self.answer_query(dset,
                                                                                   use_snomed_filter=s_filter,
                                                                                   query_id=qid,
                                                                                   vectorization=vectorization,
                                                                                   query_mode=query_mode,
                                                                                   choice_strategy=choice_strategy)
                            # Dont use query mode with multiple queries?
                            else:
                                # Use SNOMED filter?
                                if s_filter:
                                    for count in parameters['count']:
                                        run_name = vectorization+'-'+query_mode + \
                                            '-filt_'+str(s_filter) + \
                                            '-'+str(count)
                                        answers[qid][run_name] = self.answer_query(dset,
                                                                                   use_snomed_filter=s_filter,
                                                                                   query_id=qid,
                                                                                   vectorization=vectorization,
                                                                                   query_mode=query_mode,
                                                                                   count=count)
                                # Dont use SNOMED filter?
                                else:
                                    run_name = vectorization+'-' + \
                                        query_mode+'-filt_'+str(s_filter)
                                    answers[qid][run_name] = self.answer_query(dset,
                                                                               use_snomed_filter=s_filter,
                                                                               query_id=qid,
                                                                               vectorization=vectorization,
                                                                               query_mode=query_mode)
                    # Use tfidf?
                    else:
                        # Use SNOMED filter?
                        if s_filter:
                            for count in parameters['count']:
                                answers[qid][vectorization+'-filt_'+str(s_filter)+'-'+str(count)] = self.answer_query(
                                    dset, use_snomed_filter=s_filter, query_id=qid, vectorization=vectorization, count=count)
                        # Dont use SNOMED filter
                        else:
                            answers[qid][vectorization+'-filt_'+str(s_filter)] = self.answer_query(
                                dset, use_snomed_filter=s_filter, query_id=qid, vectorization=vectorization)

        self.answers = answers
        return answers

    def evaluate(self, answers=None):
        if answers is None:
            answers = self.answers
        # metrics=list()
        for qid in answers:
            for run in answers[qid]:
                met = self.ndcg_score_query(answers[qid][run], query=qid, depth=100)
                print(str(run) + ' query '+qid+' ndcg??  : '+str(met))

            print('__________________________')

    def ndcg_score_query(self, answers, query, depth):
        pred_ids, pred_relevance = answers
        pred_relevance = 1-pred_relevance
        true_relevance = np.zeros((len(pred_relevance)))

        # gia kathe paper apo ta relevant
        for paper_id, score in self.q_dict[query]["relevant_papers"]:
            # psaxnw na vrw to shmeio opou to paper_id einai mesa stis apanthseis tou susthmatos
            for i, k in enumerate(pred_ids):
                if paper_id == k:
                    break
            true_relevance[i] = int(score)

        score = ndcg_score([true_relevance], [pred_relevance], k=depth)
        return score

# ------------------- /CLASS DEFINITION ----------------------
