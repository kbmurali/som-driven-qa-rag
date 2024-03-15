#!/usr/bin/env python
# coding: utf-8
'''
------------------------------------------------------------------------------
   Copyright 2024 Murali Kashaboina

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
------------------------------------------------------------------------------
'''
# In[3]:


import warnings

warnings.simplefilter("ignore")

from tqdm.autonotebook import tqdm

import pandas as pd
import torch
import torch.nn as nn

from vector_encoder_parent import VectorEncoder

from vector_indexer import SOMBasedVectorIndexer


# In[4]:


class SOM_Based_RAG_Util():
    def __init__( 
                    self,
                    vector_encoder : VectorEncoder, 
                    som_lattice_height : int = 20,
                    som_lattice_width : int = 30,
                    learning_rate : float = 0.3,
                    neighborhood_radius : float = None,
                    topk_bmu_for_indexing : int = 1,
                    vectorize_batch_size : int = 100,
                    device : str = None 
                ):
        if vector_encoder == None:
            raise ValueError( "ERROR: vector_encoder must be specified" )
            
        self.vector_encoder = vector_encoder
        
        self.vectorize_batch_size = int( vectorize_batch_size )
        
        if device == None:
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            self.device = torch.device( device )
        
        self.vector_indexer = SOMBasedVectorIndexer(
                                                        self.vector_encoder.get_encoded_vector_dimensions(),
                                                        som_lattice_height = som_lattice_height,
                                                        som_lattice_width = som_lattice_width,
                                                        learning_rate = learning_rate,
                                                        neighborhood_radius = neighborhood_radius,
                                                        topk_bmu_for_indexing = topk_bmu_for_indexing,
                                                        device = self.device
                                                   )
        
        self.data_loaded_n_vectorized = False
        
        self.data_vectors_indexed = False
        
        self.df = None
        
        self.vectors = None
    
    def find_semantically_similar_data( self, query: str, sim_evaluator = None, sim_threshold : float = 0.8  ):
        if not self.data_vectors_indexed:
            raise ValueError( "ERROR: Data vectors not indexed." )
            
        if query == None or len( query.strip() ) == 0:
            raise ValueError( "ERROR: Required query text is not specified." )
            
        sim_threshold = float( sim_threshold )
        
        if sim_evaluator == None:
            sim_evaluator = nn.CosineSimilarity(dim=0, eps=1e-6)
        
        query_vector = self.vector_encoder.encode( query )
        
        query_vector = query_vector.view( self.vector_encoder.get_encoded_vector_dimensions() )
        
        query_vector = query_vector.to( self.device )
        
        nearest_indexes = self.vector_indexer.find_nearest_indexes( query_vector )
        
        nearest_indexes = nearest_indexes[0]
        
        sim_scores = []
        
        for idx in nearest_indexes:
            data_vector = self.vectors[ idx ]
            
            data_vector = data_vector.view( self.vector_encoder.get_encoded_vector_dimensions() )
            
            sim_score = sim_evaluator( query_vector, data_vector )
            
            if sim_score >= sim_threshold:
                sim_score_tuple = (idx, sim_score.item() )
            
                sim_scores.append( sim_score_tuple )
        
        sim_scores.sort( key = lambda x: x[1], reverse=True )
        
        semantically_similar_data = [ 
                                        { 
                                            'text': self.df[ 'text' ][ idx ],
                                            'sim_score' : sim_score
                                        } for idx, sim_score in sim_scores
                                    ]
        
        return semantically_similar_data
    
    def find_semantically_similar_data_idx( self, query: str, sim_evaluator = None, sim_threshold : float = 0.8  ):
        if not self.data_vectors_indexed:
            raise ValueError( "ERROR: Data vectors not indexed." )
            
        if query == None or len( query.strip() ) == 0:
            raise ValueError( "ERROR: Required query text is not specified." )
            
        sim_threshold = float( sim_threshold )
        
        if sim_evaluator == None:
            sim_evaluator = nn.CosineSimilarity(dim=0, eps=1e-6)
        
        query_vector = self.vector_encoder.encode( query )
        
        query_vector = query_vector.view( self.vector_encoder.get_encoded_vector_dimensions() )
        
        query_vector = query_vector.to( self.device )
        
        nearest_indexes = self.vector_indexer.find_nearest_indexes( query_vector )
        
        nearest_indexes = nearest_indexes[0]
        
        sim_scores = []
        
        for idx in nearest_indexes:
            data_vector = self.vectors[ idx ]
            
            data_vector = data_vector.view( self.vector_encoder.get_encoded_vector_dimensions() )
            
            sim_score = sim_evaluator( query_vector, data_vector )
            
            if sim_score >= sim_threshold:
                sim_score_tuple = (idx, sim_score.item() )
            
                sim_scores.append( sim_score_tuple )
        
        sim_scores.sort( key = lambda x: x[1], reverse=True )
        
        return sim_scores
    
    def load_n_vectorize_data( self, data_source ):
        if self.data_loaded_n_vectorized:
            print( "WARNING: Data already loaded and vectorized. Ignoring the request..." )
            return
        
        data_source.fetch_n_prepare_data()
        
        self.df = data_source.get_data()
        
        vectors = None
        
        for i in tqdm( range(0, len(self.df), self.vectorize_batch_size ), desc="Vectorized Data Batch" ):
            list_of_text = self.df.iloc[ i:i+self.vectorize_batch_size ]["text"].tolist()
            
            batch_encoded_vectors = self.vector_encoder.encode_batch( list_of_text )
            
            if vectors == None:
                vectors = batch_encoded_vectors
            else:
                vectors = torch.cat( [ vectors, batch_encoded_vectors], dim=0 )
                
        self.vectors = vectors.to( self.device )
        
        self.data_loaded_n_vectorized = True
    
    def train_n_index_data_vectors( self, train_epochs : int = 100  ):
        if not self.data_loaded_n_vectorized:
            raise ValueError( "ERROR: Data not loaded and vectorized." )
        
        if self.data_vectors_indexed:
            print( "WARNING: Data vectors already indexed. Ignoring the request..." )
            return
        
        self.vector_indexer.train_n_gen_indexes( self.vectors, train_epochs )
        
        self.data_vectors_indexed = True

