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

import openai

from openai.embeddings_utils import get_embedding

import torch

from vector_encoder_parent import VectorEncoder

from typing import List


class OpenAIEmbeddingsVectorEncoder( VectorEncoder ):
    def __init__( 
                    self,
                    encoded_vector_dimensions : int,
                    vector_encoder_id : str,
                    openai_key : str
                ):
        super().__init__( encoded_vector_dimensions )
        
        if vector_encoder_id == None or len( vector_encoder_id.strip() ) == 0:
            raise ValueError( "ERROR: vector_encoder_id must be specified" )
            
        if openai_key == None or len( openai_key.strip() ) == 0:
            raise ValueError( "ERROR: openai_key must be specified" )
        
        self.vector_encoder_id = vector_encoder_id.strip()
        
        self.openai_key = openai_key.strip()

    def encode( self, text : str ) -> torch.Tensor:
        if text == None or len( text.strip() ) == 0:
            raise ValueError( "ERROR: Required text is not specified" )
        
        openai.api_key = self.openai_key
        
        embeddings = get_embedding( text, engine = self.vector_encoder_id )
        
        vector = torch.tensor( embeddings, dtype=torch.float )
        
        return vector
    
    def encode_batch( self, list_of_text : List[ str ] ) -> torch.Tensor :
        if list_of_text == None or len( list_of_text ) == 0:
            raise ValueError( "ERROR: Required list_of_text is None or empty" )
        
        list_of_text = [ str( text ) for text in list_of_text ]
        
        openai.api_key = self.openai_key
        
        response = openai.Embedding.create(
                                            input = list_of_text,
                                            engine = self.vector_encoder_id
                                          )
        
        embeddings = [ data["embedding"] for data in response["data"] ] 
        
        vectors = torch.tensor( embeddings, dtype=torch.float )
        
        return vectors
