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
# In[1]:


from abc import ABC, abstractmethod

import torch

from typing import List


# In[2]:


class VectorEncoder( ABC ):
    def __init__( self, encoded_vector_dimensions : int ):
        self.encoded_vector_dimensions = int( encoded_vector_dimensions )
    
    def get_encoded_vector_dimensions( self ) -> int :
        return self.encoded_vector_dimensions
    
    @abstractmethod
    def encode( self, text : str ) -> torch.Tensor :
        pass
    
    @abstractmethod
    def encode_batch( self, list_of_text : List[ str ] ) -> torch.Tensor :
        pass


# In[ ]:




