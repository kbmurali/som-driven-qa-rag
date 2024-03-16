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


from abc import ABC, abstractmethod
import torch
import math

class QuestionAnswerChatBot( ABC ):
    def __init__( 
                    self,
                    vector_db_util = None,
                    question_input_max_token_count : int = 768,
                    context_trim_percent :float = 0.05,
                    device : str = None
                ):
        if vector_db_util == None:
            raise ValueError( "ERROR: Required vector_db_util is not specified" )

        self.vector_db_util = vector_db_util

        self.question_input_max_token_count = int( question_input_max_token_count )

        self.context_trim_percent = float( context_trim_percent )

        if device == None:
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            self.device = torch.device( device )

    def find_answer_to_question( self, question : str, sim_threshold = 0.68, max_new_tokens : int = 5 ):
        if question == None or len( question.strip() ) == 0:
            raise ValueError( "ERROR: Required question is not specified" )
        
        sim_threshold = float( sim_threshold )
            
        max_new_tokens = int( max_new_tokens )
        
        qa_instruction = self.get_qa_instruction( question, sim_threshold = sim_threshold )
        
        answer_text = self.__get_answer_text__( qa_instruction, max_new_tokens = max_new_tokens )
    
        answer_text = self.__clean_answer_text__( qa_instruction, answer_text )
        
        return answer_text
    
    def find_answer_to_question_without_context( self, question : str, max_new_tokens : int = 5 ):
        if question == None or len( question.strip() ) == 0:
            raise ValueError( "ERROR: Required question is not specified" )
            
        max_new_tokens = int( max_new_tokens )
        
        qa_template = self.__qa_template__()
        
        qa_instruction = qa_template.format( "", question )
        
        answer_text = self.__get_answer_text__( qa_instruction, max_new_tokens = max_new_tokens )
    
        answer_text = self.__clean_answer_text__( qa_instruction, answer_text )
        
        return answer_text
        
    
    def get_qa_instruction( self, question : str, sim_threshold = 0.68 ) -> str :
        if question == None or len( question.strip() ) == 0:
            raise ValueError( "ERROR: Required question is not specified" )
            
        sim_threshold = float( sim_threshold )
        
        question = question.strip()
        
        qa_template = self.__qa_template__()
        
        semantic_sim_texts = self.vector_db_util.find_semantically_similar_data( question, sim_threshold = sim_threshold )

        if len( semantic_sim_texts ) == 0:
            no_context = "No context is found"
            qa_instruction = qa_template.format( no_context, question )
            return qa_instruction

        context_texts = []

        for sim_text in semantic_sim_texts:
            context = sim_text[ "text" ]
            context = context.strip()
            context_texts.append( context )
            
        context = " ".join( context_texts )
        
        qa_instruction = ""
        
        while True:
            qa_instruction = qa_template.format( context, question )
            
            tok_count = self.__token_count__( qa_instruction )
    
            if tok_count <= self.question_input_max_token_count:
                break
            else:
                if len( context ) == 0:
                    no_context = "No context is found"
                    qa_instruction = qa_template.format( no_context, question )
                    break
                    
                trim_len = math.floor( len(context) * self.context_trim_percent )

                trimmed_len = len(context) - trim_len
                
                if trim_len > 0:
                    context = context[ :trimmed_len ]
                else:
                    context = ""
    
        return qa_instruction
    
    @abstractmethod
    def __get_answer_text__( self, qa_instruction : str, max_new_tokens : int = 5 ) -> str :
        pass

    @abstractmethod
    def __token_count__( self, text : str ):    
        pass

    def __clean_answer_text__( self, qa_instruction : str, answer_text : str ) -> str :
        if qa_instruction in answer_text:
            index = answer_text.index( qa_instruction )
    
            index = index + len( qa_instruction )
    
            answer_text = answer_text[ index+1: ]
        
        extra_answer_token = "\nAnswer"
        
        if extra_answer_token in answer_text:
            extra_token_len = len( extra_answer_token )
            
            index = answer_text.index( extra_answer_token )
            
            answer_text = answer_text[ :index ] + answer_text[ index+extra_token_len: ]
        
        answer_text = answer_text.strip()
        
        return answer_text
    
    def __qa_template__( self ):
        qa_template = """Context: 
    
    {}
    
    ---
    
    Question: {}
    Answer:"""
        return qa_template