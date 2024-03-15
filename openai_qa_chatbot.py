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

import tiktoken

from qa_chatbot import QuestionAnswerChatBot

class OpenAIQuestionAnswerChatBot( QuestionAnswerChatBot ):
    def __init__( 
                    self,
                    vector_db_util = None,
                    openai_tokenizer_name : str = None,
                    openai_model_name : str = None,
                    openai_key : str = None,
                    question_input_max_token_count : int = 768,
                    context_trim_percent :float = 0.05,
                    device : str = None
                ):
        super().__init__( 
                    vector_db_util,
                    question_input_max_token_count = question_input_max_token_count,
                    context_trim_percent = context_trim_percent,
                    device = device
                )
        if openai_tokenizer_name == None or len( openai_tokenizer_name.strip() ) == 0:
            raise ValueError( "ERROR: openai_tokenizer_name must be specified" )
        
        if openai_model_name == None or len( openai_model_name.strip() ) == 0:
            raise ValueError( "ERROR: openai_model_name must be specified" )

        if openai_key == None or len( openai_key.strip() ) == 0:
            raise ValueError( "ERROR: openai_key must be specified" )

        self.openai_tokenizer_name = openai_tokenizer_name.strip()
        
        self.tokenizer = tiktoken.get_encoding( self.openai_tokenizer_name )
        
        self.openai_model_name = openai_model_name.strip()
        
        self.openai_key = openai_key.strip()
    
    def __get_answer_text__( self, qa_instruction : str, max_new_tokens : int = 5 ) -> str :
        openai.api_key = self.openai_key
        
        basic_answer = openai.Completion.create(
                                                    model = self.openai_model_name,
                                                    prompt = qa_instruction, 
                                                    
                                               )
    
        answer_text = basic_answer[ "choices" ][0][ "text" ]
    
        return answer_text

    def __token_count__( self, text : str ):    
        return len( self.tokenizer.encode( text ) )

