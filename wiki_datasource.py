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

import requests
import pandas as pd
from dateutil.parser import parse
from typing import List


class WikiEventsDataSource():
    def __init__( 
                    self, 
                    event_years_to_fetch : List[int]
                ):
        if event_years_to_fetch == None or len( event_years_to_fetch ) == 0:
            raise ValueError( "Argumemt event_years_to_fetch is required. Specified value is None or empty." )
        
        self.event_years_to_fetch = [ int(year) for year in event_years_to_fetch ]
        
        self.event_years_to_fetch = [ str(year) for year in self.event_years_to_fetch ]
        
        self.fetched = False
        
        self.df = None
    
    def get_data( self ) -> pd.DataFrame :
        return self.df
    
    def fetch_n_prepare_data( self ):
        if self.fetched:
            print( "WARNING: Wiki events for the specified years already fetched. Ignoring the request..." )
            return
        
        main_df = pd.DataFrame()
        
        for year in self.event_years_to_fetch:
            wiki_api_params = {
                                "action": "query", 
                                "prop": "extracts",
                                "exlimit": 1,
                                "titles": year,
                                "explaintext": 1,
                                "formatversion": 2,
                                "format": "json"
                              }
        
            response = requests.get( "https://en.wikipedia.org/w/api.php", params=wiki_api_params )
        
            response_dict = response.json()
        
            df = pd.DataFrame()
        
            df[ "text" ] = response_dict["query"]["pages"][0]["extract"].split("\n")
            
            df = self.__clean_df__( df, year )
            
            main_df = pd.concat( [ main_df, df ] )
        
        self.df = main_df.reset_index(drop=True)
        
        self.fetched = True
        
    def __clean_df__( self, df : pd.DataFrame, year : str ) -> pd.DataFrame :
        #Filter off blank text and other header text
        df = df[ (df[ "text" ].str.len() > 0) & (~df[ "text" ].str.startswith( "==" )) ]
        
        prefix = ""

        #Iterate through rows of the dataframe and format events that are listed under the date of their occurrence.
        for (i, row) in df.iterrows():
            #Only check for the events for which the date is not already prefixed, separated by " – "
            if " – " not in row["text"]:
                try:
                    # Check if the row text is actually the date (date header row). The parse function call will pass if it is a valid date such as August 29
                    parse(row["text"])

                    # If the parse function call is successful, then the row text is a date and hence can be used as a prefix for the following rows that are not formatted with an " – "
                    prefix = row["text"]
                except:
                    # Since the parse function call threw an exception, it means the row text is not a date and must be the event text without a date prefix. Hence, prepend the date prefix captured in the previous loop.
                    row["text"] = prefix + " – " + row["text"]
                    
        #Local date functions to filter rows using parse date logic
        def startswith_date( evt : str, sep='–' ):
            dt_str = evt.split( sep )[0]
            try:
                parse( dt_str )
                return True
            except:
                return False
    
        def dateOnlyText( dt_str : str ):
            try:
                parse( dt_str )
                return True
            except:
                return False
    
        #Select only the rows that contain the date-prefixed event texts - essentially remove the rows with pure dates since we do not need them anymore, and reset the index of the dataframe
        df = df[ df[ "text" ].apply( lambda evt: not dateOnlyText( evt ) and startswith_date( evt ) ) ].reset_index(drop=True)
        
        #Iterate through rows of the dataframe and format date to append year.
        for (i, row) in df.iterrows():
            evt = row["text"]
            
            splits = evt.split( '–' )
            
            dt_str = splits[0].strip()
            
            #Example: Need length for something like "January 15 – "
            first_toks_len = len( dt_str ) + 3
            
            remainder_str = evt[first_toks_len:].strip()
            
            row["text"] = "".join( [ dt_str, ", ", year, " – ", remainder_str ])
                    
        return df