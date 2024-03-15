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

import warnings

warnings.simplefilter("ignore")

import torch
import torch.nn as nn

from tqdm.autonotebook import tqdm

from typing import List

class KohonenSOM():
    """
    The code is developed based on the following article:
    http://www.ai-junkie.com/ann/som/som1.html
    
    The vector and matrix operations are developed using PyTorch Tensors.
    """
    def __init__( 
                    self,
                    input_dimensions : int, 
                    som_lattice_height : int = 20,
                    som_lattice_width : int = 20,
                    learning_rate : float = 0.3,
                    neighborhood_radius : float = None,
                    device : str = None 
                ):
        
        self.input_dimensions = int( input_dimensions )
        
        self.som_lattice_height = int( som_lattice_height )
        
        self.som_lattice_width = int( som_lattice_width )
        
        if learning_rate == None:
            self.learning_rate = 0.3
        else:
            self.learning_rate = float( learning_rate )
            
        if neighborhood_radius == None:
            self.neighborhood_radius = max( self.som_lattice_height, self.som_lattice_width ) / 2.0
        else:
            self.neighborhood_radius = float( neighborhood_radius )
            
        if device == None:
            self.device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
        else:
            self.device = torch.device( device )
        
        def dist_eval( data_points, weights ):
            distances = torch.cdist( data_points, weights,  p=2 )
            return distances
            
        self.dist_evaluator = dist_eval
        
        self.total_lattice_nodes = self.som_lattice_height * self.som_lattice_width
        
        self.lattice_node_weights = torch.randn( self.total_lattice_nodes, self.input_dimensions, device=self.device )
        
        lattice_coordinates = torch.tensor( [ [[i,j] for j in range(self.som_lattice_width)] for i in range( self.som_lattice_height ) ], dtype=torch.int )

        self.lattice_coordinates = lattice_coordinates.view( self.total_lattice_nodes, 2 )
        
        self.trained = False
        
        #self.dist_evaluator = nn.PairwiseDistance(p=2)

    def train( self, data_points : torch.Tensor, train_epochs : int = 100 ):
        if self.trained:
            print( "WARNING: Model is already trained. Ignoring the request..." )
            return
        
        train_epochs = int( train_epochs )
        
        total_dpoints = data_points.shape[0]
        
        data_points = data_points.to( self.device )
        
        for epoch in tqdm( range( train_epochs ), desc="Kohonen's SOM Train Epochs" ):
            decay_factor = 1.0 - (epoch/train_epochs)
            
            #learning rate is alpha in the paper
            adjusted_lr = self.learning_rate * decay_factor 
        
            #sigma in the paper
            adjusted_lattice_node_radius = self.neighborhood_radius * decay_factor 
            
            #sigma square in the paper
            squared_adjusted_lattice_node_radius = adjusted_lattice_node_radius**2
            
            distances = self.dist_evaluator( data_points, self.lattice_node_weights )
            
            best_matching_units = torch.argmin( distances, dim=1 )
            
            for i in range( total_dpoints ):
                data_point = data_points[i]
                
                bmu_index = best_matching_units[i].item()
                
                bmu_coordinates = self.lattice_coordinates[ bmu_index ]
                
                #squared distances of the lattice nodes from the bmu :: dist^2 from equation 6 shown in the paper
                squared_lattice_node_radii_from_bmu = torch.sum( torch.pow( self.lattice_coordinates.float() - bmu_coordinates.float(), 2), dim=1)
                
                squared_lattice_node_radii_from_bmu = squared_lattice_node_radii_from_bmu.to( self.device )
                
                #adjust function phi in the paper
                lattice_node_weight_adj_factors = torch.exp( -0.5 * squared_lattice_node_radii_from_bmu / squared_adjusted_lattice_node_radius )
                
                lattice_node_weight_adj_factors = lattice_node_weight_adj_factors.to( self.device )
                
                final_lattice_node_weight_adj_factors = adjusted_lr * lattice_node_weight_adj_factors
                
                final_lattice_node_weight_adj_factors = final_lattice_node_weight_adj_factors.view( self.total_lattice_nodes, 1 )
                
                final_lattice_node_weight_adj_factors = final_lattice_node_weight_adj_factors.to( self.device )
                
                lattice_node_weight_adjustments = torch.mul( final_lattice_node_weight_adj_factors, (data_point - self.lattice_node_weights) )
                
                self.lattice_node_weights = self.lattice_node_weights + lattice_node_weight_adjustments
                
                self.lattice_node_weights = self.lattice_node_weights.to( self.device )
        
        self.trained = True

    def find_best_matching_unit( self, data_points : torch.Tensor ) -> List[ List[ int ] ] :
        if len( data_points.size() ) == 1:
            #batching 
            data_points = data_points.view( 1, data_points.shape[0] )
            
        distances = self.dist_evaluator( data_points, self.lattice_node_weights )
        
        best_matching_unit_indexes = torch.argmin( distances, dim=1 )
        
        best_matching_units = [ self.lattice_coordinates[ bmu_index.item() ].tolist() for bmu_index in best_matching_unit_indexes ]
        
        return best_matching_units
    
    def find_topk_best_matching_units( self, data_points : torch.Tensor, topk : int = 1 ) -> List[ List[ int ] ] :
        if len( data_points.size() ) == 1:
            #batching 
            data_points = data_points.view( 1, data_points.shape[0] )
        
        topk = int( topk )
        
        distances = self.dist_evaluator( data_points, self.lattice_node_weights )
        
        topk_best_matching_unit_indexes = torch.topk( distances, topk, dim=1, largest=False ).indices
        
        topk_best_matching_units = []
        
        for i in range( data_points.shape[0] ):
            best_matching_unit_indexes = topk_best_matching_unit_indexes[i]
            
            best_matching_units = [ self.lattice_coordinates[ bmu_index.item() ].tolist() for bmu_index in best_matching_unit_indexes ]
            
            topk_best_matching_units.append( best_matching_units )
        
        return topk_best_matching_units