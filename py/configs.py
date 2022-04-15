from distutils.command.config import config


fully_fused = {
	"otype": "FullyFusedMLP",    
	"activation": "ReLU",        
	"output_activation": "None", 
	"n_neurons": 128,            
	                             
	"n_hidden_layers": 5,        
	"feedback_alignment": False  
	                             
}

cutlass = {
	"otype": "CutlassMLP",       
	"activation": "ReLU",        
	"output_activation": "None", 
	"n_neurons": 128,            
	"n_hidden_layers": 5         
}

adam_config = {
	"otype": "Adam",      
	"learning_rate": 1e-3,
	"beta1": 0.9,         
	"beta2": 0.999,       
	"epsilon": 1e-8,      
	"l2_reg": 1e-8,       
	                      
	"relative_decay": 0,  
	"absolute_decay": 0,  
	"adabound": False     
}

def get_CUTLASS(n_neurons, n_hidden, output_act = "ReLU", hidden_act = "ReLU"):
    config = cutlass
    config['n_neurons'] = n_neurons
    config['n_hidden'] = n_hidden
    config['output_activation'] = output_act
    config['output_activation'] = hidden_act
    return config

def get_FULLY_FUSED(n_neurons, n_hidden, output_act = "ReLU", hidden_act = "ReLU"):
    config = fully_fused
    config['n_neurons'] = n_neurons
    config['n_hidden'] = n_hidden
    config['output_activation'] = output_act
    config['output_activation'] = hidden_act
    return config

def get_ADAM(lr):
	config = adam_config
	config['learning_rate'] = lr
	return config