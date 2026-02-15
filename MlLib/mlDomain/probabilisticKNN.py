
class ProbabilisticKNN():
    metadata = {
        "name": "Probabilistic KNN",
        "description": "Probabilistic KNN skeleton for prior/likelihood/posterior-based classification."
    }
    # TODO: review metadata (auto-generated)
    
    ''' 
        how does ProbabilisticKNN work?
        ProbabilisticKNN is broken down into three components
        first a prior to give a base line for what distribution the probabilities of classifications already fall under
        second a function for inference that needs to be trained using a "nearest neighbors function" see classes
        third a posterior that combines the learned likely hoods and the prior
    '''
    
    def __init__(self, prior=None):
        self.prior = prior
        
    '''
        what functionality do I need?
        uno need to get the data in
        dos need to learn da data
        tres to learn da data we need the distance function
        quatro to keep learning the data we need to be able to update the probabilities with the distance function
        cinco need to be able to construct the posterior and return
        seis need to be able to evaluate the modal 
    '''
