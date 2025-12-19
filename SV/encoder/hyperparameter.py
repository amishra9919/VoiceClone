from sklearn.metrics import accuracy_score
import optuna

# learning rate
# Optimizer

# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init) 




def objective(trial):
    # model = 
    optimzier_name = trial.suggest_categorial("optimizer",["Adam","RMSprop","SGD"])
    lr = trial.suggest_float("lr",1e-5,1e-1, log=True)
    optimizer = getattr(optim,optimzier_name)(model.parmeters(),lr=lr)
    epochs = 100
    for epoch in range(epochs):
        trial.report(accuracy,epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy 

    