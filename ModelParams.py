# This class contains a list of dictionaries of model parameters

param_list = [
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 16,
             "n_inputs": 36100},
            {"activation": "relu",
             "n_outputs": 16},
            {"activation": "sigmoid",
             "n_outputs": 4}
        ]
    },
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 256,
             "n_inputs": 36100},
            {"activation": "relu",
             "n_outputs": 64},
            {"activation": "sigmoid",
             "n_outputs": 4}
        ]
    },
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 256,
             "n_inputs": 36100},
            {"activation": "relu",
             "n_outputs": 64},
            {"activation": "relu",
             "n_outputs": 32},
            {"activation": "sigmoid",
             "n_outputs": 4}
        ]
    },
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 256,
             "n_inputs": 36100},
            {"activation": "leaky_relu",
             "n_outputs": 64},
            {"activation": "relu",
             "n_outputs": 32},
            {"activation": "sigmoid",
             "n_outputs": 4}
        ]
    }

]
