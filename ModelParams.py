# This class contains a list of dictionaries of model parameters

alzheimers_param_list = [
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 16,
             "bias": 0.00,
             "n_inputs": 36100},
            {"activation": "relu",
             "bias": 0.00,
             "n_outputs": 16},
            {"activation": "sigmoid",
             "n_outputs": 4}
        ]
    }]
alzheimers_param_list_standoff = [{
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
        },
        "layers": [
            {"activation": "relu",
             "bias": 0.001,
             "n_inputs": 36100,
             "n_outputs": 32},
            {"activation": "relu",
             "bias": 0.001,
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
             "bias": False,
             "n_outputs": 256,
             "n_inputs": 36100},
            {"activation": "relu",
             "bias": False,
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
             "bias": False,
             "n_outputs": 256,
             "n_inputs": 36100},
            {"activation": "relu",
             "bias": False,
             "n_outputs": 64},
            {"activation": "relu",
             "bias": False,
             "n_outputs": 32},
            {"activation": "sigmoid",
             "bias": False,
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

mnist_param_list = [
    {
        "network": {
            "learning_rate": "inverse_batch_accuracy",
            "lrn_rate_modifier": 0.1,
            "labels": ["0","1","2","3","4","5","6","7","8","9"]
        },
        "layers": [
            {"activation": "sigmoid",
             "n_outputs": 16,
             "n_inputs": 784},
            {"activation": "sigmoid",
             "n_outputs": 10}
        ]
    }]

mnist_param_list_standoff =[{
        "network": {
            "learning_rate": "inverse_batch_accuracy",
            "lrn_rate_modifier": 0.01,
            "labels": ["0","1","2","3","4","5","6","7","8","9"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 256,
             "n_inputs": 784},
            {"activation": "relu",
             "n_outputs": 64},
            {"activation": "sigmoid",
             "n_outputs": 10}
        ]
    },
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["0","1","2","3","4","5","6","7","8","9"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 256,
             "n_inputs": 784},
            {"activation": "relu",
             "n_outputs": 64},
            {"activation": "relu",
             "n_outputs": 32},
            {"activation": "sigmoid",
             "n_outputs": 10}
        ]
    },
    {
        "network": {
            "learning_rate": "constant",
            "lrn_rate_modifier": 0.01,
            "labels": ["0","1","2","3","4","5","6","7","8","9"]
        },
        "layers": [
            {"activation": "relu",
             "n_outputs": 256,
             "n_inputs": 784},
            {"activation": "leaky_relu",
             "n_outputs": 64},
            {"activation": "relu",
             "n_outputs": 32},
            {"activation": "sigmoid",
             "n_outputs": 10}
        ]
    }
]
