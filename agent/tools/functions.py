from tools.tool_numbers import *
xai = PrizePredictor()
# def get_prediction(num1, num2, num3):
#     """
#     Call this whenever you neet to get model prediction on the given instance, for example, user asked "What is the prediction for this instance?"
#     Get prize based on three numbers (0-9).
    
#     Args:
#         num1: First number (0-9)
#         num2: Second number (0-9)
#         num3: Third number (0-9)
    
#     Returns:
#         str: Prize level ('first prize', 'second prize', 'third prize', or 'no prize')
#     """
#     return xai.get_prediction(num1, num2, num3)
def get_prediction(instance: str) -> str:
    """
    Call this whenever you neet to get model prediction on the given instance, for example, user asked "What is the prediction for this instance?"
    
    Args:
        instance: Input instance as a string
    
    Returns:
        str: label of the classifier for the instance
    """
    return xai.get_prediction(instance)
def get_global_explanation() -> dict:
    """
    Call this whenever you need a global understanding of the model's logic or decision-making process, for example, user asked "How does the model generally work?"

    Output:
        A human-readable explanation of how the model generally works,
        e.g., which features are important globally or what the overall structure is.
    """
    return xai.get_global_explanation()
def get_local_explanation(prediction_label: str = None) -> dict:
    """
    Call this whenever you need reason for the prediction, such as features contributed most to a specific prediction made by the model, for example, user asked "Why did the model predict this for the given instance?"

    Args:
        instance: The specific input instance to explain.
        prediction_label: Optional target label to explain (if None, explains predicted class)

    Output:
        Feature-level attribution or explanation (e.g., SHAP, LIME, or attention weights).
    """
    return xai.get_important_features(prediction_label)
def explain_why_not(instance: str, expected_label: str = None) -> dict:
    """
    Call this whenever you need to explains why the model did NOT predict the expected or desired outcome but the current outcome. For example, user asked "Why didn't the model predict X for this instance?"
    The tool identifies features or patterns that led to the actual prediction then proposes changes needed to achieve the desired prediction.
    
    Args:
        instance: The input instance.
        expected_label: The expected or desired label.

    Output:
        Explanation of which features or patterns prevented the desired prediction.
    """
    why = xai.get_important_features()
    if expected_label:
        cf = xai.get_counterfactuals(instance, expected_label.lower())
    else:
        cf = xai.get_counterfactuals(instance)
    return "The reason for the current prediction is: " + why + ". To achieve the expected prediction, consider the following changes: " + ", ".join(cf)

def explain_how_to_be_that(instance: str, target_label: str = None, number_of_iterations: int = 1) -> str:
    """
    Call this whenever you need to explains how to modify the instance minimally to achieve the target prediction (counterfactual). For example, user asked "How can I change this instance to get prediction X?"
    The tool suggests changes needed to flip the current prediction to the desired one.
    
    Args:
        instance: The input instance.
        target_label: (Optional) The desired prediction or class. If None, suggests changes to flip the current prediction.
        number_of_iterations: Number of counterfactual suggestions to generate.
    
    Output:
        Counterfactual changes (e.g., "increase feature X by 0.2").
    """
    return xai.get_counterfactuals(instance, target_label.lower(), number_of_iterations)


def explain_how_to_still_be_this(instance: str = None) -> str:
    """
    Call this whenever you need to identify what changes can be made while keeping the same prediction. For example, user asked "What changes can I make to this instance without changing its prediction?" 
    The tool finds the range of feature values or perturbations that do not alter the model's output.
    
    Args:
        instance: (Optional) The input instance. If None, uses the current instance.
    
    Output:
        Range of feature values or perturbations that do not change the prediction.
    """
    return xai.get_instance_with_same_prediction(instance)


def explain_what_if(instance: str) -> str:
    """
    Call this whenever you need to know how the prediction changes if input features are modified.
    The tool answers "what if" scenarios by simulating changes to the input and observing the effect on the prediction.
    
    Args:
        instance: The original instance.
    
    Output:
        The new prediction and an explanation of the effect of the change.
    """
    return xai.get_prediction(instance)
def get_model_performance(metric) -> str:
    """
    Call this whenever you need  to access overall performance of the model. For example, user asked "How well does the model perform?"
    The tool provides metrics like accuracy, F1-score, precision, and insights into failure cases.
    Answer questions about accuracy, F1-score, and where the model struggles, how accurate is it?
    
    
    Output:
        Metrics (accuracy, F1, etc.) and insight into which cases the model struggles with.
    """
    return xai.get_model_performance(metric)
def get_data_information() -> str:
    """
    Call this whenever you need to access the data information used to train and evaluate the model. For example, user asked "What data was used to train the model?"
    The tool provides details like summary statistics, feature distributions, class balance, and potential biases.
    
    Output:
        Summary statistics, feature distributions, class balance, and potential biases in the data.
    """
    return xai.get_data_information()
def get_model_information() -> str:
    """
    Call this whenever you need to provide information about the model's design, architecture, training process, limitations, and strengths. For example, user asked "Tell me about when the model fails?"
    
    The tool access the model information including limitations, strengths, limitation, design, architecture, the whole system
   

    Output:
        Information about the model's design, training process, and performance.
    """
    return xai.get_model_information()

def get_output_information() -> str:
    """
    Call this whenver you need to know any information about the model’s output such as type, purpose or how to interpret it in the whole system. For example, user asked "What does the model output mean?"
   
    Output:
        Guidance on interpretation, usage, or next actions.
    """
    return xai.get_output_information()

