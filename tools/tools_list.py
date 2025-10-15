from . import tool_library
tools = None

def _ensure_tools():
    """Lazily instantiate the XAITools object to avoid heavy model loading at import time."""
    global tools
    if tools is None:
        tools = tool_library.XAITools()
    return tools
def get_prediction_from_classifier(instance: str) -> str:
    """
    Gets a prediction from the classifier for a given instance.
    
    Use this tool when:
    - User asks "What does the model predict for this input?"
    - User wants to know the model's output for a specific instance
    - User needs the predicted class or label
    
    This tool provides:
    - The predicted class label (e.g., "positive", "negative")
    - Confidence score if available
    
    Args:
        instance: The text instance to get a prediction for (e.g., "The movie was great")
        
    Returns:
        str: The predicted label/class for the instance
        
    Example:
        >>> get_prediction_from_classifier("This movie was amazing!")
        "positive"
    """
    tools = _ensure_tools()
    predicted_label, _ = tools.predict(instance)
    return predicted_label
def get_important_features(instance: str) -> dict:
    """
    Gets important features (words/tokens) contributing to the model's prediction, explaining why the model made this prediction.
    
    Use this tool when:
    - User wants to understand why the model made a certain prediction
    - User wants to understand which words influenced the model's decision
    - User needs a list of key tokens with importance scores
    Types of Questions this tool can answer:
    - What features does the system consider?
    - What features of this instance lead to the system's prediction?
    - What are the top features it uses?
    - How is this instance given this prediction?
    - Why is this instance given this prediction?
    This tool provides:
    - A list of important words/tokens for the current prediction
    - Importance scores indicating contribution to the prediction
    
    Args:
        instance: The text instance to analyze (e.g., "The movie was great")
    """
    # return {"great": 0.8, "movie": 0.6}  # dummy important words with scores
    tools = _ensure_tools()
    feature_attributions = tools.get_feature_attributions_shap(instance)
    # print("get_important_features called with instance:", instance, "and prediction_label:", prediction_label)
    return feature_attributions
def get_counterfactual_explanation(instance: str, target_label: str = None) -> list:

    """
    Generates counterfactual explanations showing what minimal changes to the instance
    would flip the prediction to a different class.
    
    Use this tool when:
    - User asks "What would need to change to get a different prediction?"
    - User asks "Why this prediction instead of another?"
    - User wants to understand decision boundaries
    
    This tool CAN:
    - Show minimal edits needed to change the prediction
    - Answer contrastive questions (why X not Y?)
    - Reveal decision boundaries
    
    This tool provides:
    - Modified version of the instance with flipped prediction
    - Specific changes made (words added/removed/replaced)
    - Distance/similarity to original instance
    
    Args:
        instance: The original instance to generate counterfactuals for
        target_label: Optional desired class (if None, finds nearest alternative)
        
    Returns:
        dict: Contains counterfactual text, changes made, and new prediction
        
    Example:
        >>> get_counterfactual_explanation("The movie was great", target_class="negative")
        {
            "original": "The movie was great",
            "counterfactual": "The movie was terrible",
            "changes": [{"from": "great", "to": "terrible"}],
            "original_prediction": "positive",
            "counterfactual_prediction": "negative"
        }
    """
    tools = _ensure_tools()
    counterfactuals = tools.get_counterfactual_explanation(instance, target_label)
    return counterfactuals
def get_multiple_counterfactuals(instance: str, target_label: str = None, num_counterfactuals: int = 3) -> list:
    """
    Generates multiple counterfactual explanations for a given instance and target label.

    Use this tool when
    - User asks "What are different ways to change the prediction?"
    - User wants to explore various minimal changes to flip the prediction
    - User needs a range of counterfactual examples
    This tool provides:
    - A list of counterfactual instances
    - Specific changes made for each counterfactual
    Args:
        instance: The original instance to generate counterfactuals for (e.g., "The movie was great")
        target_label: The desired target label to flip the prediction to (e.g., "negative")
        num_counterfactuals: Number of counterfactual examples to generate (default is 3)
    Returns:
        list: A list of counterfactual explanations with details
    """
    tools = _ensure_tools()
    counterfactuals = tools.get_multiple_counterfactuals(instance, target_label, num_counterfactuals)
    return counterfactuals
def get_feature_attribution(instance: str, feature: str, prediction_label: str = None) -> dict:
    """
    Gets attribution of a specific feature for the model's prediction on a given instance.

    Use this tool when:
    - User asks "What is the importance of feature X for this prediction?"
    - User wants to understand how a specific feature impacts the model's decision
    - User needs detailed insights into feature contributions

    This tool provides:
    - Attribution score for the specified feature
    - Contextual information about the instance and prediction

    Args:
        instance: The text instance to analyze (e.g., "The movie was great")
        feature: The specific feature to get attribution for (e.g., "great")
        prediction_label: Optional target label to explain (if None, explains predicted class)

    Returns:
        dict: Contains attribution score and related information
    """
    tools = _ensure_tools()
    feature_attribution = tools.get_feature_attribution(instance, feature, prediction_label)
    return feature_attribution

def get_classifier_performance() -> dict:
    """
    Gets overall performance metrics of the classifier model.

    Use this tool when:
    - User asks "How well does the model perform?"
    - User wants to understand model accuracy, precision, recall, F1-score
    - User needs insights into model strengths and weaknesses

    This tool provides:
    - Key performance metrics (accuracy, precision, recall, F1-score)
    - Confusion matrix if applicable

    Returns:
        dict: Contains performance metrics and related information
    """
    # return {"accuracy": 0.85, "precision": 0.80, "recall": 0.75, "f1_score": 0.77}  # dummy performance metrics
    tools = _ensure_tools()
    performance_metrics = tools.get_classifier_performance()
    return performance_metrics

def get_classifier_metadata() -> dict:
    """
    Gets metadata information about the classifier model.

    Use this tool when:
    - User asks "What kind of model is being used?"
    - User wants to understand model architecture, training data, and version
    - User needs insights into model characteristics

    This tool provides:
    - Model type (e.g., logistic regression, neural network)
    - Training dataset details
    - Model version and update history

    Returns:
        dict: Contains metadata information about the model
    """
    tools = _ensure_tools()
    metadata = tools.get_model_information("all")
    return metadata
def get_system_information() -> dict:
    """
    Gets information about the system's capabilities and limitations.

    Use this tool when:
    - User asks "What can the system do?"
    - User wants to understand system scope and limitations
    - User needs insights into system features

    This tool provides:
    - Overview of system functionalities
    - Known limitations and constraints
    - Future improvement plans

    Returns:
        dict: Contains information about the system's capabilities and limitations
    """
    tools = _ensure_tools()
    system_info = tools.get_system_information()
    return system_info
def get_data_information() -> dict:
    """
    Gets information about the data used to train and evaluate the model.

    Use this tool when:
    - User asks "What data was used to train the model?"
    - User wants to understand data sources, collection methods, and preprocessing
    - User needs insights into dataset characteristics

    This tool provides:
    - Data sources and collection methods
    - Preprocessing steps applied to the data
    - Dataset statistics (size, diversity, etc.)

    Returns:
        dict: Contains information about the data used for training and evaluation
    """
    tools = _ensure_tools()
    data_info = tools.get_data_information()
    return data_info
def unknown() -> str:
    """
    Returns a response indicating that the answer is unknown.

    Use this tool when:
    - The system does not have enough information to answer the user's question
    - The question is outside the scope of the available tools and knowledge

    This tool provides:
    - A polite response indicating that the answer is unknown

    Returns:
        str: A message indicating that the answer is unknown
    """
    return "I'm sorry, I don't have enough information to answer that question."