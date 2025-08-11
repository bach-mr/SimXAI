from src.information_extraction.extraction import intent_recognition_and_extraction

USER_INPUT_SUPPORTED_OPERATIONS = ["cfe", "SHAP"]


def intent_recognition_pipeline(model, tokenizer, user_input, option=1):
    if option == 1:
        # first check the intent and then extract the user input from the question
        intent = intent_recognition_and_extraction(model, tokenizer, None, "intention", user_input)

        flag = False
        for ops in USER_INPUT_SUPPORTED_OPERATIONS:
            if ops in intent:
                flag = True
        if flag:
            inputs = intent_recognition_and_extraction(model, tokenizer, "default", "extraction", user_input)
        else:
            inputs = ""
    else:
        # first check whether there is any user input in the question
        inputs = intent_recognition_and_extraction(model, tokenizer, "default", "extraction", user_input)

        if inputs == "":
            # we should then exclude those operations which support use input
            # TODO
            pass

        intent = intent_recognition_and_extraction(model, tokenizer, None, "intention", user_input)

    return intent, inputs
