class PrizePredictor:
    """Predict prize levels based on three numbers and provide explanations."""
    
    def __init__(self, ):

        """
        Initialize with three numbers (0-9).

        Args:
            num1: First number (0-9)
            num2: Second number (0-9)
            num3: Third number (0-9)
        """
        self.label = "no prize"

    def _parse_instance(self, instance: str):
        parts = [part.strip() for part in instance.split(",")]
        if len(parts) != 3 or any(not part.isdigit() or not 0 <= int(part) <= 9 for part in parts):
            return "Please provide three digits between 0 and 9 in the format x1,x2,x3."
        return tuple(int(part) for part in parts)

    def get_prediction(self, instance) -> str:
        parsed = self._parse_instance(instance)
        if isinstance(parsed, str):
            return parsed

        self.num1, self.num2, self.num3 = parsed
        """
        Check prize based on three numbers.

        Returns:
            str: Prize level ('first prize', 'second prize', 'third prize', or 'no prize')
        """
        if self.num1 == self.num2 == self.num3:
            self.label = "first prize"

        elif self.num1 == self.num3:
            self.label = "second prize"

        elif self.num1 > self.num2:
            self.label = "third prize"
        else:
            self.label = "no prize"

        return self.label

    def get_important_features(self, label: str = None) -> str:
        
        """Get important features based on the prize label."""
        features = {
            "first prize": "Number 1 and 2 and 3 are the same",
            "second prize": "Number 1 and 3 are the same and 2 is different",
            "third prize": "Number 1 is greater than Number 2",
            "no prize": "Number 1, 2, and 3 are all different"
        }
        if label:
            return features.get(label, "")
        return features.get(self.label, "")

    def get_counterfactuals(self, instances = None, target_label: str = None, number_of_iterations: int = 1) -> list:
        """Get counterfactual explanations based on the prize label."""
        if instances:
            numbers = instances.replace(" ", "").split(",")
            num1 = int(numbers[0])
            num2 = int(numbers[1])
            num3 = int(numbers[2])
        else:
            num1, num2, num3 = self.num1, self.num2, self.num3

        if target_label == "third prize":
            if num2 < 9:
                num1 = num2 + 1
            elif num1 > 0:
                num2 = num1 - 1

        if target_label == "no prize":
            if num1 == num2:
                num2 = num2 + 1 if num2 < 9 else num1 - 1
            if num2 == num3:
                num3 = num3 + 1 if num3 < 9 else num2 - 1
            if num1 == num3:
                num3 = num3 + 1 if num3 < 9 else num1 - 1
        if target_label == "second prize":
            # Make first and third same and different from second but with minimal change
            candidates = []

    # Option 1: make both first and third equal to num1
            new1, new2, new3 = num1, num2, num1
            if new2 == new1:
                new2 = (new1 + 1) % 10
            diff1 = abs(new1 - num1) + abs(new2 - num2) + abs(new3 - num3)
            candidates.append((diff1, (new1, new2, new3)))

            # Option 2: make both first and third equal to num3
            new1, new2, new3 = num3, num2, num3
            if new2 == new1:
                new2 = (new1 + 1) % 10
            diff2 = abs(new1 - num1) + abs(new2 - num2) + abs(new3 - num3)
            candidates.append((diff2, (new1, new2, new3)))

            # Option 3: make both first and third equal to a middle value (if that’s cheaper)
            mid = round((num1 + num3) / 2)
            new1, new2, new3 = mid, num2, mid
            if new2 == new1:
                new2 = (new1 + 1) % 10
            diff3 = abs(new1 - num1) + abs(new2 - num2) + abs(new3 - num3)
            candidates.append((diff3, (new1, new2, new3)))

            # Choose minimal total change
            best = min(candidates, key=lambda x: x[0])
            num1, num2, num3 = best[1]
        if target_label == "first prize":
            num2 = num1
            num3 = num1

        counterfactuals = {
            "first prize": [f"number 1: {num1}, number 2: {num1}, number 3: {num1}"],
            "second prize": [f"number 1: {num1}, number 2: {num2}, number 3: {num1}"],
            "third prize": [f"number 1: {num1}, number 2: {num2}, number 3: {num3}"],
            "no prize": [f"number 1: {num1 if num1 < num2 else num2}, number 2: {num2 if num1 < num2 else num1}, number 3: {num3}"]
        }
        if target_label:
            return counterfactuals.get(target_label, [])
        
        else:
            # get a different target label:
            for label in counterfactuals.keys():
                if label != self.label:
                    return counterfactuals.get(label, [])

    def get_global_explanation(self) -> str:
        """Get global explanation of the prize prediction logic."""
        explanation = (
            "The prize prediction is based on the following rules:\n"
            "- First prize: All three numbers are the same.\n"
            "- Second prize: The first and third numbers are the same.\n"
            "- Third prize: The first number is greater than the second number.\n"
            "- No prize: All numbers are different."
        )
        return explanation
    def get_instance_with_same_prediction(self, instances = None, label: str = None) -> str:
        """Get an instance that would yield the same prize prediction."""
        if instances:
            label = self.get_prediction(instances)
        if instances:
            num1, num2, num3 = map(int, instances.replace(" ", "").split(","))
        else:
            num1, num2, num3 = self.num1, self.num2, self.num3
        
        # Modify to ensure different instance
        if label == "first prize":
            # Keep all same, but change value if possible
            new_val = (num1 + 1) % 10
            prizes = {"first prize": f"number 1: {new_val}, number 2: {new_val}, number 3: {new_val}"}
        elif label == "second prize":
            # Keep 1st and 3rd same, change 2nd
            new_num2 = (num2 + 1) % 10
            prizes = {"second prize": f"number 1: {num1}, number 2: {new_num2}, number 3: {num1}"}
        elif label == "third prize":
            # Keep num1 > num2, but change values
            new_num1 = (num1 + 1) % 10 if num1 < 9 else (num1 - 1) if num1 > 0 else num1
            new_num2 = num2 if new_num1 > num2 else (num2 - 1) if num2 > 0 else num2
            prizes = {"third prize": f"number 1: {new_num1}, number 2: {new_num2}, number 3: {num3}"}
        else:  # no prize
            # Change at least one number
            new_num2 = (num2 + 1) % 10
            prizes = {"no prize": f"number 1: {num1 if num1 < new_num2 else new_num2}, number 2: {new_num2 if new_num2 > num1 else num1}, number 3: {num3}"}
        if label:
            return prizes.get(label, "")
        return prizes.get(self.label, "")
    def get_model_performance(self, metric) -> str:
        performance = {"accuracy": "100%", "precision": "100%", "recall": "100%"}
        """Get information about the model's behavior."""
        return f"The model has {metric} of {performance.get(metric, 'N/A')} on all possible combinations of three numbers (0-9) based on the defined prize rules."
    def get_data_information(self) -> str:
        """Get information about the data used by the model."""
        return "The model does not use any training data; it operates based on predefined rules."
    def get_output_information(self) -> str:
        """Get information about the model's output."""
        return "The model outputs one of four prize levels: 'first prize', 'second prize', 'third prize', or 'no prize' based on the input numbers."
    def get_model_information(self) -> str:
        """Get general information about the model."""
        return "This is a rule-based prize prediction model that determines prize levels based on the relationships between three input numbers (0-9)."